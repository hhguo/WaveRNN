from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from os.path import dirname, join, expanduser, splitext, split
from tensorboard_logger import log_value
from tensorboardX import SummaryWriter
from termcolor import colored
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils import data as data_utils
from tqdm import tqdm, trange

import fire
import io
import os
import sys
import tensorboard_logger
import torch
import zipfile

import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from hparams import hparams
from models.fatchord_wavernn import *
from utils.distribution import *
from utils.infolog import *

use_cuda = torch.cuda.is_available()
cudnn.benchmark = False if use_cuda else True


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = os.path.join(checkpoint_dir,
                                   "checkpoint_step{}.pth".format(step))
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)
    os.system("echo 'checkpoint_step{}.pth' > {}".format(
        step, os.path.join(checkpoint_dir, 'checkpoints')))


class LRDecaySchedule(object):
    def __init__(self, init_lr, warmup_steps=4000):
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps

    def __call__(self, steps):
        steps = steps + 1
        lr = self.init_lr * min(
             self.warmup_steps ** 0.5 * steps ** (-0.5),
             steps / self.warmup_steps)
        return lr


class _NPYDataSource(FileDataSource):
    def __init__(self, data_dir, file_list, ext='.npy'):
        self.storage = dict()
        with open(file_list) as fin:
            self.data_ids = [line.strip() for line in fin.readlines()]

        if data_dir[-4: ] == '.zip':
            zfile = zipfile.ZipFile(data_dir)
            for filename in tqdm(list(zfile.namelist())):
                data_id, data_ext = splitext(os.path.split(filename)[-1])
                if data_ext == ext and data_id in self.data_ids:
                    zip_npy = zfile.open(filename, 'r')
                    raw_npy = io.BytesIO(zip_npy.read())
                    self.storage[data_id] = np.load(raw_npy)
        else:
            for data_id in tqdm(self.data_ids):
                self.storage[data_id] = np.load(
                    os.path.join(data_dir, data_id + ext))

    def collect_files(self):
        return self.data_ids

    def collect_features(self, data_id):
        return self.storage[data_id]


class AcousticDataSource(_NPYDataSource):
    def __init__(self, data_dir, file_list, ext='.npy'):
        super(AcousticDataSource, self).__init__(data_dir, file_list, ext='.npy')


class PyTorchDataset(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


def collate_fn(batch):
    mel_win = hparams.seq_len // hparams.hop_length + 2 * hparams.pad
    max_offsets = [x[0].shape[0] - (mel_win + 2 * hparams.pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hparams.pad) * hparams.hop_length for offset in mel_offsets]

    mels = [x[0][mel_offsets[i]: mel_offsets[i] + mel_win]\
            for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]: sig_offsets[i] + hparams.seq_len + 1]\
              for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.FloatTensor(mels)
    labels = torch.LongTensor(labels)

    x = labels[:, :hparams.seq_len]
    y = labels[:, 1:]

    bits = 16 if hparams.mode == 'MOL' else hparams.bits

    x = label_2_float(x.float(), bits)
    if hparams.mode == 'MOL':
        y = label_2_float(y.float(), bits)

    return x, y, mels


def train(model, optimizer, data_loader, checkpoint_dir,
          global_epoch=0, global_step=0, init_lr=1e-3,
          checkpoint_interval=10000, nepochs=240, clip_thresh=1.0):
    # Setting log
    writer = SummaryWriter(checkpoint_dir)
    init(os.path.join(checkpoint_dir, 'train.log'))

    if use_cuda:
        model = model.cuda()
    model.train()

    lr_decay = LRDecaySchedule(init_lr)

    while global_epoch < nepochs:
        running_loss = 0.
        for step, (x, y, m) in tqdm(enumerate(data_loader)):

            optimizer.zero_grad()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay(global_step)

            # Feed data
            x, y, m = Variable(x), Variable(y), Variable(m)
            if use_cuda:
                x, y, m = x.cuda(), y.cuda(), m.cuda()

            outputs = model(x, m.transpose(1, 2))

            y = y.unsqueeze(-1)

            if hparams.mode == 'MOL':
                y = y.float()
                loss = discretized_mix_logistic_loss(outputs, y)
            else:
                outputs = outputs.transpose(1, 2).unsqueeze(-1)
                loss = F.cross_entropy(outputs, y)
            running_loss += loss.item()

            # save checkpoint
            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, global_step, checkpoint_dir,
                                global_epoch)

            # Update
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(),
                                                      clip_thresh)
            optimizer.step()

            # Logs
            log("loss: {}".format(float(loss)), end='\r')
            writer.add_scalar("loss", float(loss), global_step)
            writer.add_scalar("gradient norm", grad_norm, global_step)
            global_step += 1

        averaged_loss = running_loss / (len(data_loader))
        log("loss ({} epoch): {}".format(global_epoch, averaged_loss))
        writer.add_scalar("epoch_loss", averaged_loss, global_epoch)
        global_epoch += 1


def main(input_dir, output_dir, file_list, checkpoint_dir,
         checkpoint_path=None, reset_optimizer=False, config=''):
    # Override hyper parameters
    hparams.parse(config)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Input dataset definitions
    X = FileSourceDataset(AcousticDataSource(input_dir, file_list))
    Y = FileSourceDataset(AcousticDataSource(output_dir, file_list))

    # Dataset and Dataloader setup
    dataset = PyTorchDataset(X, Y)

    # Collect function
    data_loader = data_utils.DataLoader(dataset,
                                        batch_size=hparams.batch_size,
                                        num_workers=hparams.num_workers,
                                        shuffle=True,
                                        sampler=None,
                                        drop_last=True,
                                        collate_fn=collate_fn,
                                        pin_memory=hparams.pin_memory)

    # Model
    model = Model(rnn_dims=hparams.rnn_dims,
                  fc_dims=hparams.fc_dims,
                  bits=hparams.bits,
                  pad=hparams.pad,
                  upsample_factors=hparams.upsample_factors,
                  feat_dims=hparams.num_mels,
                  compute_dims=hparams.compute_dims,
                  res_out_dims=hparams.res_out_dims,
                  res_blocks=hparams.res_blocks,
                  hop_length=hparams.hop_length,
                  sample_rate=hparams.sample_rate,
                  mode=hparams.mode)

    if use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=hparams.lr)

    # Load checkpoint
    global_epoch, global_step = 0, 0
    if not checkpoint_path:
        if os.path.exists(os.path.join(checkpoint_dir, 'checkpoints')):
            with open(os.path.join(checkpoint_dir, 'checkpoints')) as fin:
                ckpt = fin.readline().strip()
            checkpoint_path = os.path.join(checkpoint_dir, ckpt)
    if checkpoint_path:
        print("Load checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if use_cuda:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except RuntimeError:
            print(colored('model file is uncompleted', 'yellow'))
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    # Setup tensorboard logger
    tensorboard_logger.configure(checkpoint_dir)
    print(hparams.hparams_debug_string())

    # Train!
    train(model, optimizer, data_loader, checkpoint_dir,
          global_epoch, global_step, hparams.lr,
          hparams.checkpoint_interval, hparams.nepochs, hparams.clip_thresh)

    print("Finished")
    sys.exit(0)


if __name__ == '__main__':
    fire.Fire(main)
    print("Done!")