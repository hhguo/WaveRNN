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
import soundfile as sf
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from hparams import hparams
from models import *
from utils.audio import *
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


class AcousticSliceDataSource(FileDataSource):
    def __init__(self, data_dir_dict, file_list, ext='.npy'):
        self.storage = dict()
        pad = hparams.pad
        seq_frames = hparams.seq_len
        hop_length = int(hparams.frame_shift_ms * hparams.sample_rate / 1000)
        self.pad = pad
        self.seq_frames = seq_frames
        self.hop_length = hop_length

        with open(file_list) as fin:
            file_ids = [line.strip() for line in fin.readlines()]

        log('Collect Mel spectrum dataset...')
        mels = self.load_dataset(data_dir_dict['mel'], file_ids, '.npy')

        log('Collect Waveform dataset...')
        wavs = self.load_dataset(data_dir_dict['wav'], file_ids, '.npy')

        log('Reshape Mel and Waveform...')
        mel_block, wav_block = [], []
        self.indices = []
        total_frames = 0
        for i in tqdm(range(len(file_ids))):
            mel, wav = mels[i], wavs[i]
            wav_length = int(min(mel.shape[0] * hop_length, wav.shape[0]))
            mel, wav = mel[: wav_length // hop_length], wav[: wav_length]
            nb_chunks = (mel.shape[0] -  2 * pad) // seq_frames
            mel = mel[: nb_chunks * seq_frames + 2 * pad]
            wav = wav[: mel.shape[0] * hop_length]
            mel_block.append(mel)
            wav_block.append(wav)
            self.indices += range(total_frames, total_frames + mel.shape[0] - 2 * pad, seq_frames)
            total_frames += mel.shape[0]
        
        log('Concatenate all chunks togethers...')
        mel_block = self.concatenate(mel_block)
        wav_block = self.concatenate(wav_block)
        self.data_block = {'mel': mel_block, 'wav': wav_block}

    def collect_files(self):
        return self.indices

    def collect_features(self, index):
        mel = self.data_block['mel'][index: index + 2 * self.pad + self.seq_frames]
        begin_sample = (index + self.pad) * self.hop_length - 1
        end_sample = (index + self.pad + self.seq_frames) * self.hop_length
        wav = self.data_block['wav'][begin_sample: end_sample]
        return [mel, wav]

    def load_dataset(self, data_dir, file_ids, ext='.npy'):
        dataset = []
        if data_dir[-4: ] == '.zip':
            zfile = zipfile.ZipFile(data_dir)
            for filename in tqdm(list(zfile.namelist())):
                data_id, data_ext = splitext(os.path.split(filename)[-1])
                if data_ext == ext and data_id in file_ids:
                    zip_data = zfile.open(filename, 'r')
                    raw_data = io.BytesIO(zip_data.read())
                    if ext == '.npy':
                        dataset.append(np.load(raw_data))
                    elif ext == '.wav':
                        dataset.append(sf.read(raw_data))
        else:
            for data_id in tqdm(file_ids):
                if ext == '.npy':
                    dataset.append(np.load(
                        os.path.join(data_dir, data_id + ext)))
                elif ext == '.wav':
                    dataset.append(sf.read(
                        os.path.join(data_dir, data_id + ext)))
        return dataset
    
    def concatenate(self, data_list):
        shape = list(data_list[0].shape)
        shape[0] = sum([x.shape[0] for x in data_list])
        data = np.zeros(shape, dtype=data_list[0].dtype)
        index = 0
        for x in tqdm(data_list):
            data[index: index + x.shape[0]] = x
            index += x.shape[0]
        return data

class PyTorchDataset(object):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, idx):
        return self.X[idx]

    def __len__(self):
        return len(self.X)


def collate_fn(batch):
    mels = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    mels = torch.FloatTensor(np.stack(mels).astype(np.float32))
    labels = torch.LongTensor(np.stack(labels).astype(np.int32))

    bits = 16 if hparams.mode != 'RAW' else hparams.bits
    x = label_2_float(labels[:, : -1].float(), bits)
    y = labels[:, 1: ]
    if hparams.mode != 'RAW':
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
            elif hparams.mode == 'SG':
                y = y.float()
                nll_loss = single_gaussian_loss(outputs, y)
                pow_loss = power_loss(sample_from_single_gaussian(outputs), y)
                loss = nll_loss + 10 * pow_loss
                writer.add_scalar("nll_loss", float(nll_loss), global_step)
                writer.add_scalar("power_loss", float(pow_loss), global_step)
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
            log("loss: {:.3f}".format(float(loss)), end='\r')
            writer.add_scalar("loss", float(loss), global_step)
            writer.add_scalar("gradient norm", grad_norm, global_step)
            global_step += 1

        averaged_loss = running_loss / (len(data_loader))
        log("loss ({} epoch): {:.3f}".format(global_epoch, averaged_loss))
        writer.add_scalar("epoch_loss", averaged_loss, global_epoch)
        global_epoch += 1


def main(input_dir, output_dir, file_list, checkpoint_dir,
         checkpoint_path=None, reset_optimizer=False, config=''):
    # Override hyper parameters
    hparams.parse(config)
    os.makedirs(checkpoint_dir, exist_ok=True)
    Model = get_model(hparams)

    # Input dataset definitions
    X = FileSourceDataset(AcousticSliceDataSource(
        {'mel': input_dir, 'wav': output_dir},
        file_list))
    dataset = PyTorchDataset(X)

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
                  hop_length=int(hparams.frame_shift_ms * hparams.sample_rate),
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
