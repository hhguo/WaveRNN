"""Trainining script for Tacotron speech synthesis model.

usage: train.py [options]

options:
    --data-root=<dir>         Directory contains preprocessed features. [default: corpus]
    --checkpoint-dir=<dir>    Directory where to save model checkpoints [default: checkpoints].
    --checkpoint-path=<name>  Restore model from checkpoint path if given.
    --reset-optimizer         Don't restore optimizer from checkpoint
    --hparams=<params>        Hyper parameters [default: ].
    --input=<name>            Directory contains input data [default: mel].
    --output=<name>           Directory contains output data [default: quant].
    --list=<name>             Path of training list [default: train.list].

    -h, --help                Show this help message and exit
"""

import io
import os
import sys
import tensorboard_logger
import torch
import zipfile

import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from docopt import docopt
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from os.path import dirname, join, expanduser
from tensorboard_logger import log_value
from tensorboardX import SummaryWriter
from termcolor import colored
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils import data as data_utils
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from hparams import hparams, hparams_debug_string
from hparams import hparams as hp
from models.fatchord_wavernn import *
from utils.distribution import *
from utils.infolog import *


DATA_ROOT = './'
use_cuda = torch.cuda.is_available()
cudnn.benchmark = False if use_cuda else True


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
  checkpoint_path = os.path.join(
      checkpoint_dir, "checkpoint_step{}.pth".format(step))
  torch.save({
      "state_dict": model.state_dict(),
      "optimizer": optimizer.state_dict(),
      "global_step": step,
      "global_epoch": epoch,
  }, checkpoint_path)
  print("Saved checkpoint:", checkpoint_path)
  os.system("echo 'checkpoint_step{}.pth' > {}".format(
      step, os.path.join(checkpoint_dir, 'checkpoints')))


class _NPYDataSource(FileDataSource):
  def __init__(self, name, list_file, use_zip=False):
    self._name = name
    self._list_file = list_file
    self._ftype = name.split('/')[-1]
    self._use_zip = use_zip
    self._storage = dict()
    if use_zip:
      zfile = zipfile.ZipFile(os.path.join(DATA_ROOT, name + '.zip'))
      for filename in tqdm(list(zfile.namelist())):
        if '.npy' in filename:
          zip_npy = zfile.open(filename, 'r')
          raw_npy = io.BytesIO(zip_npy.read())
          self._storage[filename] = np.load(raw_npy)
            

  def collect_files(self):
    with open(self._list_file) as fin:
      lines = [line.strip() for line in fin.readlines()]
      if self._use_zip:
        paths = [os.path.join(self._ftype, 
                 line + '.npy') for line in lines]
      else:
        paths = [os.path.join(DATA_ROOT, self._name, 
                 line + '.npy') for line in lines]
      return paths

  def collect_features(self, path):
    if self._use_zip:
      data = self._storage[path]
    else:
      data = np.load(path)
    return data


class AcousticDataSource(_NPYDataSource):
  def __init__(self, name, list_file, use_zip=False):
    super(AcousticDataSource, self).__init__(
      name, list_file, use_zip)


class PyTorchDataset(object):
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y

  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]

  def __len__(self):
    return len(self.X)


def collate_fn(batch):
    mel_win = hp.seq_len // hp.hop_length + 2 * hp.pad
    max_offsets = [x[0].shape[0] - (mel_win + 2 * hp.pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.pad) * hp.hop_length for offset in mel_offsets]
    
    mels = [x[0][mel_offsets[i]: mel_offsets[i] + mel_win]\
            for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]: sig_offsets[i] + hp.seq_len + 1]\
              for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.FloatTensor(mels)
    labels = torch.LongTensor(labels)

    x = label_2_float(labels[:, : hp.seq_len].float(), hp.bits)

    y = labels[:, 1: ]

    return x, y, mels


def train(model, optimizer, data_loader, global_epoch, global_step,
          init_lr, checkpoint_dir, checkpoint_interval,
          nepochs, clip_thresh):
  # Setting log
  writer = SummaryWriter(checkpoint_dir)
  init(os.path.join(checkpoint_dir, 'train.log'))


  if use_cuda:
    model = model.cuda()
  model.train()

  criterion = nn.CrossEntropyLoss()
  
  for param_group in optimizer.param_groups:
      param_group['lr'] = init_lr
  
  while global_epoch < nepochs:
    running_loss = 0.
    for step, (x, y, m) in tqdm(enumerate(data_loader)):
      
      optimizer.zero_grad()

      # Feed data
      x, y, m = Variable(x), Variable(y), Variable(m)
      if use_cuda:
        x, y, m = x.cuda(), y.cuda(), m.cuda()
     
      outputs = model(x, m.transpose(1, 2))
      
      y = y.unsqueeze(-1)
      
      if model.mode == 'MOL':
        y = y.float()
        loss = discretized_mix_logistic_loss(outputs, y)
      else:
        outputs = outputs.transpose(1, 2).unsqueeze(-1)
        loss = F.cross_entropy(outputs, y)
      running_loss += loss.item()

      # save checkpoint
      if global_step > 0 and global_step % checkpoint_interval == 0:
        save_checkpoint(model, optimizer, global_step, 
                        checkpoint_dir, global_epoch)

      # Update
      loss.backward()
      grad_norm = torch.nn.utils.clip_grad_norm(
        model.parameters(), clip_thresh)
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


if __name__ == "__main__":
  args = docopt(__doc__)
  print("Command line args:\n", args)
  checkpoint_dir = args["--checkpoint-dir"]
  checkpoint_path = args["--checkpoint-path"]
  reset_optimizer = args["--reset-optimizer"]
  DATA_ROOT = args["--data-root"]
  input_dir = args["--input"]
  output_dir = args["--output"]
  id_list = os.path.join(DATA_ROOT, args["--list"])
  # Override hyper parameters
  hparams.parse(args["--hparams"])
  os.makedirs(checkpoint_dir, exist_ok=True)

  # Input dataset definitions
  X = FileSourceDataset(AcousticDataSource(input_dir, id_list, hp.use_zip))
  Y = FileSourceDataset(AcousticDataSource(output_dir, id_list, hp.use_zip))

  # Dataset and Dataloader setup
  dataset = PyTorchDataset(X, Y)

  # Collect function
  data_loader = data_utils.DataLoader(
    dataset, batch_size=hparams.batch_size,
    num_workers=hparams.num_workers, shuffle=True,
    sampler=None, drop_last=True,
    collate_fn=collate_fn, pin_memory=hparams.pin_memory)

  # Model
  model = Model(rnn_dims=hp.rnn_dims,
                fc_dims=hp.fc_dims,
                bits=hp.bits,
                pad=hp.pad,
                upsample_factors=hp.upsample_factors,
                feat_dims=hp.num_mels,
                compute_dims=hp.compute_dims,
                res_out_dims=hp.res_out_dims,
                res_blocks=hp.res_blocks,
                hop_length=hp.hop_length,
                sample_rate=hp.sample_rate,
                mode=hp.mode).cuda()

  optimizer = optim.Adam(model.parameters(),
    lr=hp.lr, betas=(hp.adam_beta1, hp.adam_beta2),
    weight_decay=hp.weight_decay)

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
    if reset_optimizer == False:
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
    try:
      global_step = checkpoint["global_step"]
      global_epoch = checkpoint["global_epoch"]
    except:
        # TODO
        pass

  # Setup tensorboard logger
  tensorboard_logger.configure(checkpoint_dir)
  print(hparams_debug_string())

  # Train!
  train(model, optimizer, data_loader, global_epoch, global_step,
        hp.lr, checkpoint_dir, hp.checkpoint_interval,
        hp.nepochs, hp.clip_thresh)
  
  print("Finished")
  sys.exit(0)
