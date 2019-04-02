import io
import os
import torch
import zipfile

import numpy as np

from nnmnkwii.datasets import FileDataSource
from os.path import join
from tqdm import tqdm

from hparams import hparams, hparams_debug_string
from hparams import hparams as hp
from utils import *


DATA_ROOT = './'


def _pad(seq, max_len, constant_values=0):
  return np.pad(seq, (0, max_len - len(seq)),
                mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, constant_values=0):
  x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
             mode="constant", constant_values=constant_values)
  return x


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
    max_offsets = [x[0].shape[-1] - (mel_win + 2 * hp.pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.pad) * hp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]:sig_offsets[i] + hp.seq_len + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.FloatTensor(mels)
    labels = torch.LongTensor(labels)

    x = label_2_float(labels[:, :hp.seq_len].float(), hp.bits)

    y = labels[:, 1:]

    return x, y, mels
