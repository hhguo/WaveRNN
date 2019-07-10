import fire
import numpy as np
import os

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count
from tqdm import tqdm

from utils.audio import *
from hparams import hparams as hp

def convert_file(path):
  y = load_wav(path)
  mel = melspectrogram(y).T
  if hp.mode == 'RAW':
      quant = encode_mu_law(y, mu=2 ** hp.bits) if hp.mu_law else \
          float_2_label(y, bits=hp.bits)
  elif hp.voc_mode == 'MOL' :
      quant = float_2_label(y, bits=16)
  return mel.astype(np.float32), quant.astype(np.int64)


def _process_utterance(path, mel_dir, quant_dir):
  directory = os.path.split(path)[: -1]
  fid = os.path.split(path)[-1].split('.')[0]
  m, x = convert_file(path)
  np.save(f'{mel_dir}/{fid}.npy', m)
  np.save(f'{quant_dir}/{fid}.npy', x)
  return fid

def main(wav_dir, mel_dir, quant_dir):
  executor = ProcessPoolExecutor(max_workers=cpu_count())
  futures = []
  for filename in os.listdir(wav_dir):
    wav_path = os.path.join(wav_dir, filename)
    futures.append(executor.submit(partial(
      _process_utterance, wav_path, mel_dir, quant_dir)))
  results = [future.result() for future in tqdm(futures)]


if __name__ == '__main__':
  fire.Fire(main)
  print('\nCompleted. Ready to run "python train.py"')
