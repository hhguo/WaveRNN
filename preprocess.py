from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count
from tqdm import tqdm

import fire
import numpy as np
import os

from utils.audio import *
from hparams import hparams


def convert_file(path, extract_quant=False, extract_sample=False):
    y = load_wav(path)

    outputs = {}
    outputs['mel'] = melspectrogram(y).T.astype(np.float32)
    if extract_quant:
        quant = encode_mu_law(y, mu=2 ** hparams.bits).astype(np.int32)
        outputs['quant'] = quant
    if extract_sample:
        sample = float_2_label(y, bits=16).astype(np.int32)
        outputs['sample'] = sample
    return outputs


def _process_utterance(path, mel_dir, quant_dir=None, sample_dir=None):
    fid = os.path.split(path)[-1].split('.')[0]
    outputs = convert_file(path, quant_dir is not None, sample_dir is not None)
    np.save(f'{mel_dir}/{fid}.npy', outputs['mel'])
    if quant_dir is not None:
        np.save(f'{quant_dir}/{fid}.npy', outputs['quant'])
    if sample_dir is not None:
        np.save(f'{sample_dir}/{fid}.npy', outputs['quant'])
    return fid


def main(wav_dir, mel_dir, quant_dir=None, sample_dir=None, config=''):
    os.makedirs(mel_dir, exist_ok=True)
    if quant_dir is not None:
        os.makedirs(quant_dir, exist_ok=True)
    if sample_dir is not None:
        os.makedirs(sample_dir, exist_ok=True)
    hparams.parse(config)

    executor = ProcessPoolExecutor(max_workers=cpu_count())
    futures = []
    for filename in os.listdir(wav_dir):
        wav_path = os.path.join(wav_dir, filename)
        futures.append(executor.submit(partial(
            _process_utterance, wav_path, mel_dir, quant_dir, sample_dir)))
    results = [future.result() for future in tqdm(futures)]


if __name__ == '__main__':
    fire.Fire(main)
    print('\nCompleted. Ready to run "python train.py"')
