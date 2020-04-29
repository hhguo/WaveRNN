import fire
import os
import re
import torch

from models.fatchord_wavernn import Model
from utils.audio import *
from utils.display import simple_table
from hparams import hparams

use_cuda = torch.cuda.is_available()
batch_size = 4


def _pad_2d(x, max_len, constant_values=0):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant",
               constant_values=constant_values)
    return x


def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    match = re.compile(r'.*checkpoint_step([0-9]+)\.pth').match(
        checkpoint_path)
    name = 'eval-%d' % int(match.group(1)) if match else 'eval'
    return os.path.join(base_dir, name)


def gen_from_file(model, mel, save_path, batched, target, overlap):
    if isinstance(mel, list):
        upsample = int(hparams.sample_rate * hparams.frame_shift_ms / 1000)
        for i in range(0, len(mel), batch_size):
            inputs = mel[i:min(i + batch_size, len(mel))]
            input_lengths = [x.shape[0] for x in inputs]
            max_length = max(input_lengths)
            inputs = [_pad_2d(x, max_length, -4) for x in inputs]
            inputs = torch.tensor(np.stack(inputs)).permute(0, 2, 1)
            inputs = inputs.cuda() if use_cuda else inputs
            samples = model.generate(inputs, batched, target, overlap,
                                     hparams.mu_law)
            for bi in range(inputs.size(0)):
                input_length = input_lengths[bi] * upsample
                save_wav(samples[bi, :input_length], save_path[i + bi])
    else:
        mel = np.load(mel).T
        mel = torch.tensor(mel).unsqueeze(0)
        mel = mel.cuda() if use_cuda else mel
        samples = model.generate(mel, batched, target, overlap, hparams.mu_law)
        save_wav(samples[0], save_path)


def main(ckpt_path, input_path, output_path=None, list_path=None):
    batched = hparams.batched
    samples = hparams.gen_at_checkpoint
    target = hparams.target
    overlap = hparams.overlap

    if output_path is None:
        output_path = get_output_base_path(ckpt_path)
    os.makedirs(output_path, exist_ok=True)
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if list_path is not None:
        with open(list_path) as fin:
            fids = [line.strip() for line in fin.readlines()]
    else:
        fids = []
        for filename in os.listdir(input_path):
            if '.npy' in filename:
                fids.append(filename.split('.')[0])

    mel, output = [], []
    for fid in fids:
        mel.append(np.load(os.path.join(input_path, fid + '.npy')))
        output.append(os.path.join(output_path, fid + '.wav'))

    print('\nInitialising Model...\n')

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

    model.load_state_dict(checkpoint["state_dict"])

    with torch.no_grad():
        gen_from_file(model, mel, output, batched, target, overlap)


if __name__ == '__main__':
    fire.Fire(main)
    print('\nDone!\n')
