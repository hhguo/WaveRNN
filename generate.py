import argparse
import os
import torch

from models.fatchord_wavernn import Model
from utils.audio import *
from utils.display import simple_table
from hparams import hparams as hp 


use_cuda = torch.cuda.is_available()
batch_size = 4


def _pad_2d(x, max_len, constant_values=0):
  x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
             mode="constant", constant_values=constant_values)
  return x


def gen_from_file(model, mel, save_path, batched, target, overlap):
    if isinstance(mel, list):
        upsample = int(hp.sample_rate * hp.frame_shift_ms / 1000)
        for i in range(0, len(mel), batch_size):
            inputs = mel[i: min(i + batch_size, len(mel))]
            input_lengths = [x.shape[0] for x in inputs]
            max_length = max(input_lengths)
            inputs = [_pad_2d(x, max_length, -4) for x in inputs]
            inputs = torch.tensor(np.stack(inputs)).permute(0, 2, 1)
            inputs = inputs.cuda() if use_cuda else inputs
            samples = model.generate(
                inputs, batched, target, overlap, hp.mu_law)
            for bi in range(inputs.size(0)):
                input_length = input_lengths[bi] * upsample
                save_wav(samples[bi, : input_length],
                         save_path[i + bi])
    else:
        mel = np.load(mel).T
        mel = torch.tensor(mel).unsqueeze(0)
        mel = mel.cuda() if use_cuda else mel
        samples = model.generate(
            mel, batched, target, overlap, hp.mu_law)
        save_wav(samples[0], save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate WaveRNN Samples')
    parser.add_argument('--mel', '-m', type=str, help='[string/path] for Mel file')
    parser.add_argument('--list', '-l', type=str, help='[string/path] for Mel List')
    parser.add_argument('--output', '-o', type=str, help='[string/path] for output')
    parser.add_argument('--checkpoint', '-c', type=str, help='[string/path] checkpoint file')

    parser.set_defaults(batched=hp.batched)
    parser.set_defaults(samples=hp.gen_at_checkpoint)
    parser.set_defaults(target=hp.target)
    parser.set_defaults(overlap=hp.overlap)

    args = parser.parse_args()

    batched = hp.batched
    samples = hp.gen_at_checkpoint
    target = hp.target
    overlap = hp.overlap
    
    mel = args.mel
    output = args.output
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if args.list is not None:
        mel, output = [], []
        with open(args.list) as fin:
            fids = [line.strip() for line in fin.readlines()]
        for fid in fids:
            mel.append(np.load(os.path.join(args.mel, fid + '.npy')))
            output.append(os.path.join(args.output, fid + '.wav'))

    print('\nInitialising Model...\n')

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
                  sample_rate=hp.sample_rate)

    if use_cuda:
        model = model.cuda()

    model.load_state_dict(checkpoint["state_dict"])
    
    gen_from_file(model, mel, output, batched, target, overlap)
    
    print('\nExiting...\n')
