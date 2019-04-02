from utils.dataset import get_datasets
from utils.dsp import *
from models.fatchord_wavernn import Model
from utils.paths import Paths
from utils.display import simple_table
import torch
import argparse


def gen_testset(model, test_set, samples, batched, target, overlap, save_path) :

    k = model.get_step() // 1000

    for i, (m, x) in enumerate(test_set, 1):

        if i > samples : break

        print('\n| Generating: %i/%i' % (i, samples))

        x = x[0].numpy()

        if hp.mu_law :
            x = decode_mu_law(x, 2**hp.bits, from_labels=True)
        else :
            x = label_2_float(x, hp.bits)

        save_wav(x, f'{save_path}{k}k_steps_{i}_target.wav')

        batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
        save_str = f'{save_path}{k}k_steps_{i}_{batch_str}.wav'

        _ = model.generate(m, save_str, batched, target, overlap, hp.mu_law)


def gen_from_file(model, load_path, save_path, batched, target, overlap) :

    k = model.get_step() // 1000
    file_name = load_path.split('/')[-1]

    wav = load_wav(load_path)
    save_wav(wav, f'{save_path}__{file_name}__{k}k_steps_target.wav')

    mel = melspectrogram(wav)
    mel = torch.tensor(mel).unsqueeze(0)

    batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
    save_str = f'{save_path}__{file_name}__{k}k_steps_{batch_str}.wav'

    _ = model.generate(mel, save_str, batched, target, overlap, hp.mu_law)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate WaveRNN Samples')
    parser.add_argument('--batched', '-b', dest='batched', action='store_true')
    parser.add_argument('--unbatched', '-u', dest='batched', action='store_false')
    parser.add_argument('--samples', '-s', type=int, help='[int] number of samples to generate')
    parser.add_argument('--target', '-t', type=int, help='[int] number of samples in each batch index')
    parser.add_argument('--overlap', '-o', type=int, help='[int] number of crossover samples')
    parser.add_argument('--file', '-f', type=str, help='[string/path] for testing a wav outside dataset')
    parser.add_argument('--weights', '-w', type=str, help='[string/path] checkpoint file to load weights from')

    parser.set_defaults(batched=hp.batched)
    parser.set_defaults(samples=hp.gen_at_checkpoint)
    parser.set_defaults(target=hp.target)
    parser.set_defaults(overlap=hp.overlap)
    parser.set_defaults(file=None)
    parser.set_defaults(weights=None)

    args = parser.parse_args()

    batched = args.batched
    samples = args.samples
    target = args.target
    overlap = args.overlap
    file = args.file

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
                  sample_rate=hp.sample_rate).cuda()

    paths = Paths(hp.data_path, hp.model_id)

    restore_path = args.weights if args.weights else paths.latest_weights

    model.restore(restore_path)

    simple_table([('Generation Mode', 'Batched' if batched else 'Unbatched'),
                  ('Target Samples', target if batched else 'N/A'),
                  ('Overlap Samples', overlap if batched else 'N/A')])

    _, test_set = get_datasets(paths.data)

    if file :
        gen_from_file(model, file, paths.output, batched, target, overlap)
    else :
        gen_testset(model, test_set, samples, batched, target, overlap, paths.output)

    print('\n\nExiting...\n')
