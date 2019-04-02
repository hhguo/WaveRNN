import librosa
import librosa.filters
import math
import numpy as np
import torch
import tensorflow as tf
from scipy import signal
from hparams import hparams


def load_wav(path):
  return librosa.core.load(path, sr=hparams.sample_rate)[0]


def save_wav(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  librosa.output.write_wav(path, wav.astype(np.int16), hparams.sample_rate)


def preemphasis(x):
  return signal.lfilter([1, -hparams.preemphasis], [1], x)


def inv_preemphasis(x):
  return signal.lfilter([1], [1, -hparams.preemphasis], x)


def spectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
  return _normalize(S)


def fast_inv_spectrogram(spectrogram):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
  return inv_preemphasis(_fast_griffin_lim(S ** hparams.power))          # Reconstruct phase


def inv_mel_spectrogram(mel_spectrogram):
  D = _denormalize(mel_spectrogram.T)
  S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db))
  return inv_preemphasis(_griffin_lim(S ** hparams.power))


def inv_spectrogram(spectrogram):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
  return inv_preemphasis(_griffin_lim(S ** hparams.power))          # Reconstruct phase


def melspectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
  return _normalize(S)


def _stft(y):
  n_fft, hop_length, win_length = _stft_parameters()
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
  _, hop_length, win_length = _stft_parameters()
  return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
  n_fft = (hparams.num_freq - 1) * 2
  hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
  win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
  return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None
_inv_mel_basis = None
_inv_mel_basis_tensor = None


def _linear_to_mel(spectrogram):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)


def _mel_to_linear(mel_spectrogram):
  global _inv_mel_basis
  if _inv_mel_basis is None:
    _inv_mel_basis = np.linalg.pinv(_build_mel_basis())
  return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis():
  n_fft = (hparams.num_freq - 1) * 2
  return librosa.filters.mel(hparams.sample_rate, n_fft, htk=False, n_mels=hparams.num_mels)


def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
  return np.power(10.0, x * 0.05)


def _db_to_amp_pytorch(x):
  return torch.pow(torch.ones(x.size()) * 10.0, x * 0.05)


def _db_to_amp_tensorflow(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _normalize(S):
  if hparams.symmetric_specs:
    return np.clip(
      (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
      -hparams.max_abs_value, hparams.max_abs_value)
  else:
    return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)


def _denormalize(S):
  if hparams.symmetric_specs:
    return (((np.clip(S, -hparams.max_abs_value, hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
  else:
    return ((np.clip(S, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)


def label_2_float(x, bits):
  return 2 * x / (2 ** bits - 1.) - 1.


def float_2_label(x, bits):
  return (x + 1.) * (2 ** bits - 1) / 2


def split_signal(x):
  unsigned = x + 2 ** 15
  return unsigned // 256, unsigned % 256


def combine_signal(coarse, fine):
  return coarse * 256 + fine - 2 ** 15


def encode_16bits(x):
  return np.clip(x * (2 ** 15), -2 ** 15, 2 ** 15 - 1).astype(np.int16)


def encode_mu_law(x, mu):
  mu = mu - 1
  fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
  return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True):
  if from_labels:
    y = label_2_float(y, math.log2(mu))
  mu = mu - 1
  x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
  return x
