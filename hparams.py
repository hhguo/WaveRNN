import tensorflow as tf

hparams=tf.contrib.training.HParams(
  # Audio
  num_mels=80,
  num_freq=1025,
  sample_rate=16000,
  frame_length_ms=50,
  frame_shift_ms=12.5,
  hop_length=200,
  preemphasis=0.97,
  min_level_db=-100,
  ref_level_db=20,
  max_abs_value=4.0,
  symmetric_specs=True,
  bits=9,
  mu_law=True,

  # Model
  mode = 'RAW', # RAW or MOL
  upsample_factors=(4, 5, 10),
  rnn_dims=512,
  fc_dims=512,
  compute_dims=128,
  res_out_dims=128,
  res_blocks=10,

  # Data loader
  pin_memory=True,
  num_workers=16,
  
  # Training
  use_zip=False,
  batch_size=32,
  lr=1e-4,
  checkpoint_interval=25000,
  gen_at_checkpoint=5,          # number of samples to generate at each checkpoint
  nepochs=10000,                 # Total number of training steps
  seq_len=1000,                 # must be a multiple of hop_length
  pad=2,

  # Optimizer
  adam_beta1=0.9,
  adam_beta2=0.999,
  weight_decay=0.0,
  clip_thresh=1.0,
  
  # Generate
  batched=False,                 # very fast (realtime+) single utterance batched generation
  target=8000,                 # target number of samples to be generated in each batch entry
  overlap=400,                  # number of samples for crossfading between batches
)

def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
  return 'Hyperparameters:\n' + '\n'.join(hp)
