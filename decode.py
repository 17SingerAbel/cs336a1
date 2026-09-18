from __future__ import annotations

import numpy.typing as npt
import torch

from cs336_basics.bpe_tokenizer import BpeTokenizer
from cs336_basics.LanguageModel import LanguageModel
from cs336_basics.AdamW import AdamW
from cs336_basics.utils import run_load_checkpoint


vocab_size = 10000
context_length = 1024
d_model = 1024
num_layers = 10
num_heads = 4
d_ff = 2048

batch_size=32
device='cpu'
dtype=torch.bfloat16

rope_theta = 1
betas = [0.9, 0.95]
weight_decay = 1


iterations = 10

# learning rate hyperparam
max_lr = 1e-2
min_lr = 1e-5
warmup_iters = iterations * 0.2
cosine_cycle_iters = iterations * 0.9
max_l2_norm = 1



lm = LanguageModel(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=device, dtype=dtype)
optimizer = AdamW(lm.parameters(), betas, weight_decay, lr=1e-3, eps=1e-8)

run_load_checkpoint('checkpoints/final', lm, optimizer)

input_text = 'I went to school'
input_data = ''

lm.forward(input_data, context_length)

 