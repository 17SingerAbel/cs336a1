from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.bpe import train_bpe, train_bpe_heap
from cs336_basics.bpe_tokenizer import BpeTokenizer
from cs336_basics.Linear import Linear
import torch.nn as nn
from cs336_basics.Embedding import Embedding
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.SwiGLU import SwiGlu
from cs336_basics.RoPE import RoPE
from cs336_basics.MultiHeadSelfAttention import MultiHeadSelfAttention
from cs336_basics.TransformerBlock import TransformerBlock
from cs336_basics.LanguageModel import LanguageModel
from cs336_basics.AdamW import AdamW
from einops import reduce, rearrange, einsum
import math
import numpy as np
from utils import run_get_batch, run_get_lr_cosine_schedule, run_cross_entropy, run_gradient_clipping, run_save_checkpoint, run_load_checkpoint

vocabs = 'placeholder'
input_file = 'placeholder'
# read file

# ===========

vocab_size = 10000
context_length = 1024
d_model = 1024
num_layers = 10
num_heads = 4
d_ff = 2048

batch_size=32
device='cpu'

rope_theta = 1
betas = [0.9, 0.95]
weight_decay = 1

# how to connect optimizer params, to  language model weights



iterations = 10

# learning rate hyperparam
max_lr = 1e-2
min_lr = 1e-5
warmup_iters = iterations * 0.2
cosine_cycle_iters = iterations * 0.9
max_l2_norm = 1

# build train data
dataset = np.array([])

lm = LanguageModel(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights)

optimizer = AdamW(lm.parameters(), betas, weight_decay, lr=1e-3, eps=1e-8)

losses = []
# lm model
for it in range(iterations):
    optimizer.zero_grad()

    indices, labels = run_get_batch(dataset, batch_size, context_length, device=device)
    output = lm.forward(indices, context_length)
    loss = run_cross_entropy(output, labels)
    losses.append(loss)
    loss.backward()

    lr = run_get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
    for group in optimizer.param_groups:
        group["lr"] = lr

    run_gradient_clipping(lm.parameters, max_l2_norm)
    optimizer.step()
    if it % 5 == 0:
        run_save_checkpoint(lm, optimizer, it, 'checkpoints/{it}_iteration')

