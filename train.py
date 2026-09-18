from __future__ import annotations

import numpy.typing as npt
import torch

from cs336_basics.bpe_tokenizer import BpeTokenizer
from cs336_basics.LanguageModel import LanguageModel
from cs336_basics.AdamW import AdamW
from cs336_basics.utils import run_get_batch, run_get_lr_cosine_schedule, run_cross_entropy, run_gradient_clipping, run_save_checkpoint, run_load_checkpoint


# ===========

vocab_size = 10000
context_length = 1024
d_model = 256
num_layers = 4
num_heads = 2
d_ff = 2048

batch_size=8
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

# build train data from tokenizer
merges_filepath = 'output/TinyStoriesV2-GPT4-train-heap-merges.json'
vocab_filepath = 'output/TinyStoriesV2-GPT4-train-heap-vocab.json'

print('loading bpe tokenizer')
tokenizer = BpeTokenizer.from_files(vocab_filepath, merges_filepath, ['<|endoftext|>'])
with open('data/smallest.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print('embedding text from tokenizer')
dataset = tokenizer.encode(text) 

lm = LanguageModel(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=device, dtype=dtype)
optimizer = AdamW(lm.parameters(), betas, weight_decay, lr=1e-3, eps=1e-8)

losses = []
# lm model
for it in range(iterations):
    print(f'========= itr {it} =========')
    optimizer.zero_grad()

    indices, labels = run_get_batch(dataset, batch_size, context_length, device=device)
    output = lm.forward(indices, context_length)
    print(output.shape)
    loss = run_cross_entropy(output, labels)
    print(loss)
    losses.append(loss)
    loss.backward()

    lr = run_get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
    for group in optimizer.param_groups:
        group["lr"] = lr

    run_gradient_clipping(lm.parameters(), max_l2_norm)
    optimizer.step()
    if it % 5 == 0:
        run_save_checkpoint(lm, optimizer, it, 'checkpoints/{it}_iteration')


run_save_checkpoint(lm, optimizer, it, 'checkpoints/final')