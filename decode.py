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


lm = LanguageModel(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=device, dtype=dtype)
optimizer = AdamW(lm.parameters(), betas, weight_decay, lr=1e-3, eps=1e-8)

print('loading model')
run_load_checkpoint('checkpoints/final', lm, optimizer)

input_text = 'I went to'
input_data = tokenizer.encode(input_text)

def predict_next_token(lm, tokenizer, input_data):
    logits = lm.forward(input_data, len(input_data))
    print('output logits', logits)
    tlida = 0.01

    # softmax
    max_entity = torch.amax(logits, dim=-1, keepdim=True)
    softmax_score = torch.exp((logits - max_entity) / tlida) / torch.sum(torch.exp((logits - max_entity) / tlida), dim=-1, keepdim=True)

    last_token_probs = softmax_score[..., -1, :]   # (B, V)
    idx = torch.argmax(last_token_probs, dim=-1) # (B,)

    tokens = input_data + [idx.item()]

    return tokens


output_tokens = input_data

i=0

end_ids = tokenizer.encode('<|endoftext|>')

while i < 100 and input_data[-1] not in end_ids:

    output_tokens = predict_next_token(lm, tokenizer, input_data)
    print('i loop', i, input_data, output_tokens)
    input_data = output_tokens
    i = i + 1


result = tokenizer.decode(output_tokens)
print(result)