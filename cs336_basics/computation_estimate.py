
# embedding table vocab_size x d_model
# transformer block
# for layer i
#     pre_norm1 = d_model
#     q_proj = d_model x d_model
#     k_proj = d_model x d_model
#     v_proj = d_model x d_model
#     o_proj = d_model x d_model
#     pre_norm2 = d_model
#     swiglu: w1: d_ff x d_model  w2: d_model x d_ff, w3: d_ff x d_model
# ln_final_norm: d_model
# lm head: vocab_size x d_model

# total:

# vocab_size * d_model * 2 + (2 * layers + 1) * d_model + 4 * d_model * d_model * layers + 3 * d_ff * d_model* layers


vocab_size = 50257
context_length = 1024
num_layers = 36
d_model = 1280
num_heads = 20
d_ff = 4288

trainable_parameters_amount = 2 * vocab_size * d_model + (2 * num_layers + 1) * d_model + 4 * (d_model ** 2) * num_layers + 3 * num_layers * d_model * d_ff

memory_used = trainable_parameters_amount *4 / (1024 * 1024 * 1024)

print('===============', num_layers, d_model, num_layers)
print(trainable_parameters_amount / 1000000000.0, 'B params')
print(memory_used, ' GB')

# flops x: batch seq d_model
# embedding_flops = 0
 
# transformer_block_norm1 = 2 * d_model * context_length
multi_heads_flops = 2 * d_model * context_length * d_model * 4 + 2 * d_model * context_length * context_length * 2
# residual_1 = context_length * d_model
# transformer_block_norm2 = 2 * d_model * context_length
swiglu = 2 * d_model * d_ff * context_length * 3
# residual_2 = context_length * d_model
flops_per_layers = multi_heads_flops + swiglu 
# ln_final_flops = 2 * d_model * context_length
lm_head_flops = 2 * d_model * context_length * vocab_size
total_flops =  flops_per_layers * num_layers  + lm_head_flops


print(total_flops / (1000000000000), 'T flops')
print('transformer flop ration: ', (flops_per_layers * num_layers) / total_flops * 100)
print('swiglu per transformer block: ', (swiglu) / flops_per_layers * 100)
print('multi head flop ration: ', (multi_heads_flops ) / flops_per_layers * 100)

batch_size = 64

norm_activation = batch_size * context_length * d_model

QKV_proj_activation = batch_size * context_length * d_model

QK_activation = batch_size * num_heads * context_length * context_length

# softmax_activation = 




import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
# from cs336_basics.bpe import train_bpe, train_bpe_heap
# from cs336_basics.bpe_tokenizer import BpeTokenizer
# from cs336_basics.Linear import Linear
# import torch.nn as nn
# from cs336_basics.Embedding import Embedding
# from cs336_basics.RMSNorm import RMSNorm
# from cs336_basics.SwiGLU import SwiGlu
# from cs336_basics.RoPE import RoPE
# from cs336_basics.MultiHeadSelfAttention import MultiHeadSelfAttention
# from cs336_basics.TransformerBlock import TransformerBlock
# from cs336_basics.LanguageModel import LanguageModel
# from cs336_basics.AdamW import AdamW
from einops import reduce, rearrange, einsum
import math
import numpy as np
