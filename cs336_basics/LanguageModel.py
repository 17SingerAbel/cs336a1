import torch
import torch.nn as nn
from cs336_basics.TransformerBlock import TransformerBlock
from cs336_basics.Embedding import Embedding
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.Linear import Linear

class LanguageModel(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype))

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids, seq_len):
        x = self.token_embeddings.forward(input_ids)

        for layer in self.layers:
            x = layer.forward(x, seq_len)

        x = self.ln_final.forward(x)
        logits = self.lm_head(x)
        return logits
