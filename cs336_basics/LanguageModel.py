import torch
import torch.nn as nn
from cs336_basics.TransformerBlock import TransformerBlock
from cs336_basics.Embedding import Embedding
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.Linear import Linear

class LanguageModel(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights):
        super().__init__()

        self.token_embedding = Embedding(vocab_size, d_model, weights['token_embeddings.weight'])
        self.num_layers = num_layers
        self.layers: dict[int, TransformerBlock] = {}
        for i in range(num_layers):
            prefix = f"layers.{i}."
            block_weights = {
                key[len(prefix):]: value
                for key, value in weights.items() if key.startswith(prefix)
            }
            self.layers[i] = TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, weights=block_weights)

        self.ln_final = RMSNorm(d_model, weights=weights['ln_final.weight'])
        self.lm_head = Linear(d_model, vocab_size, weights=weights['lm_head.weight'])

    def forward(self, input_ids, seq_len):

        x = self.token_embedding.forward(input_ids)

        for i in range(self.num_layers):
            x = self.layers[i].forward(x, seq_len)

        x = self.ln_final.forward(x)
        logits = self.lm_head(x)
        return logits
