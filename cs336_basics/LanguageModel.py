import torch
import torch.nn as nn
from cs336_basics.TransformerBlock import TransformerBlock
from cs336_basics.Embedding import Embedding
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.Linear import Linear

class LanguageModel(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights=None, device=None, dtype=None):
        super().__init__()


        embedding_weights = weights['token_embeddings.weight'] if weights is not None else None
        self.token_embedding = Embedding(vocab_size, d_model, weights=embedding_weights, device=device, dtype=dtype)

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            prefix = f"layers.{i}."
            if weights is not None:
                block_weights = {
                    key[len(prefix):]: value
                    for key, value in weights.items() 
                    if key.startswith(prefix)
                }
            else:
                block_weights = None
            self.layers.append(TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, weights=block_weights, device=device, dtype=dtype))

        ln_final_weights = weights['ln_final.weight'] if weights is not None else None
        self.ln_final = RMSNorm(d_model, weights=ln_final_weights)

        lm_head_weights = weights['lm_head.weight'] if weights is not None else None
        self.lm_head = Linear(d_model, vocab_size, weights=lm_head_weights)

    def forward(self, input_ids, seq_len):

        x = self.token_embedding.forward(input_ids)

        for layer in self.layers:
            x = layer.forward(x, seq_len)

        x = self.ln_final.forward(x)
        logits = self.lm_head(x)
        return logits
