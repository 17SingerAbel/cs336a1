import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum

class Embedding(nn.Module):

    def __init__(self, num_embeddings,embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.embedding_matrix = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            ), requires_grad=True
        )

        std = math.sqrt(2 / (num_embeddings + embedding_dim))

        nn.init.trunc_normal_(
            self.embedding_matrix,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, token_ids: torch.Tensor ) -> torch.Tensor:
        return self.embedding_matrix[token_ids]
