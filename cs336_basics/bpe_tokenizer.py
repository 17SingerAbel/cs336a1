from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.bpe import train_bpe, train_bpe_heap
import json
import ast
import math

class BpeTokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        """Given a vocabulary, a list of merges, and a list of special tokens,
        return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

        Args:
            vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
            special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
                be split into multiple tokens, and will always be kept as a single token.

        Returns:
            A BPE tokenizer that uses the provided vocab, merges, and special tokens.
        """
        # self.vocab = vocab
        # self.merges = merges
        self.merges_dict = {pair: rank for rank, pair in enumerate(merges)}
        self.vocab_dict = {token: id for id, token in vocab.items()}
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {
            int(token_id): ast.literal_eval(token_object['bytes'])
                for token_id, token_object in raw_vocab.items()
        }

        with open(merges_filepath, encoding='utf-8') as f:
            raw_merges = json.load(f)
        merges = [
            (ast.literal_eval(left), ast.literal_eval(right)) for left, right in raw_merges
        ]
        return cls(vocab=vocab, merges=merges,special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        # seperate to binary
        pre_token = tuple(bytes([byte]) for byte in text.encode("utf-8"))

        while True:
            merge_idx = -1
            smallest_rank = math.inf

            for i in range(len(pre_token)):
                if i < len(pre_token) - 1:
                    pair = (pre_token[i], pre_token[i+1])
                    if self.merges_dict.get(pair, math.inf) < smallest_rank:
                        smallest_rank = self.merges_dict.get(pair, math.inf)
                        merge_idx = i

            if merge_idx == -1:
                break
            # merge at idx i
            j = 0
            new_pre_token_elements = []
            while j < len(pre_token):
                if j != merge_idx:
                    new_pre_token_elements.append(pre_token[j])
                    j += 1
                else:
                    new_pre_token_elements.append(pre_token[j] + pre_token[j+1])
                    j += 2

            pre_token = tuple(new_pre_token_elements)

        print(pre_token)
        result = []
        for token in pre_token:
            print(token)
            result.append(self.vocab_dict[token])
        print(result)
        return result
                

        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # for special_token in self.special_tokens:
        #     texts = [
        #         part
        #         for text_part in texts
        #         for part in text_part.split(special_token)
        #     ]
        # for chunck in texts:
        #     self.encode(chunck)
        pass

    def decode(self, ids: list[int]) -> str:
        return

merges_filepath = 'output/TinyStoriesV2-GPT4-valid-heap-merges.json'
vocab_filepath = 'output/TinyStoriesV2-GPT4-valid-heap-vocab.json'

tokenizer = BpeTokenizer.from_files(vocab_filepath, merges_filepath, ['<|endoftext|>'])

tokenizer.encode('hello world')