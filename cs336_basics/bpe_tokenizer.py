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
import sys
import regex as re

from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

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
        self.vocab = vocab
        # self.merges = merges
        self.merges_dict = {pair: rank for rank, pair in enumerate(merges)}
        self.vocab_dict = {token: id for id, token in vocab.items()}
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)

        # vocab = {
        #     int(token_id): ast.literal_eval(token_object)
        #         for token_id, token_object in raw_vocab.items()
        # }

        vocab = {
            int(token_id): token_object.encode('utf-8')
                for token_object, token_id in raw_vocab.items()
        }
        # with open(merges_filepath, encoding='utf-8') as f:
        #     raw_merges = json.load(f)

        # merges = [
        #     (ast.literal_eval(left), ast.literal_eval(right)) for left, right in raw_merges
        # ]
        merges = []

        with open(merges_filepath, encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                left, right = line.split()
                # print(left, right)
                # print((left.encode('utf-8'), right.encode('utf-8')))
                # sys.exit(-1)
                merges.append((left.encode('utf-8'), right.encode('utf-8')))
        # print(merges)
        return cls(vocab=vocab, merges=merges,special_tokens=special_tokens)

    def _encode_pre_token(self, text):
        tokens = tuple(bytes([byte]) for byte in text.encode("utf-8"))
        while len(tokens) >= 2:
            best_pair = None
            best_rank = math.inf

            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                rank = self.merges_dict.get(pair, math.inf)

                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break
            j = 0
            new_tokens = []
            while j < len(tokens):
                if (j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best_pair):
                    new_tokens.append(tokens[j] + tokens[j+1])
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = tuple(new_tokens)
 
        return [self.vocab_dict[token] for token in tokens]
    
    def encode(self, text: str) -> list[int]:
        # seperate to binary
        result = []
        if self.special_tokens:
            special_pattern = "|".join(
                re.escape(token)
                for token in sorted(
                    self.special_tokens,
                    key=len,
                    reverse=True,
                )
            )

            text_parts = re.split(
                f"({special_pattern})",
                text,
            )
        else:
            text_parts = [text]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        for text_part in text_parts:
            if not text_part:
                continue
            if self.special_tokens and (text_part in self.special_tokens):
                special_token_id = self.vocab_dict[
                    text_part.encode("utf-8")
                ]
                result.append(special_token_id)
                continue

            for match in re.finditer(PAT, text_part):
                pre_token = match.group()

                result.extend(self._encode_pre_token(pre_token))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        byte_text = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )
        return byte_text.decode("utf-8", errors="replace")

# merges_filepath = 'output/TinyStoriesV2-GPT4-valid-heap-merges.json'
# vocab_filepath = 'output/TinyStoriesV2-GPT4-valid-heap-vocab.json'

# tokenizer = BpeTokenizer.from_files(VOCAB_PATH, MERGES_PATH, ['<|endoftext|>'])

# # print(tokenizer.encode('hello <|endoftext|> hel <|endoftext|> world'))
# # print(tokenizer.decode([5518, 32, 256, 430, 32, 256, 1592]))

# print(tokenizer.encode('s'))
# print(tokenizer.decode([82]))
 