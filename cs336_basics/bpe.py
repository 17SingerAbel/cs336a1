from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import regex as re

# from pretokenization_example import find_chunk_boundaries
from collections import defaultdict


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    # **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # init vocab set 256 byte include special character
    vocab = {i: bytes([i]) for i in range(256)}

    merges = []

    for sepcial_token in special_tokens:
        vocab[len(vocab)] = sepcial_token.encode('utf-8')

    # print(vocab)

    # pre-tokenization text
    f = open(input_path)


    text = f.read()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    
    texts = [text]  # 先把完整文本放进 list

    # remove special character
    for special_token in special_tokens:
        texts = [
            part
            for text_part in texts
            for part in text_part.split(special_token)
        ]

    # pre-tokenization
    pre_token_counts = {}
    for text_part in texts:
        for match in re.finditer(PAT, text_part):
            pre_token = match.group()
            pre_token_counts[pre_token] = pre_token_counts.get(pre_token, 0) + 1

    frequency_table = {
        tuple(bytes([byte]) for byte in pre_token.encode("utf-8")): count
        for pre_token, count in pre_token_counts.items()
    }

    pairs_count = {}
    pairs_index = defaultdict(set)
    
    for pre_token, freq in frequency_table.items():
        for i in range(len(pre_token)-1):
            pair = tuple([pre_token[i], pre_token[i+1]])
            pairs_count[pair] = pairs_count.get(pair, 0) + freq
            pairs_index[pair].add(pre_token)

    while len(vocab) < vocab_size:

        if len(pairs_count) == 0:
            return vocab
        best_pair, best_freq = max(pairs_count.items(), key=lambda item: (item[1], item[0]))


        if best_freq == 0:
            return vocab
        new_token = best_pair[0] + best_pair[1]
        # print(new_token)
        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        affected_per_tokens = pairs_index[best_pair].copy()

        new_pre_tokens = set()

        for affected_token in affected_per_tokens:
            freq = frequency_table[affected_token]

            new_pre_token = []

            for i in range(len(affected_token)-1):
                old_pair = (affected_token[i], affected_token[i+1])
                pairs_count[old_pair] -= freq
                if pairs_count[old_pair] == 0:
                    del pairs_count[old_pair]
                pairs_index[old_pair].discard(affected_token)
                if len(pairs_index[old_pair]) == 0:
                    del pairs_index[old_pair]

            i = 0
            while i < len(affected_token)-1:
                old_pair = (affected_token[i], affected_token[i+1])
                if old_pair == best_pair:
                    new_pre_token.append(new_token)
                    i += 2
                else:
                    new_pre_token.append(affected_token[i])
                    i += 1

            if i == len(affected_token) - 1:
                new_pre_token.append(affected_token[i])
            new_pre_tokens.add(tuple(new_pre_token))

            frequency_table[tuple(new_pre_token)] = frequency_table.get(tuple(new_pre_token), 0) + freq
            frequency_table[affected_token] -= freq
            if frequency_table[affected_token] == 0:
                del  frequency_table[affected_token]


        for new_pre_token in new_pre_tokens:
            for i in range(len(new_pre_token)-1):
                new_pair = (new_pre_token[i], new_pre_token[i+1])
                pairs_count[new_pair] = pairs_count.get(new_pair, 0) + frequency_table[new_pre_token] 
                pairs_index[new_pair].add(new_pre_token)

    # print(vocab)
    # print(merges)
    return vocab, merges

# train_bpe(input_path='data/smallest.txt', vocab_size=300,special_tokens=['<|endoftext|>'])


def train_bpe2(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    # **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # init vocab set 256 byte include special character
    vocab = {i: bytes([i]) for i in range(256)}

    merges = []

    for sepcial_token in special_tokens:
        vocab[len(vocab)] = sepcial_token.encode('utf-8')

    # print(vocab)

    # pre-tokenization text
    f = open(input_path)


    text = f.read()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    
    texts = [text]  # 先把完整文本放进 list

    # remove special character
    for special_token in special_tokens:
        texts = [
            part
            for text_part in texts
            for part in text_part.split(special_token)
        ]

    # pre-tokenization
    pre_token_counts = {}
    for text_part in texts:
        for match in re.finditer(PAT, text_part):
            pre_token = match.group()
            pre_token_counts[pre_token] = pre_token_counts.get(pre_token, 0) + 1

    frequency_table = {
        tuple(bytes([byte]) for byte in pre_token.encode("utf-8")): count
        for pre_token, count in pre_token_counts.items()
    }
    # print(frequency_table)

    while len(vocab) < vocab_size:
        pairs = {}
        for pre_token, freq in frequency_table.items():
            for i in range(len(pre_token)-1):
                pair = tuple([pre_token[i], pre_token[i+1]])
                pairs[pair] = pairs.get(pair, 0) + freq

        # print(pairs)
        best_pair, best_freq = max(pairs.items(), key=lambda item: (item[1], item[0]))
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        new_freq_table = {}

        for pre_token, freq in frequency_table.items():
            new_pre_token = []
            idx=0

            while idx < len(pre_token)-1:
                if (pre_token[idx] == best_pair[0] and pre_token[idx + 1] == best_pair[1]):

                    new_pre_token.append(new_token)
                    idx += 2
                else:

                    new_pre_token.append(pre_token[idx])
                    idx += 1

            if idx == len(pre_token) -1:
                new_pre_token.append(pre_token[idx])

            new_key = tuple(new_pre_token)
            new_freq_table[new_key] = new_freq_table.get(new_key, 0) + freq

        frequency_table = new_freq_table

    return vocab, merges

# train_bpe(input_path='data/smallest.txt', vocab_size=300,special_tokens=['<|endoftext|>'])
