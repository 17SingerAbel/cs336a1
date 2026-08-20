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
    # print(frequency_table)

    pairs_count = {}
    pairs_index = defaultdict(set)
    
    for pre_token, freq in frequency_table.items():
        for i in range(len(pre_token)-1):
            pair = tuple([pre_token[i], pre_token[i+1]])
            pairs_count[pair] = pairs_count.get(pair, 0) + freq
            pairs_index[pair].add(pre_token)

    k = 0
    while len(vocab) < vocab_size:
    # for j in range(2):
        # print(pairs)
        best_pair, best_freq = max(pairs_count.items(), key=lambda item: (item[1], item[0]))
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        affected_per_tokens = list(pairs_index[best_pair]).copy()
        # print(affected_per_tokens)
        # print(frequency_table)
        # print(best_pair, best_freq)
        # print(affected_per_tokens)
        for affected_token in affected_per_tokens:
            new_affected_token = []
            freq = frequency_table[affected_token]

            old_pairs = {}
            i = 0
            while i < len(affected_token)-1:
                old_pair = (affected_token[i], affected_token[i+1])
                old_pairs[old_pair] = old_pairs.get(old_pair, 0) + 1

                if old_pair == best_pair:
                    new_affected_token.append(new_token)
                    i += 2
                else:
                    new_affected_token.append(affected_token[i])
                    i += 1
            if i == len(affected_token)-1:
                new_affected_token.append(affected_token[i])
            # print('debugger')
            # print(old_pairs)
            # print(pairs_index)
            for old_pair, count in old_pairs.items():
                pairs_count[old_pair] -= count * freq
                pairs_index[old_pair].discard(affected_token)
            # print('---------affect token')
            # print(affected_token)

            frequency_table[affected_token] = 0

            new_affected_token = tuple(new_affected_token)
            # print('---------new affected token')
            # print(new_affected_token)
            for j in range(len(new_affected_token)-1):
                new_pair = tuple([new_affected_token[j], new_affected_token[j+1]])
                pairs_count[new_pair] = pairs_count.get(new_pair, 0) + freq
                
                pairs_index[new_pair].add(new_affected_token)

            frequency_table[new_affected_token] = frequency_table.get(new_affected_token, 0) + freq

            
        # print(pairs_index)
        # print('----- round ' + str(k))
        # k+= 1
        # print(pairs_count)
        # print('---------frequency table')
        # print(frequency_table)
        # print('----- round ' + str(j))
        # break
    return vocab, merges

# train_bpe(input_path='data/smallest.txt', vocab_size=300,special_tokens=['<|endoftext|>'])
