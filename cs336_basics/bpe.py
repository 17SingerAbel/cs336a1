from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import regex as re
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import defaultdict
import time
import psutil
import sys

def print_memory(tag):
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss / 1024 ** 3
    print(f"{tag}: {rss: .2f} GB")
    if rss > 10:
        print("exit to prevent memory full")
        sys.exit(-1)
    return rss

def worker(start, end, input_path, special_tokens):
    pre_token_counts = {}
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        texts = [chunk]  # 先把完整文本放进 list
    
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

    return pre_token_counts

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

    print_memory("start")
    # init vocab set 256 byte include special character
    vocab = {i: bytes([i]) for i in range(256)}

    merges = []

    for sepcial_token in special_tokens:
        vocab[len(vocab)] = sepcial_token.encode('utf-8')

    # 1. Build tasks

    
    tasks = []
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            # chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            tasks.append((start, end, input_path, special_tokens))

    # 2. Run workers
    num_workers = min(os.cpu_count(), len(tasks))


    start = time.perf_counter()
    
    with Pool(processes=num_workers) as pool:
        worker_results = pool.starmap(worker, tasks)

    # 3. Merge results
    pre_token_counts = {}

    for local_counts in worker_results:
        for token, count in local_counts.items():
            pre_token_counts[token] = pre_token_counts.get(token, 0) + count

    elapsed = time.perf_counter() - start
    print(f"BPE Pre-tokenization time: {elapsed:.2f} seconds")
    print_memory("after pre-tokenization")

    start = time.perf_counter()

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

    print_memory("after BPE initialization")

    loop = 0
    while len(vocab) < vocab_size:

        if len(pairs_count) == 0:
            return vocab
        best_pair, best_freq = max(pairs_count.items(), key=lambda item: (item[1], item[0]))

        if 6640 <= loop <= 6720:
            print("\n==========")
            print("loop:", loop)
            # print("best pair:", best_pair)
            print("best pair count:", pairs_count[best_pair])

        if best_freq == 0:
            return vocab
        new_token = best_pair[0] + best_pair[1]

        if len(new_token) > 5:
            print("best freq:", best_freq)
            print("best pair:", best_pair)

            print("left token:", repr(best_pair[0]))
            print("right token:", repr(best_pair[1]))

            print("left len:", len(best_pair[0]))
            print("right len:", len(best_pair[1]))
            print("new token len:", len(new_token))
            print("affected tokens", affected_per_tokens)

            sys.exit(-1)

        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        affected_per_tokens = pairs_index[best_pair].copy()

        new_pre_tokens = set()

        for affected_token in affected_per_tokens:
            freq = frequency_table[affected_token]

            new_pre_token = []

            idx = 0
            for i in range(len(affected_token)-1):
                old_pair = (affected_token[i], affected_token[i+1])
                pairs_count[old_pair] -= freq
                if pairs_count[old_pair] == 0 :
                    del pairs_count[old_pair]
                pairs_index[old_pair].discard(affected_token)
                if len(pairs_index[old_pair]) == 0:
                    del pairs_index[old_pair]

                if old_pair == best_pair:
                    new_pre_token.append(new_token)
                    idx += 2
                else:
                    if idx == i:
                        new_pre_token.append(affected_token[i])
                        idx += 1

            if idx == len(affected_token) - 1:
                new_pre_token.append(affected_token[idx])
            new_pre_tokens.add(tuple(new_pre_token))

            frequency_table[tuple(new_pre_token)] = frequency_table.get(tuple(new_pre_token), 0) + freq
            frequency_table[affected_token] -= freq
            if frequency_table[affected_token] == 0:
                del  frequency_table[affected_token]

        if 6640 <= loop <= 6720:
            print("--- after create/cleanup ---")
            print("new token length:", len(new_token))
        print_memory(f"after {loop}th loop, after create new pre-token and clean up old pre-token")

        for new_pre_token in new_pre_tokens:
            for i in range(len(new_pre_token)-1):
                new_pair = (new_pre_token[i], new_pre_token[i+1])
                pairs_count[new_pair] = pairs_count.get(new_pair, 0) + frequency_table[new_pre_token] 
                pairs_index[new_pair].add(new_pre_token)

        if 6640 <= loop <= 6720:
            print(f"\n--- loop {loop} ---")

            print("pair_counts:", len(pairs_count))
            print("pair_index:", len(pairs_index))
            print("vocab:", len(vocab))
            print("merges:", len(merges))
        print_memory(f"after {loop}th loop, after update pair counts and pair index")
        loop += 1
    elapsed = time.perf_counter() - start
    print(f"BPE Merge BPE time: {elapsed:.2f} seconds")
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

