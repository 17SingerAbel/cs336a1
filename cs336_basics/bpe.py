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
import heapq


class HeapItem:
    __slots__ = ("freq", "pair")

    def __init__(self, freq, pair):
        self.freq = freq
        self.pair = pair

    def __lt__(self, other):
        # heapq 弹出“最小”元素，所以这里反过来比较
        if self.freq != other.freq:
            return self.freq > other.freq

        # frequency 相同时，选择 lexicographically 最大的 pair
        return self.pair > other.pair

def print_memory(tag):
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss / 1024 ** 3
    print(f"{tag}: {rss: .2f} GB")
    if rss > 10:
        print("exit to prevent memory full")
        sys.exit(-1)
    return rss

def worker(start, end, input_path, special_tokens):

    pid = os.getpid()
    process = psutil.Process(pid)

    def log(tag):
        rss = process.memory_info().rss / 1024**3
        print(
            f"[PID {pid}] {tag} | "
            f"range={start}:{end} | "
            f"RAM={rss:.2f} GB",
            flush=True,
        )

    log("worker started")

    pre_token_counts = {}
    texts = []
    
    with open(input_path, 'rb') as f:
        f.seek(start)
        log("before read")
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        log("after read")
        
        texts.append(chunk) # 先把完整文本放进 list

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # remove special character
    for special_token in special_tokens:
        texts = [
            part
            for text_part in texts
            for part in text_part.split(special_token)
        ]
    log("after special token split")
    # pre-tokenization
    pre_token_counts = {}
    for text_part in texts:
        for match in re.finditer(PAT, text_part):
            pre_token = match.group()
            pre_token_counts[pre_token] = pre_token_counts.get(pre_token, 0) + 1
    log("after pre-tokenization")
    return pre_token_counts

def worker_wrapper(args):
    return worker(*args)

def train_bpe_heap(
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
        num_chuncks = 40
        boundaries = find_chunk_boundaries(f, num_chuncks, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            # chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            tasks.append((start, end, input_path, special_tokens))

    num_processes = 6
    # 2. Run workers
    num_workers = min(os.cpu_count() // 2, num_processes)
    pre_token_counts = {}
    
    start = time.perf_counter()
    
    with Pool(processes=num_workers) as pool:
        # worker_results = pool.starmap(worker, tasks)
        results = pool.imap_unordered(worker_wrapper, tasks)
        for local_counts in results:
            for token, count in local_counts.items():
                pre_token_counts[token] = pre_token_counts.get(token, 0) + count

    print_memory("workes are done, merge results from workers")
    # 3. Merge results

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

    pair_heap = [
        HeapItem(freq, pair)
        for pair, freq in pairs_count.items()
    ]

    heapq.heapify(pair_heap)
        
    num_old_pair_updates = 0
    num_new_pair_updates = 0
    num_merged_occurrences = 0
    num_affected_token = 0

    total_find_max_time = 0
    total_merge_time = 0

    loop = 0
    loop_100_start_time = time.perf_counter()
    while len(vocab) < vocab_size and pair_heap:
        changed_pairs = set()
        
        if len(pairs_count) == 0:
            print(loop)
            return vocab, merges
        t0 = time.perf_counter()

        best_freq, best_pair = None, None

        while pair_heap:
            # get max pair
            item = heapq.heappop(pair_heap)
            # print(item.pair)
    
            current_freq = pairs_count.get(item.pair)
    
            # heap 中的记录仍然与真实 count 一致
            if current_freq == item.freq:
                best_pair = item.pair
                best_freq = item.freq
                break
                
        # best_pair, best_freq = max(pairs_count.items(), key=lambda item: (item[1], item[0]))
        # if best_freq == 0:
        #     return vocab
        total_find_max_time += time.perf_counter() - t0

        new_token = best_pair[0] + best_pair[1]

        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        affected_per_tokens = pairs_index[best_pair].copy()
        # if 6640 <= loop <= 6720:

        if len(new_token) > 2000:
            print("best freq:", best_freq)
            print("left len:", len(best_pair[0]))
            print("right len:", len(best_pair[1]))
            print("new token len:", len(new_token))

            sys.exit(-1)

        new_pre_tokens = set()

        t0 = time.perf_counter()
        for affected_token in affected_per_tokens:
            # sum_affect_token_len += len(affected_token)
            num_affected_token += 1
            freq = frequency_table[affected_token]

            new_pre_token = []

            idx = 0
            for i in range(len(affected_token)-1):
        
                old_pair = (affected_token[i], affected_token[i+1])
   
                pairs_count[old_pair] -= freq
                num_old_pair_updates += 1

                changed_pairs.add(old_pair)
                if pairs_count[old_pair] == 0 :
                    del pairs_count[old_pair]
                pairs_index[old_pair].discard(affected_token)

                # sum_pair_count_change += 1

                if len(pairs_index[old_pair]) == 0:
                    del pairs_index[old_pair]
                if idx != i:
                    continue
                if old_pair == best_pair:
                    new_pre_token.append(new_token)
                    idx += 2

                    num_merged_occurrences += 1
                else:
                    new_pre_token.append(affected_token[i])
                    idx += 1

            if idx == len(affected_token) - 1:
                new_pre_token.append(affected_token[idx])
            new_pre_tokens.add(tuple(new_pre_token))

            frequency_table[tuple(new_pre_token)] = frequency_table.get(tuple(new_pre_token), 0) + freq
            frequency_table[affected_token] -= freq
            if frequency_table[affected_token] == 0:
                del  frequency_table[affected_token]

        for new_pre_token in new_pre_tokens:

            j = 0
            while j < len(new_pre_token)-1:
                new_pair = (new_pre_token[j], new_pre_token[j+1])
                # print("new_pair:", new_pair)
                pairs_count[new_pair] = pairs_count.get(new_pair, 0) + frequency_table[new_pre_token] 

                num_new_pair_updates += 1
                # print("new pair counts:", pairs_count[new_pair])
                pairs_index[new_pair].add(new_pre_token)
                # if new_pair == best_pair:
                changed_pairs.add(new_pair)
                j += 1
        total_merge_time += time.perf_counter() - t0

        for pair in changed_pairs:
            current_freq = pairs_count.get(pair)

            if current_freq is not None and current_freq > 0:
                heapq.heappush(
                    pair_heap,
                    HeapItem(current_freq, pair),
                )

        if loop % 1000 == 0:
            print(f'========= loop {loop} is done =========')
            print_memory(f"at{loop}th loop,")
            print(f"last 1000 loop costs: {time.perf_counter() - loop_100_start_time:.2f} seconds")
            loop_100_start_time = time.perf_counter()
        loop += 1
    elapsed = time.perf_counter() - start
    print('======== summary ========')
    print(f"BPE Pre-tokenization time: {elapsed:.2f} seconds")
    print(f"BPE Merge BPE time: {elapsed:.2f} seconds")
    print('========= detail =========')
    print(f"BPE Find Max total time: {total_find_max_time:.2f} seconds")
    print(f"BPE Merge total time: {total_merge_time:.2f} seconds")
    total_pair_updates = num_old_pair_updates + num_new_pair_updates
    print(f"averge total pair update per affected pre-token:, { total_pair_updates / num_affected_token:2f}")
    print(f"avg merges per affected pre-token:, {num_merged_occurrences/num_affected_token: 2f}" )
    return vocab, merges



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
        num_chuncks = 40
        boundaries = find_chunk_boundaries(f, num_chuncks, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            # chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            tasks.append((start, end, input_path, special_tokens))

    num_processes = 6
    # 2. Run workers
    num_workers = min(os.cpu_count() // 2, num_processes)
    pre_token_counts = {}
    
    start = time.perf_counter()
    
    with Pool(processes=num_workers) as pool:
        # worker_results = pool.starmap(worker, tasks)
        results = pool.imap_unordered(worker_wrapper, tasks)
        for local_counts in results:
            for token, count in local_counts.items():
                pre_token_counts[token] = pre_token_counts.get(token, 0) + count

    print_memory("workes are done, merge results from workers")
    # 3. Merge results

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

    
    num_old_pair_updates = 0
    num_new_pair_updates = 0
    num_merged_occurrences = 0
    num_affected_token = 0

    total_find_max_time = 0
    total_merge_time = 0

    loop = 0
    loop_100_start_time = time.perf_counter()
    while len(vocab) < vocab_size:
        
        if len(pairs_count) == 0:
            return vocab, merges
        t0 = time.perf_counter()
        best_pair, best_freq = max(pairs_count.items(), key=lambda item: (item[1], item[0]))
        total_find_max_time += time.perf_counter() - t0

        if best_freq == 0:
            return vocab
        new_token = best_pair[0] + best_pair[1]

        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        affected_per_tokens = pairs_index[best_pair].copy()
        # if 6640 <= loop <= 6720:

        if len(new_token) > 2000:
            print("best freq:", best_freq)
            print("left len:", len(best_pair[0]))
            print("right len:", len(best_pair[1]))
            print("new token len:", len(new_token))

            sys.exit(-1)

        new_pre_tokens = set()

        t0 = time.perf_counter()
        for affected_token in affected_per_tokens:
            # sum_affect_token_len += len(affected_token)
            num_affected_token += 1
            freq = frequency_table[affected_token]

            new_pre_token = []

            idx = 0
            for i in range(len(affected_token)-1):
        
                old_pair = (affected_token[i], affected_token[i+1])
   
                pairs_count[old_pair] -= freq
                num_old_pair_updates += 1


                if pairs_count[old_pair] == 0 :
                    del pairs_count[old_pair]
                pairs_index[old_pair].discard(affected_token)

                # sum_pair_count_change += 1

                if len(pairs_index[old_pair]) == 0:
                    del pairs_index[old_pair]
                if idx != i:
                    continue
                if old_pair == best_pair:
                    new_pre_token.append(new_token)
                    idx += 2

                    num_merged_occurrences += 1
                else:
                    new_pre_token.append(affected_token[i])
                    idx += 1

            if idx == len(affected_token) - 1:
                new_pre_token.append(affected_token[idx])
            new_pre_tokens.add(tuple(new_pre_token))

            frequency_table[tuple(new_pre_token)] = frequency_table.get(tuple(new_pre_token), 0) + freq
            frequency_table[affected_token] -= freq
            if frequency_table[affected_token] == 0:
                del  frequency_table[affected_token]

        for new_pre_token in new_pre_tokens:

            j = 0
            while j < len(new_pre_token)-1:
                new_pair = (new_pre_token[j], new_pre_token[j+1])
                # print("new_pair:", new_pair)
                pairs_count[new_pair] = pairs_count.get(new_pair, 0) + frequency_table[new_pre_token] 

                num_new_pair_updates += 1
                # print("new pair counts:", pairs_count[new_pair])
                pairs_index[new_pair].add(new_pre_token)
                # if new_pair == best_pair:
                j += 1

        total_merge_time += time.perf_counter() - t0

        if loop % 1000 == 0:
            print(f'========= loop {loop} is done =========')
            print_memory(f"at{loop}th loop,")
            print(f"last 1000 loop costs: {time.perf_counter() - loop_100_start_time:.2f} seconds")
            loop_100_start_time = time.perf_counter()
        loop += 1
    elapsed = time.perf_counter() - start
    print('======== summary ========')
    print(f"BPE Pre-tokenization time: {elapsed:.2f} seconds")
    print(f"BPE Merge BPE time: {elapsed:.2f} seconds")
    print('========= detail =========')
    print(f"BPE Find Max total time: {total_find_max_time:.2f} seconds")
    print(f"BPE Merge total time: {total_merge_time:.2f} seconds")
    total_pair_updates = num_old_pair_updates + num_new_pair_updates
    print(f"averge total pair update per affected pre-token:, { total_pair_updates / num_affected_token:2f}")
    print(f"avg merges per affected pre-token:, {num_merged_occurrences/num_affected_token: 2f}" )
    return vocab, merges