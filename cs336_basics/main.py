from bpe import train_bpe, train_bpe_heap
import time
import json


# string = "ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss"
# print(len(string))

# dic = (b'P', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss'), (b'S', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss', b'ss')
# print(len(dic[0]))
# print(len(dic[1]))
start = time.perf_counter()
# vocab, merges = train_bpe(input_path='data/owt_train.txt', vocab_size=32000,special_tokens=['<|endoftext|>'])

# input_filepath = 'smallest.txt'
# input_filepath = 'TinyStoriesV2-GPT4-train.txt'
input_filepath = 'owt_valid.txt'
# input_filepath = 'owt_train.txt'
input_file_prefix =  input_filepath.split('.')[0]
print(input_file_prefix)

vocab_heap, merges_heap = train_bpe_heap(input_path=f'data/{input_file_prefix}.txt', vocab_size=32000,special_tokens=['<|endoftext|>'])

# vocab, merges = train_bpe(input_path=f'data/{input_file_prefix}.txt', vocab_size=32000,special_tokens=['<|endoftext|>'])

elapsed = time.perf_counter() - start
print(f"BPE Training total time: {elapsed:.2f} seconds")


# readable_vocab = {
#     key: {
#         "bytes": repr(value),
#         "text": value.decode("utf-8", errors="replace")
#     }
#     for key, value in vocab.items()
# }

# readable_merges = [
#     [repr(pairs[0]), repr(pairs[1])]
#     for pairs in merges
# ]

# with open(f"output/{input_file_prefix}-vocab.json", "w") as f:
#     json.dump(readable_vocab, f, indent=2)


# with open(f"output/{input_file_prefix}-merges.json", "w") as f:
#     json.dump(readable_merges, f, indent=2)




readable_vocab_heap = {
    str(key): {
        "bytes": repr(value),
        "text": value.decode("utf-8", errors="replace")
    }
    for key, value in vocab_heap.items()
}

readable_merges_heap = [
    [repr(pairs[0]), repr(pairs[1])]
    for pairs in merges_heap
]

with open(f"output/{input_file_prefix}-heap-vocab.json", "w") as f:
    json.dump(readable_vocab_heap, f, indent=2)


with open(f"output/{input_file_prefix}-heap-merges.json", "w") as f:
    json.dump(readable_merges_heap, f, indent=2)






# with open(f"output/{input_file_prefix}-vocab.json", encoding="utf-8") as f:
#     expected_vocab = json.load(f)

# with open(f"output/{input_file_prefix}-merges.json", encoding="utf-8") as f:
#     expected_merges = json.load(f)


# # with open(f"output/{input_file_prefix}-heap-vocab.json", encoding="utf-8") as f:
# #     readable_vocab_heap = json.load(f)

# # with open(f"output/{input_file_prefix}-heap-merges.json", encoding="utf-8") as f:
# #     readable_merges_heap = json.load(f)


# # assert readable_vocab_heap.keys() == expected_vocab.keys(), (
# #     "Vocab keys do not match"
# # )

# # for key in readable_vocab_heap:
# #     assert readable_vocab_heap[key] == expected_vocab[key], (
# #         f"Vocab mismatch at key {key}:\n"
# #         f"current:  {readable_vocab_heap[key]}\n"
# #         f"expected: {expected_vocab[key]}"
# #     )


# assert expected_merges == readable_merges_heap, "Merges do not match"
# assert expected_vocab == readable_vocab_heap, "Vocab does not match"


# print("Vocab and merges match.")


