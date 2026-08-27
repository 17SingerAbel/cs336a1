import json
import ast

vocab_filepath = 'output/owt_train-vocab-object.json'
with open(vocab_filepath, encoding="utf-8") as f:
        raw_vocab = json.load(f)

vocab = {
    token_id: token_object["bytes"]
        for token_id, token_object in raw_vocab.items()
}

print(vocab)

with open(f"output/owt_train-vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

