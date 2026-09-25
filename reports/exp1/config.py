import torch
# actually ran 75k
vocab_size = 10000
context_length = 256
d_model = 512
num_layers = 4
num_heads = 16
d_ff = 1344

batch_size=32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype=torch.bfloat16

rope_theta = 10000
betas = [0.9, 0.95]
weight_decay = 1


iterations = 10000

# learning rate hyperparam
max_lr = 1e-2
min_lr = 1e-5
warmup_iters = iterations * 0.2
cosine_cycle_iters = iterations * 0.9
max_l2_norm = 1


top_p_threshold = 0.8