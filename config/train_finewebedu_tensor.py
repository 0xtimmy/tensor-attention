# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-proj'
eval_interval = 300 # keep frequent because we'll overfit
eval_iters = 10
log_interval = 30 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

dataset = 'fineweb-edu'
gradient_accumulation_steps = 5
batch_size = 12
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 8*64
dropout = 0.2

learning_rate = 6e-4 # with baby networks can afford to go a bit higher
max_iters = 100000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 6e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 2 # not super necessary potentially

device = "cuda"

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

attn_impl = "tensor"