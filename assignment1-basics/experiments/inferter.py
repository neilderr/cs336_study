import torch
from tests.adapters import TransformerLM, Tokenizer, AdamW, decoding, load_checkpoint
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TOKENIZER_DIR = DATA_DIR / "tokenizer" / "tinystories"
VOCAB_PATH = TOKENIZER_DIR / "vocab.json"
MERGES_PATH = TOKENIZER_DIR / "merges.txt"
SPECIAL_TOKENS = ["<|endoftext|>"]
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "lm_step_440.pt"


vocab_size = 10000
context_length = 256
d_model = 512
num_layers = 4
num_heads = 16
d_ff = 1344
rope_theta = 10000.0

batch_size = 32
max_steps = 1000
lr_max = 3e-4
lr_min = 3e-5
T_w = 50
T_c = max_steps
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.1
max_l2_norm = 1.0

max_nex_tokens = 128
temperature = 1.0
top_p = 0.8

prompt = "Hello World"
eos_token = "<|endoftext|>"

# 确定设备
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = TransformerLM(
    vocab_size,
    context_length,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    rope_theta,
    device=device,
)
optimizer = AdamW(
    model.parameters(),
    lr=lr_max,
    betas=betas,
    eps=eps,
    weight_decay=weight_decay,
)
load_checkpoint(
    src=MODEL_PATH,
    model=model,
    optimizer=optimizer,
)

tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, SPECIAL_TOKENS)
token_ids = tokenizer.encode(prompt)
eos_token_id = tokenizer.encode(eos_token)[0]

token_ids = decoding(
    model, token_ids, max_nex_tokens, temperature, top_p, eos_token_id, device=device
)
print(tokenizer.decode(token_ids))
