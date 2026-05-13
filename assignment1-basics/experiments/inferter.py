import torch
from tests.adapters import TransformerLM, Tokenizer, AdamW, decoding, load_checkpoint
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TOKENIZER_DIR = DATA_DIR / "tokenizer" / "tinystories"
VOCAB_PATH = TOKENIZER_DIR / "vocab.json"
MERGES_PATH = TOKENIZER_DIR / "merges.txt"
SPECIAL_TOKENS = ["<|endoftext|>"]
runs_name = "run_001"
MODEL_PATH = PROJECT_ROOT / "runs" / runs_name / "best.pt"
CONFIG_PATH = PROJECT_ROOT / "experiments" / "config.json"


# 读取配置参数
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

max_next_tokens = config["max_next_tokens"]
temperature = config["temperature"]
top_p = config["top_p"]

prompt = "Once upon a time there was a little boy named Ben. "
eos_token = "<|endoftext|>"

# 确定设备
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = TransformerLM(
    vocab_size=config["vocab_size"],
    context_length=config["context_length"],
    d_model=config["d_model"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    d_ff=config["d_ff"],
    rope_theta=config["rope_theta"],
    device=device,
)

optimizer = AdamW(
    model.parameters(),
    lr=config["lr_max"],
    betas=tuple(config["betas"]),
    eps=config["eps"],
    weight_decay=config["weight_decay"],
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
    model, token_ids, max_next_tokens, temperature, top_p, eos_token_id, device=device
)
print(tokenizer.decode(token_ids))
