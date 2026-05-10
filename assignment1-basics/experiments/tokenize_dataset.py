# 输出文本文件路径
# 利用分词器生成rain_tokens.npy和valid_tokens.npy

from tests.adapters import Tokenizer
from pathlib import Path
from tqdm import tqdm
import numpy as np


# 配置信息
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
TOKENIZER_DIR = DATA_DIR / "tokenizer" / "tinystories"
TOKENS_DIR = DATA_DIR / "tokens" / "tinystories"

TRAIN_TEXT_PATH = RAW_DIR / "TinyStoriesV2-GPT4-train.txt"
VALID_TEXT_PATH = RAW_DIR / "TinyStoriesV2-GPT4-valid.txt"

VOCAB_PATH = TOKENIZER_DIR / "vocab.json"
MERGES_PATH = TOKENIZER_DIR / "merges.txt"

TRAIN_TOKENS_PATH = TOKENS_DIR / "train_tokens.npy"
VALID_TOKENS_PATH = TOKENS_DIR / "valid_tokens.npy"

SPECIAL_TOKENS = ["<|endoftext|>"]

# 分词器处理训练文本
tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, SPECIAL_TOKENS)
# 创建父文件夹
TOKENS_DIR.mkdir(parents=True, exist_ok=True)

# 训练集
total_bytes = TRAIN_TEXT_PATH.stat().st_size
with open(TRAIN_TEXT_PATH, "r", encoding="utf-8") as f, tqdm(
    total=total_bytes,
    unit="B",
    unit_scale=True,
    desc=TRAIN_TEXT_PATH.name,
) as pbar:
    train_token_ids = list(tokenizer.encode_iterable(f, pbar=pbar))
train_token_ids = np.array(train_token_ids, dtype=np.int32)
try:
    np.save(TRAIN_TOKENS_PATH, train_token_ids)
    print(f"训练集转换成功: {TRAIN_TOKENS_PATH}\n")
except Exception as e:
    print(f"训练集转换失败！！！")
    print(f"错误信息: {e}\n")

# 测试集
total_bytes = VALID_TEXT_PATH.stat().st_size
with open(VALID_TEXT_PATH, "r", encoding="utf-8") as f, tqdm(
    total=total_bytes,
    unit="B",
    unit_scale=True,
    desc=VALID_TEXT_PATH.name,
) as pbar:
    valid_token_ids = list(tokenizer.encode_iterable(f, pbar=pbar))
valid_token_ids = np.array(valid_token_ids, dtype=np.int32)
try:
    np.save(VALID_TOKENS_PATH, valid_token_ids)
    print(f"验证集转换成功: {TRAIN_TOKENS_PATH}\n")
except Exception as e:
    print(f"验证集转换失败！！！")
    print(f"错误信息: {e}\n")
