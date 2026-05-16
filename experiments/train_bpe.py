# 在大数据集上训练bpe

import json, os

from pathlib import Path

from cs336_basics.tokenizer import run_train_bpe, gpt2_bytes_to_unicode

_PROJECT_DIR = Path(__file__).resolve().parents[1]


#
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    output_path: str | os.PathLike,
    output_prefix: str | None,  # 保存文件的前缀名
):
    # 不管在哪里运行，都能正确读取根目录下的文件
    input_path = Path(input_path)
    if not input_path.is_absolute():
        input_path = _PROJECT_DIR / input_path

    # 训练
    vocab, merges = run_train_bpe(
        input_path,
        vocab_size,
        special_tokens,
        num_workers=4,
        mini_chunk_size=4096,
    )
    print()

    # 按照 gpt-2格式保存vocab和merges
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_DIR / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    vocab_output_path = output_path / f"{output_prefix}_vocab.json"
    merges_output_path = output_path / f"{output_prefix}_merges.txt"
    # print(vocab_output_path)

    # vocab: dict[int, bytes]
    # saved_vocab: dict[str,int]
    # 构建 byte-> unicode 转换器
    byte_encoder = gpt2_bytes_to_unicode()
    saved_vocab = {}
    for vocab_id, vocab_bytes in vocab.items():
        encoded_chars = []
        # vocab_byte是int，查询byte_encoder内的key获得对应str，再拼接成一个str保存下来
        for vocab_byte in vocab_bytes:
            encoded_chars.append(byte_encoder[vocab_byte])
        token_str = "".join(encoded_chars)
        saved_vocab[token_str] = vocab_id
    # print(saved_vocab)
    try:
        with open(vocab_output_path, "w", encoding="utf-8") as f:
            json.dump(saved_vocab, f, ensure_ascii=False, indent=2)
        print(f"vocab 保存成功: {vocab_output_path}")
    except Exception as e:
        print(f"vocab 保存失败！！！: {vocab_output_path}")
        print(f"错误信息: {e}")
    # merges: list[tuple[bytes,bytes]]
    # saved_merges: list[str]
    saved_merges = []
    for left_bytes, right_bytes in merges:
        left_str = []
        for merges_byte in left_bytes:
            left_str.append(byte_encoder[merges_byte])
        left_str = "".join(left_str)

        right_str = []
        for merges_byte in right_bytes:
            right_str.append(byte_encoder[merges_byte])
        right_str = "".join(right_str)
        token_str = left_str + " " + right_str
        saved_merges.append(token_str)
    # print(f"saved_merges: {saved_merges}")
    try:
        with open(merges_output_path, "w", encoding="utf-8") as f:
            for merge_line in saved_merges:
                f.write(merge_line + "\n")
        print(f"merges 保存成功: {merges_output_path}")
    except Exception as e:
        print(f"merges 保存失败！！！: {merges_output_path}")
        print(f"错误信息: {e}")


if __name__ == "__main__":

    input_path = "data/raw/TinyStoriesV2-GPT4-train.txt"
    output_path = "data/output"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    output_path = "data/output"
    output_prefix = Path(input_path).stem

    train_bpe(input_path, vocab_size, special_tokens, output_path, output_prefix)
