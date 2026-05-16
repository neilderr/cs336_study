import regex as re
import os, resource, sys, time, json

from collections import OrderedDict
from tqdm import tqdm
from torch import nn
from multiprocessing import Pool
from typing import BinaryIO, Iterable


class Tokenizer:
    # 初始化tokenizer对象
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # 把merge建成dict，减少encode阶段时间消耗
        self.merge_ranks = {merge: rank for rank, merge in enumerate(merges)}

        # 把special_tokens从长到短排序，方便后续贪心匹配
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)

            # 构建分词模式
            escaped_tokens = []
            for token in self.special_tokens:
                escaped_token = re.escape(token)
                escaped_tokens.append(escaped_token)
            self.pattern = re.compile("(" + "|".join(escaped_tokens) + ")")

        else:
            self.special_tokens = []

        # vocab的反查链表
        self.bytes_to_id = {}
        for key, value in vocab.items():
            self.bytes_to_id[value] = key

        # 一些简短str对应的token_id，能极大减少encode时间
        self.cache = OrderedDict()
        self.max_cache_size = len(vocab) * 5  # 默认是词表大小的5倍

    # 根据文件初始化
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # 读取的vocab: token_str -> token_id, 这里token_str并非直接的bytes转变的字符串，
        # 而是通过gpt2的方式映射后的str，目的是让所有byte都可打印可存文件
        # 因此需要先转置vocab，同时把str里每个字符通过byte_decoder转换为实际byte对应的int，再组合成bytes填入vocab
        # merges: str -> str, 同理
        # 读取文件
        with open(vocab_filepath) as vocab_f:
            vocab = json.load(vocab_f)
        merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()  # 删除末尾换行符和空白字符
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    merges.append(tuple(cleaned_line.split(" ")))

        # 构建 unicode-> byte 转换器
        byte_encoder = gpt2_bytes_to_unicode()
        byte_decoder = {}  # decoder即 encoder的转置
        for byte_value, unicode_char in byte_encoder.items():
            byte_decoder[unicode_char] = byte_value
        # vocab转置并恢复原始 bytes
        vocab_t = {}
        for vocab_item, vocab_index in vocab.items():
            byte_list = []
            for char in vocab_item:
                byte_list.append(byte_decoder[char])
            vocab_t[vocab_index] = bytes(byte_list)
        vocab = vocab_t
        # 确保所有special_tokens都在vocab内
        # vocab是dict[int,bytes]
        if special_tokens:
            for special_token in special_tokens:
                byte_endcoded_special_token = special_token.encode("utf-8")
                if byte_endcoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_endcoded_special_token
            pass
        # merges内容恢复原始bytes
        merges_t = []
        for merge_token1, merge_token2 in merges:
            merges_t.append(
                (
                    bytes([byte_decoder[token] for token in merge_token1]),
                    bytes([byte_decoder[token] for token in merge_token2]),
                )
            )
        merges = merges_t
        # for k, v in list(vocab.items())[:90]:
        #     print(k, v)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # pre-tokenizer预分词
        # 处理special_tokens
        if self.special_tokens:
            parts = self.pattern.split(text)
        else:
            parts = [text]
        # print(f"special_tokens处理后: {parts}")
        # 清除空格
        filtered_parts = []
        for part in parts:
            if part:
                filtered_parts.append(part)
        parts = filtered_parts

        # 分流普通文本和special_tokens文本，获得对应 token_id
        ids = []
        for part in parts:
            if part not in self.special_tokens:
                # 按gpt-2的正则表达式分词
                part = self.pat.findall(part)
                # print(f"part = {part}")

                # 转换为bytes串
                for pre_token in part:
                    # 如果命中cache，直接加入ids
                    if pre_token in self.cache:
                        ids.extend(self.cache[pre_token])  # 添加的是一串token_id
                        continue

                    token_bytes = pre_token.encode("utf-8")
                    seq = tuple(bytes([b]) for b in token_bytes)
                    # seq = (b'h', b'e', b'l', b'l', b'o')

                    # 合并操作
                    while True:
                        pairs = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]

                        # 找到可以合并的pair和对应的rank
                        valid_pairs = [
                            pair for pair in pairs if pair in self.merge_ranks
                        ]

                        # 如果没有可以合并的就退出
                        if not valid_pairs:
                            break

                        # 找出rank最小，即优先级最大的pair
                        best_pair = min(
                            valid_pairs, key=lambda pair: self.merge_ranks[pair]
                        )
                        new_seq = []
                        i = 0
                        seq_len = len(seq)
                        while i < seq_len:
                            if i < seq_len - 1 and (seq[i], seq[i + 1]) == best_pair:
                                token = seq[i] + seq[i + 1]
                                new_seq.append(token)
                                i += 2
                            else:
                                new_seq.append(seq[i])
                                i += 1
                        # 更新seq
                        seq = tuple(new_seq)

                    # 注意这里append和extend的使用
                    pretoken_ids = []

                    for token_bytes in seq:
                        pretoken_ids.append(self.bytes_to_id[token_bytes])
                    ids.extend(pretoken_ids)
                    # 存入缓存
                    self.put_cache(pre_token, pretoken_ids)

                    # print()
            else:
                # print(f"special类型: {type(part)}")
                ids.append(self.bytes_to_id[part.encode("utf-8")])
                # print()
        # print(ids)
        return ids

    # 返回值是Iterable，所以得用yield一点点返回
    def encode_iterable(
        self,
        iterable: Iterable[str],
        pbar=None,
    ) -> Iterable[int]:
        for text in iterable:
            ids = self.encode(text)

            if pbar is not None:
                pbar.update(len(text.encode("utf-8")))

            for token_id in ids:
                yield token_id

    # ids解码为string
    def decode(self, ids: list[int]) -> str:
        token_bytes_list = []
        for token_id in ids:
            token_bytes = self.vocab[token_id]
            token_bytes_list.append(token_bytes)
        decoded_bytes = b"".join(token_bytes_list)
        # 某些单独 token 的 bytes 不一定构成合法 UTF-8，使用 replace 避免 decode 时报错
        decoded_text = decoded_bytes.decode("utf-8", errors="replace")
        # print(decoded_text)
        return decoded_text

    # cache调度策略，FIFO
    def put_cache(self, pre_token: str, pretoken_ids: list[int]):
        # 大于最大容量，执行FIFO策略,删除最老的
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)

        self.cache[pre_token] = pretoken_ids

    # 测试用，将string转换为vocab内的token_id
    def give_ids(self, text) -> list[int]:
        ids = []
        for target in text:
            target = target.encode("utf-8")
            # print(f"finding: {target}")
            for token_id, token_bytes in self.vocab.items():
                if token_bytes == target:
                    ids.append(token_id)
        return ids


# macOS和Linux上有不同的内存单位
def get_peak_memory_mb() -> float:
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return memory / 1024 / 1024
    return memory / 1024


# 根据str的内容预分词并更新pretoken_counts，适用流式读取文件
def update_pretoken_counts_from_str(
    text: str,
    pretoken_counts: dict[tuple[bytes, ...], int],
) -> None:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    tokens = re.findall(PAT, text)

    for token in tokens:
        token_bytes = token.encode("utf-8")
        token_tuple = tuple(bytes([b]) for b in token_bytes)
        pretoken_counts[token_tuple] = pretoken_counts.get(token_tuple, 0) + 1


# 找到文件的安全边界，接收二进制文件
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,  # 总共想切几块
    split_special_token: bytes,
    mini_chunk_size: int = 4096,
) -> list[int]:
    assert isinstance(split_special_token, bytes)

    # 文件移动到指针末尾
    file.seek(0, os.SEEK_END)
    # 返回当前文件指针的位置，也就是文件总字节数
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            # 如果读入的chunk为空
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))


# worker运行的内容
def process_chunk(
    args: tuple[str, int, int, list[str]],
) -> dict[tuple[bytes, ...], int]:
    input_path, start, end, special_tokens = args

    local_counts = {}

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")

    if special_tokens:
        escaped_tokens = []
        for token in special_tokens:
            escaped_tokens.append(re.escape(token))
        escaped_tokens.sort(key=len, reverse=True)
        pattern = "|".join(escaped_tokens)

        parts = re.split(pattern, chunk_text)
        for part in parts:
            if part:
                update_pretoken_counts_from_str(part, local_counts)
    else:
        update_pretoken_counts_from_str(chunk_text, local_counts)

    return local_counts


# 给定输入语料库的路径，训练一个BPE分词器，并输出其词汇表和合并规则。
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # 从kwargs里取出合并模式，默认用增量方式
    merge_mode = kwargs.get("merge_mode", "incremental")

    # 启动计时器
    start_time = time.time()
    phase_start_time = start_time

    # 读取训练语料
    print(f"[读入文件 & 并行预分词]\n文件路径：{input_path}")
    pretoken_counts = {}
    # 并行化参数
    num_workers = kwargs.get("num_workers", 4)
    num_chunks = kwargs.get("num_chunks", num_workers * 8)
    mini_chunk_size = kwargs.get("mini_chunk_size", 4096)

    print(f"num_workers = {num_workers}")
    print(f"num_chunks = {num_chunks}")
    print(f"mini_chunk_size = {mini_chunk_size}")

    # 二进制打开文件，按照special_tokens切分文件
    # 返回一串边界，如 [0, 534000123, 1067000456, file_size]
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            num_chunks,
            b"<|endoftext|>",
            mini_chunk_size,
        )

    tasks = []
    # 相邻边界组合成区间
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((str(input_path), start, end, special_tokens))
    print(f"实际 chunk 数 = {len(tasks)}")

    progress_bar = tqdm(total=len(tasks), desc="Pretokenizing Chunks")

    with Pool(processes=num_workers) as pool:
        for local_counts in pool.imap_unordered(process_chunk, tasks):
            for token_tuple, count in local_counts.items():
                pretoken_counts[token_tuple] = (
                    pretoken_counts.get(token_tuple, 0) + count
                )
            progress_bar.update(1)
        progress_bar.close()

    phase_end_time = time.time()
    print(f"消耗时间: {phase_end_time-phase_start_time:.2f} 秒")
    print()

    # 初始化vocab和merge，内容为 256token + special
    print(f"[初始化词表]")
    phase_start_time = time.time()
    vocab = {}
    merges = []
    for token_id in range(256):
        vocab[token_id] = bytes([token_id])
    for token_str in special_tokens:
        vocab[len(vocab)] = token_str.encode("utf-8")
    print(f"词表长度: {len(vocab)}")
    print(f"目标词表长度: {vocab_size}")
    phase_end_time = time.time()
    print(f"消耗时间: {phase_end_time-phase_start_time:.6f} 秒")
    print()

    # merge阶段
    # 循环直至达到vocab_size
    print(f"[merge阶段]: {merge_mode} 模式")
    phase_start_time = time.time()
    print(f"vocab_size = {vocab_size}")
    # 进度条实现
    progress_bar = tqdm(total=vocab_size, desc="BEP merges", initial=len(vocab))

    # 增量法实现
    # 核心部分！！！
    # pair_to_seqs：某个pair出现在哪些seq里，每次获取best_pair后只需要修改其对应的seq
    # dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]，类似下面的结构
    """
    {
        (b't', b'h'): {
            (b't', b'h', b'e'),
            (b't', b'h', b'a', b't'),
        }
    }
    """
    # 开始完整遍历一次获取pair_count和pair_to_seqs
    pair_to_seqs = {}
    pair_count = {}
    for seq, freq in pretoken_counts.items():
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_count[pair] = pair_count.get(pair, 0) + freq
            # 如果这个pair没有对应的集合，就先创建一个新空集合
            if pair not in pair_to_seqs:
                pair_to_seqs[pair] = set()
            # 把seq加入pair对应的集合里
            pair_to_seqs[pair].add(seq)

    # merge循环
    while len(vocab) < vocab_size:
        # 如果训练集太小，会出现没有可以merge的pair_count，此时直接退出
        if not pair_count:
            progress_bar.close()
            print(f"没有可以merge的pair，退出训练\n词表长度: {len(vocab)}\n")
            break
        # 计算频次最高的pair，频次相同则按字典序排序
        best_pair, best_count = max(
            pair_count.items(), key=lambda item: (item[1], item[0])
        )
        # 更新merges和vocab
        merges.append(best_pair)
        merged_pair = best_pair[0] + best_pair[1]
        token_id = len(vocab)
        vocab[token_id] = merged_pair

        # affected_seqs代表受影响的seqs
        # 如果best_pair存在，拿到它对应的seq。否则返回空集合set()
        # 转成list是方便后续更改
        affected_seqs = list(pair_to_seqs.get(best_pair, set()))

        for old_seq in affected_seqs:
            freq = pretoken_counts.pop(old_seq)
            old_seq_len = len(old_seq)
            # 1. 先把 old_seq 对所有 pair 的贡献删掉
            # 逐位置更新 pair_count
            old_pairs_in_seq = set()
            for i in range(old_seq_len - 1):
                pair = (old_seq[i], old_seq[i + 1])
                # 删除old_seq对pair_count的贡献
                pair_count[pair] -= freq
                if pair_count[pair] <= 0:
                    del pair_count[pair]
                # 记录old_seq里出现过几种不同的pair
                old_pairs_in_seq.add(pair)

            # 按“不同 pair”更新 pair_to_seqs
            for pair in old_pairs_in_seq:
                # pair不在出现在old_seq里，因为old_seq将要删除
                pair_to_seqs[pair].discard(old_seq)
                if not pair_to_seqs[pair]:
                    del pair_to_seqs[pair]
            # 2. 构造 new_seq
            new_seq = []
            i = 0
            while i < old_seq_len:
                if i < old_seq_len - 1 and (old_seq[i], old_seq[i + 1]) == best_pair:
                    new_seq.append(merged_pair)
                    i += 2
                else:
                    new_seq.append(old_seq[i])
                    i += 1

            # 3. 把 new_seq 放回 pretoken_counts
            new_seq = tuple(new_seq)
            pretoken_counts[new_seq] = pretoken_counts.get(new_seq, 0) + freq
            # 4. 把 new_seq 对所有 pair 的贡献加回去
            for i in range(len(new_seq) - 1):
                pair = (new_seq[i], new_seq[i + 1])
                # 添加pair对pair_count的贡献
                pair_count[pair] = pair_count.get(pair, 0) + freq
                # pair出现在new_seq里
                if pair not in pair_to_seqs:
                    pair_to_seqs[pair] = set()
                pair_to_seqs[pair].add(new_seq)
        # 更新pretoken_counts和pair_counts
        progress_bar.update(1)

    progress_bar.close()

    # 打印时间和内存
    phase_end_time = time.time()
    print(f"消耗时间: {phase_end_time-phase_start_time:.2f} 秒")
    print()
    print(f"总训练耗时: {time.time()-start_time:.2f} 秒")

    peak_memory_mb = get_peak_memory_mb()
    print(f"训练峰值内存: {peak_memory_mb:.2f} MB")

    # 打印最长token
    longest_token_id, longest_token_bytes = max(
        vocab.items(), key=lambda item: len(item[1])
    )
    print()
    print(f"最长 token id: {longest_token_id}")
    print(f"最长 token bytes 长度: {len(longest_token_bytes)}")
    byte_encoder = gpt2_bytes_to_unicode()
    longest_token_str = "".join(byte_encoder[b] for b in longest_token_bytes)
    print(f"最长 token: {longest_token_str}")
    return vocab, merges


# gpt-2的bytes转换规则
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d
