from __future__ import annotations

import torch
from .nn_utils import softmax


# 推理
def decoding(
    model: TransformerLM,
    token_ids: list[int],
    max_next_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
    device: torch.device | None = None,
) -> list[int]:
    model.eval()
    with torch.no_grad():

        # 添加一个batch维度，方便喂给model
        token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        token_ids = token_ids.unsqueeze(0)

        next_token_count = 0
        # 循环直到生成足够多的token
        while next_token_count < max_next_tokens:

            # logits.shape == (batch, seq_len, vocab_size)
            logits = model(token_ids)
            next_token_logits = logits[
                0, -1, :
            ]  # 取seq_len的最后一维，即对最后一个位置计算概率,(vocab_size,)

            # 对vocab_size维做softmax
            next_token_logits = next_token_logits / temperature  # 温度参数调节
            probs = softmax(next_token_logits, dim=-1)  # probs.shape == (vocab_size, )

            # top_p
            sorted_probs, sorted_indices = torch.sort(
                probs, descending=True
            )  # sorted_probs.shape == (vocab_size, )

            cumulative_probs = torch.cumsum(sorted_probs, dim=0)  # 计算累积和
            mask = cumulative_probs > top_p  # 找到大于top_p的位置
            mask[1:] = mask[:-1].clone()  # 对mask最后一维右移一格
            mask[0] = False
            # mask.shape == (vocab_size, )

            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum()
            next_token_id = torch.multinomial(
                sorted_probs, num_samples=1
            )  # 返回的是在重排序数组sorted_probs里的位置

            next_token_id = sorted_indices[next_token_id]  # 真实的token_id
            next_token_id = next_token_id.unsqueeze(0)  # 转变为（1, 1)

            # 如果产生终止符，退出生成
            if next_token_id.item() == eos_token_id:
                break

            # 把下一个token拼接回去
            token_ids = torch.cat([token_ids, next_token_id], dim=1)

            next_token_count += 1
        return token_ids.squeeze(0).tolist()
