from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int


def softmax(x: Float[Tensor, " ..."], dim: int):
    # 求出最大元素，并从x中减去，[0]是最大值，[1]是最大值所在的位置索引
    max_vals = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - max_vals

    # 计算指数并广播除法
    exp_x = torch.exp(x)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    output = exp_x / sum_exp
    return output


# 支持多维张量
def cross_entropy(
    inputs: Float[Tensor, "... batch_size vocab_size"],
    targets: Int[Tensor, "... batch_size"],
) -> Float[Tensor, ""]:

    # 将多维向量展开
    inputs = inputs.reshape(-1, inputs.shape[-1])
    targets = targets.reshape(-1)

    # 从最后一维vocab_size里取出最大值，全体减去最大值再求和
    # m = max(o)
    max_vals = torch.max(inputs, dim=-1, keepdim=True)[0]
    # o[a] − m
    inputs = inputs - max_vals
    # log( sum(...)...
    logsumexp = torch.log(  # logsumexp.shape == (batch,)
        torch.sum(torch.exp(inputs), dim=-1)
    )

    # o[y]−m
    # targtes 里每个元素都对应 x_{i+1}
    # 即inputs从0开始遍历每一行，每一行都取对应targets的内容
    target_logit = inputs[  # target_logit.shape == (batch,)
        torch.arange(inputs.shape[0]), targets
    ]

    # 最后损失
    loss = logsumexp - target_logit
    return torch.mean(loss)  # 对整个 batch 取平均，也就是 /m


# 困惑度，对交叉熵损失返回值取exp
def perplexity(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    return torch.exp(cross_entropy(inputs, targets))


# 梯度裁减
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6
):
    # 转换成list，防止生成器只能迭代一次
    params = list(parameters)
    # 累加梯度，grad是tensor
    grad_sum = 0
    for param in params:
        if param.grad is None:
            continue

        # 计算整体的梯度范式
        grad = param.grad.data
        grad_sum += torch.sum(grad**2)  # 返回的是一个标量，0 维 tensor

    l2_norm = math.sqrt(grad_sum)
    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for param in params:
            if param.grad is None:
                continue
            param.grad *= scale
