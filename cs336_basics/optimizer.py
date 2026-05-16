from __future__ import annotations

import torch, math
from collections.abc import Callable


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"非法的 learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)  # 获取迭代的编号，没有则置为0
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad  # 原地更新参数
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,  # 需要被优化的参数
        # 以下都是超参数
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        if lr < 0:
            raise ValueError(f"非法的 learning rate: {lr}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        # 清空梯度，前向传播计算损失 loss，调用 loss.backward() 计算梯度，返回 loss
        loss = None if closure is None else closure()

        # param_groups里是所有参数的权重列表，实际训练时不同层可能用不同的超参
        # group里有params和defaults，即可训练参数和超参数
        for group in self.param_groups:

            # 取超参数
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # 可学习参数
            for p in group["params"]:
                if p.grad is None:
                    continue

                # 取参数，如果是第一次遇到就初始化
                state = self.state[p]
                t = state.get("t", 0)
                m = state.get("m", torch.zeros_like(p))  # first moment，一阶动量估计
                v = state.get("v", torch.zeros_like(p))  # second moment，二阶动量估计

                # 当前梯度
                grad = p.grad.data

                # 更新步数，t是从1开始，我们初始化的t是从0开始
                t = t + 1

                # 权重衰减weight decay
                p.data -= lr * weight_decay * p.data

                # 更新m和v
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad**2)

                # 计算当前步的修正步长
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # 更新参数
                p.data -= lr_t * m / (torch.sqrt(v + eps))
                state["m"] = m
                state["v"] = v
                state["t"] = t

        return loss


# 余弦退火
def learning_rate_cosine_schedule(
    t: int, lr_max: float, lr_min: float, T_w: int, T_c: int
):
    if t < 0:
        raise ValueError(f"非法的 t: {t}")
    if t < T_w:  # 热身
        return t / T_w * lr_max  # python里除法默认返回float类型
    elif t <= T_c:  # 余弦退火
        return (
            lr_min
            + (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (lr_max - lr_min) / 2
        )
    else:  # 退火后
        return lr_min
