from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def data_loader(
    x: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 随机起点，np取不到右区间
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)

    inputs = []
    targets = []

    for start in starts:
        inputs.append(x[start : start + context_length])
        targets.append(x[start + 1 : start + context_length + 1])
    # 先转换成二维 NumPy 数组，再转换成tensor。比直接从list转换成tensor速度快
    inputs = np.array(inputs, dtype=np.int64)
    targets = np.array(targets, dtype=np.int64)
    # 默认为这些 token IDs 使用 int64 存储
    inputs = torch.from_numpy(inputs).to(device)
    targets = torch.from_numpy(targets).to(device)

    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    check_point = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(check_point, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device | None = None,
):
    check_point = torch.load(src, map_location=map_location)
    model.load_state_dict(check_point["model"])
    optimizer.load_state_dict(check_point["optimizer"])

    return check_point["iteration"]
