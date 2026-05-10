from pathlib import Path
import time
import numpy as np
import torch
from sys import exception

from tests.adapters import (
    TransformerLM,
    AdamW,
    cross_entropy,
    data_loader,
    learning_rate_cosine_schedule,
    gradient_clipping,
    save_checkpoint,
)

# 配置超参数
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"
tokenizer_dir = data_dir / "tokenizer" / "tinystories"
tokens_dir = data_dir / "tokens" / "tinystories"
checkpoints_dir = project_root / "checkpoints"

train_tokens_path = tokens_dir / "train_tokens.npy"
valid_tokens_path = tokens_dir / "valid_tokens.npy"

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

log_interval = 1
eval_interval = 10
save_interval = 10

# 创建文件夹
checkpoints_dir.mkdir(parents=True, exist_ok=True)


# 确定设备
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# 加载npy数据
train_tokens = np.load(train_tokens_path, mmap_mode="r")
valid_tokens = np.load(valid_tokens_path, mmap_mode="r")

# 创建模型和优化器
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

start_time = time.time()
d_last = start_time
# 训练循环
for step in range(1, max_steps + 1):
    # 告诉PyTorch，现在进入训练模式
    model.train()
    # 从训练集里随机采样，计算训练损失
    x, y = data_loader(train_tokens, batch_size, context_length, device=device)
    logits = model(x)

    loss = cross_entropy(logits, y)
    ppl = torch.exp(loss)

    dt = time.time() - d_last
    d_last = time.time()

    # 更新学习率
    lr = learning_rate_cosine_schedule(
        t=step,  # t从1开始
        lr_max=lr_max,
        lr_min=lr_min,
        T_w=T_w,
        T_c=T_c,
    )
    optimizer.param_groups[0]["lr"] = lr

    print(
        f"[train ] step = {step}, loss = {loss.item():.4f}, ppl = {ppl.item():.2f}, lr = {lr:.2e}, tok/s = {(batch_size * context_length) / dt:.2f}, 运行时间 = {(time.time()-start_time):.2f} 秒"
    )

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    gradient_clipping(model.parameters(), max_l2_norm)  # 梯度裁减
    optimizer.step()

    # B.验证日志
    # 每 eval_interval 打印一次
    if step % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            x, y = data_loader(valid_tokens, batch_size, context_length, device=device)
            logits = model(x)
            loss = cross_entropy(logits, y)
            ppl = torch.exp(loss)
            print(
                f"\n\n[eval ] step = {step}, loss = {loss.item():.4f}, ppl = {ppl.item():.2f}, lr = {lr:.2e}, 运行时间 = {(time.time()-start_time):.2f} 秒\n\n"
            )

    # checkpoint 日志
    if step % save_interval == 0:
        checkpoints_path = checkpoints_dir / f"lm_step_{step}.pt"
        try:
            save_checkpoint(model, optimizer, step, out=checkpoints_path)
            print(f"[save ] step = {step}, checkpoint = {checkpoints_path}\n\n")
        except Exception as e:
            print(f"[save ] 失败！！！ 错误原因: {e}\n\n")
