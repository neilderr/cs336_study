from pathlib import Path
import time
import numpy as np
import torch
import json


from tests.adapters import (
    TransformerLM,
    AdamW,
    cross_entropy,
    data_loader,
    learning_rate_cosine_schedule,
    gradient_clipping,
    save_checkpoint,
    load_checkpoint,
)

# 配置超参数
project_root = Path(__file__).resolve().parents[1]

run_name = "run_001"
run_dir = project_root / "runs" / run_name
run_dir.mkdir(parents=True, exist_ok=True)

data_dir = project_root / "data"
tokens_dir = data_dir / "tokens" / "tinystories"
config_path = project_root / "experiments" / "config.json"

checkpoints_dir = run_dir
metrics_path = run_dir / "metrics.jsonl"
best_info_path = run_dir / "best_info.json"


train_tokens_path = tokens_dir / "train_tokens.npy"
valid_tokens_path = tokens_dir / "valid_tokens.npy"

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

# 读取配置参数
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

lr_max = config["lr_max"]
betas = tuple(config["betas"])
eps = config["eps"]
weight_decay = config["weight_decay"]
context_length = config["context_length"]

batch_size = config["batch_size"]
max_steps = config["max_steps"]
lr_min = config["lr_min"]
T_w = config["T_w"]
T_c = config["T_c"]
max_l2_norm = config["max_l2_norm"]
log_interval = config["log_interval"]
eval_interval = config["eval_interval"]
eval_steps = config["eval_steps"]

# 创建模型和优化器
model = TransformerLM(
    vocab_size=config["vocab_size"],
    context_length=context_length,
    d_model=config["d_model"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    d_ff=config["d_ff"],
    rope_theta=config["rope_theta"],
    device=device,
)
optimizer = AdamW(
    model.parameters(),
    lr=lr_max,
    betas=betas,
    eps=eps,
    weight_decay=weight_decay,
)


# 找到路径里的最近pt
def list_step_checkpoints(checkpoints_dir: Path) -> list[Path]:
    checkpoint_paths = sorted(
        checkpoints_dir.glob("lm_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return checkpoint_paths


# 保存日志
def append_metrics(metrics_path: Path, record: dict) -> None:
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# 加载best信息
def load_best_info(best_info_path: Path) -> dict | None:
    if not best_info_path.exists():
        return None

    with open(best_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


# 优化续训时的log信息
def truncate_metrics_after_step(metrics_path: Path, step: int) -> None:
    if not metrics_path.exists():
        return

    kept_records = []

    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            record_step = record.get("step")
            if record_step is None or record_step <= step:
                kept_records.append(record)

    with open(metrics_path, "w", encoding="utf-8") as f:
        for record in kept_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def truncate_best_info_after_step(best_info_path: Path, step: int) -> float:
    if not best_info_path.exists():
        return float("inf")

    with open(best_info_path, "r", encoding="utf-8") as f:
        best_info = json.load(f)

    best_step = best_info.get("step")
    if best_step is not None and best_step > step:
        best_info_path.unlink()
        return float("inf")

    return best_info["loss"]


# 开始训练
start_time = time.time()
d_last = start_time

all_checkpoints = list_step_checkpoints(checkpoints_dir)
recent_checkpoints = all_checkpoints[-3:]

best_val_loss = float("inf")
best_info = load_best_info(best_info_path)
if best_info is not None:
    best_val_loss = best_info["loss"]


# 尝试从之前的位置继续训练
start_step = 1
if all_checkpoints:
    latest_checkpoint = all_checkpoints[-1]
    loaded_step = load_checkpoint(
        src=latest_checkpoint,
        model=model,
        optimizer=optimizer,
    )

    truncate_metrics_after_step(metrics_path, loaded_step)
    best_val_loss = truncate_best_info_after_step(best_info_path, loaded_step)

    start_step = loaded_step + 1
    print(f"从 step = {start_step - 1} 加载 ")
else:
    print(f"从新训练")

# 训练循环
for step in range(start_step, max_steps + 1):
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

    # 打印信息并保存日志
    if step % log_interval == 0:
        print(
            f"[train ] step = {step}, loss = {loss.item():.4f}, ppl = {ppl.item():.2f}, lr = {lr:.2e}, tok/s = {(batch_size * context_length) / dt:.2f}, 运行时间 = {(time.time()-start_time):.2f} 秒"
        )
        train_record = {
            "type": "train",
            "step": step,
            "loss": loss.item(),
            "ppl": ppl.item(),
            "lr": lr,
            "tok_per_s": (batch_size * context_length) / dt,
            "time_s": time.time() - start_time,
        }
        append_metrics(metrics_path, train_record)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    gradient_clipping(model.parameters(), max_l2_norm)  # 梯度裁减
    optimizer.step()

    # 每 eval_interval 打印一次
    if step % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(eval_steps):
                x, y = data_loader(
                    valid_tokens, batch_size, context_length, device=device
                )
                logits = model(x)
                loss = cross_entropy(logits, y)
                val_losses.append(loss.item())
            mean_val_loss = sum(val_losses) / eval_steps
            ppl = np.exp(mean_val_loss)

            # 打印信息并保存日志
            print(
                f"\n\n[eval ] step = {step}, loss = {mean_val_loss:.4f}, ppl = {ppl:.2f}, lr = {lr:.2e}, 运行时间 = {(time.time()-start_time):.2f} 秒\n"
            )
            eval_record = {
                "type": "eval",
                "step": step,
                "loss": mean_val_loss,
                "ppl": ppl,
                "lr": lr,
                "time_s": time.time() - start_time,
            }
            append_metrics(metrics_path, eval_record)

            # 只保存最近三个
            checkpoint_path = checkpoints_dir / f"lm_step_{step}.pt"
            save_checkpoint(model, optimizer, step, out=checkpoint_path)
            recent_checkpoints.append(checkpoint_path)
            print(f"[save ] step = {step}, checkpoint = {checkpoint_path}\n")

            recent_checkpoints = sorted(
                recent_checkpoints,
                key=lambda p: int(p.stem.split("_")[-1]),
            )

            while len(recent_checkpoints) > 3:
                old_checkpoint = recent_checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()

            # 保存best
            is_best = mean_val_loss < best_val_loss
            if is_best:
                best_val_loss = mean_val_loss
                best_checkpoint_path = checkpoints_dir / "best.pt"

                try:
                    save_checkpoint(model, optimizer, step, out=best_checkpoint_path)

                    best_info = {
                        "step": step,
                        "loss": mean_val_loss,
                        "ppl": ppl,
                        "checkpoint": str(best_checkpoint_path),
                    }
                    with open(best_info_path, "w", encoding="utf-8") as f:
                        json.dump(best_info, f, ensure_ascii=False, indent=2)

                    print(f"\t best更新 = {best_checkpoint_path}\n")
                except Exception as e:
                    print(f"\tbest保存失败！！！ 错误原因: {e}\n")
    model.train()
