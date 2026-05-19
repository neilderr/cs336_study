import json, torch, argparse, statistics

from pathlib import Path
from timeit import default_timer

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

# 加载配置文件
config_path = Path("experiments") / "config.json"


with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

parser = argparse.ArgumentParser(description="Benchmark script")

parser.add_argument(
    "--mode",
    type=str,
    default="full_step",
    choices=["forward", "forward_backward", "full_step"],
)
parser.add_argument("--warmup_steps", type=int, default=5)
parser.add_argument("--num_steps", type=int, default=10)

parser.add_argument("--vocab_size", type=int, default=config["vocab_size"])
parser.add_argument("--d_ff", type=int, default=config["d_ff"])
parser.add_argument("--rope_theta", type=float, default=config["rope_theta"])
parser.add_argument("--batch_size", type=int, default=config["batch_size"])
parser.add_argument("--context_length", type=int, default=config["context_length"])
parser.add_argument("--eps", type=float, default=config["eps"])
parser.add_argument("--d_model", type=int, default=config["d_model"])
parser.add_argument("--num_layers", type=int, default=config["num_layers"])
parser.add_argument("--num_heads", type=int, default=config["num_heads"])
parser.add_argument("--lr_max", type=float, default=config["lr_max"])
parser.add_argument("--weight_decay", type=float, default=config["weight_decay"])
parser.add_argument("--output", type=str, default="benchmark_results.csv")

args = parser.parse_args()
betas = tuple(config["betas"])

if torch.cuda.is_available():
    device = "cuda"
    sync_fn = torch.cuda.synchronize
elif torch.backends.mps.is_available():
    device = "mps"
    sync_fn = torch.mps.synchronize
else:
    device = "cpu"
    sync_fn = lambda: None


print(f"mode = {args.mode}")
print(f"warmup_steps = {args.warmup_steps}")
print(f"num_steps = {args.num_steps}")
print(f"batch_size = {args.batch_size}")
print(f"device = {device}")


# 生成随机batch
def make_batch():
    x = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        dtype=torch.long,
        device=device,
    )
    y = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        dtype=torch.long,
        device=device,
    )
    return x, y


# 按照mode运行一步
def run_step(x, y):
    # 前向传播
    logits = model(x)
    loss = cross_entropy(logits, y)

    if args.mode == "forward":
        return

    # 后向传播
    optimizer.zero_grad()
    loss.backward()

    if args.mode == "forward_backward":
        return

    # 参数更新
    optimizer.step()


if __name__ == "__main__":
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    model.train()

    optimizer = AdamW(
        model.parameters(), args.lr_max, betas, args.eps, args.weight_decay
    )

    # 预热
    print(f"\n预热阶段")
    print(f"waiting~\n")
    for _ in range(args.warmup_steps):
        x, y = make_batch()
        run_step(x, y)

    # 等待同步
    sync_fn()

    times_ms = []

    # 正式计时
    print("正式计时")
    for i in range(args.num_steps):
        x, y = make_batch()

        # 计时前同步
        sync_fn()
        start_time = default_timer()

        run_step(x, y)

        # 计时前同步
        sync_fn()
        elapsed_time = (default_timer() - start_time) * 1000
        times_ms.append(elapsed_time)

        print(f"第 {i+1} 轮: {elapsed_time:.3f} ms")

    # 统计数据
    avg_ms = statistics.mean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    print("\n=== Summary ===")
    print(f"mode: {args.mode}")
    print(f"device: {device}")
    print(f"avg_total_ms: {avg_ms:.3f} ms")
    print(f"std_total_ms: {std_ms:.3f} ms")
    print(f"min_total_ms: {min(times_ms):.3f} ms")
    print(f"max_total_ms: {max(times_ms):.3f} ms")
