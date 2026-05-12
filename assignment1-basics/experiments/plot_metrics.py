from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(metrics_path: Path) -> dict[str, list[float]]:
    train_steps: list[int] = []
    train_times: list[float] = []
    train_losses: list[float] = []
    train_ppls: list[float] = []
    train_lrs: list[float] = []

    eval_steps: list[int] = []
    eval_times: list[float] = []
    eval_losses: list[float] = []
    eval_ppls: list[float] = []
    eval_lrs: list[float] = []

    with open(metrics_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)
            record_type = record.get("type")

            if record_type == "train":
                train_steps.append(record["step"])
                train_times.append(record["time_s"])
                train_losses.append(record["loss"])
                train_ppls.append(record["ppl"])
                train_lrs.append(record["lr"])
            elif record_type == "eval":
                eval_steps.append(record["step"])
                eval_times.append(record["time_s"])
                eval_losses.append(record["loss"])
                eval_ppls.append(record["ppl"])
                eval_lrs.append(record["lr"])

    return {
        "train_steps": train_steps,
        "train_times": train_times,
        "train_losses": train_losses,
        "train_ppls": train_ppls,
        "train_lrs": train_lrs,
        "eval_steps": eval_steps,
        "eval_times": eval_times,
        "eval_losses": eval_losses,
        "eval_ppls": eval_ppls,
        "eval_lrs": eval_lrs,
    }


def plot_metrics(run_dir: Path, data: dict[str, list[float]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(
        data["train_steps"], data["train_losses"], label="train", linewidth=1.5
    )
    axes[0, 0].plot(
        data["eval_steps"], data["eval_losses"], label="eval", linewidth=2.0
    )
    axes[0, 0].set_title("Loss vs Step")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(
        data["train_times"], data["train_losses"], label="train", linewidth=1.5
    )
    axes[0, 1].plot(
        data["eval_times"], data["eval_losses"], label="eval", linewidth=2.0
    )
    axes[0, 1].set_title("Loss vs Wall-Clock Time")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(
        data["train_steps"], data["train_ppls"], label="train", linewidth=1.5
    )
    axes[1, 0].plot(data["eval_steps"], data["eval_ppls"], label="eval", linewidth=2.0)
    axes[1, 0].set_title("Perplexity vs Step")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Perplexity")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(
        data["train_steps"], data["train_lrs"], label="train lr", linewidth=1.5
    )
    axes[1, 1].plot(
        data["eval_steps"], data["eval_lrs"], label="eval lr", linewidth=2.0
    )
    axes[1, 1].set_title("Learning Rate vs Step")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle(run_dir.name, fontsize=14)
    fig.tight_layout()

    output_path = run_dir / "metrics_overview.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"已保存图像: {output_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Plot training metrics from metrics.jsonl."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="run_001",
        help="Run directory name under assignment1-basics/runs/",
    )
    args = parser.parse_args()

    run_dir = project_root / "runs" / args.run_name
    metrics_path = run_dir / "metrics.jsonl"

    if not metrics_path.exists():
        raise FileNotFoundError(f"未找到日志文件: {metrics_path}")

    data = load_metrics(metrics_path)
    plot_metrics(run_dir, data)


if __name__ == "__main__":
    main()
