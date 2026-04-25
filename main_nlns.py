from __future__ import annotations

import argparse
from pathlib import Path

from nlns.train_nlns import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep RL + ALNS runner for the original TXT dataset.")
    parser.add_argument("--instances_dir", required=True, help="e.g. data/train")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps_per_episode", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy_beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_remove", type=int, default=5)
    parser.add_argument("--max_remove", type=int, default=20)
    parser.add_argument("--save_dir", default="outputs/nlns", help="Directory for checkpoints and logs")
    parser.add_argument("--checkpoint_every", type=int, default=1)
    args = parser.parse_args()

    instances_path = Path(args.instances_dir)
    if not instances_path.exists():
        raise FileNotFoundError(f"Instances directory not found: {instances_path}")

    print("=== NLNS TRAINING STARTED ===", flush=True)
    print(f"Instances directory: {instances_path}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Steps per episode: {args.steps_per_episode}", flush=True)
    print("=============================", flush=True)

    train(
        instances_dir=args.instances_dir,
        epochs=args.epochs,
        steps_per_episode=args.steps_per_episode,
        lr=args.lr,
        seed=args.seed,
        save_dir=args.save_dir,
        gamma=args.gamma,
        entropy_beta=args.entropy_beta,
        min_remove=args.min_remove,
        max_remove=args.max_remove,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()