"""Single pretraining run from CLI flags (P2 W3/W4).

Usage examples:
  PYTHONPATH=src .venv/bin/python scripts/pretrain.py --run-name final \
      --subset full --epochs 600 --lr 8e-4 --mask-ratio 0.45 \
      --granularity block --seed 0
  ... --resume logs/pretrain/final/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mast.train_pretrain import Trainer, TrainConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--subset", default="full", choices=["full", "hpo25"])
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask-ratio", type=float, default=0.45)
    ap.add_argument("--granularity", default="mixed",
                    choices=["token", "block", "mixed"])
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = TrainConfig(
        run_name=args.run_name, subset=args.subset, epochs=args.epochs,
        lr=args.lr, mask_ratio=args.mask_ratio, granularity=args.granularity,
        batch_size=args.batch_size, seed=args.seed,
        probe_every=args.probe_every, early_stop_patience=args.patience,
        use_wandb=not args.no_wandb,
    )
    trainer = Trainer(cfg)
    if args.resume:
        trainer.load(args.resume)
        print(f"resumed from {args.resume} at epoch {trainer.epoch}")
    final = trainer.train()
    print(final)


if __name__ == "__main__":
    main()
