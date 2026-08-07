"""Upload P2 result artifacts to W&B (survives instance termination).

Uploads whatever exists: optuna study DB, per-run best checkpoints +
JSONL logs, nohup logs. Idempotent (new artifact version each call).

Usage: python3 scripts/lambda/upload_artifacts.py <stage-label>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_env():
    if Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"
    _load_env()
    import wandb

    run = wandb.init(project="mast-p2", name=f"artifacts-{stage}",
                     job_type="artifact-upload")
    art = wandb.Artifact("p2-results", type="results",
                         metadata={"stage": stage})
    added = []
    for pattern in ["logs/optuna/stage1.db",
                    "logs/optuna/stage1_proxy_formula.json"]:
        if Path(pattern).exists():
            art.add_file(pattern, name=pattern)
            added.append(pattern)
    for fname in ["best.pt", "log.jsonl"]:
        for f in sorted(Path("logs/pretrain").rglob(fname)):
            art.add_file(str(f), name=str(f))
            added.append(str(f))
    for log in Path(".").glob("*.log"):
        art.add_file(str(log), name=f"nohup/{log.name}")
        added.append(str(log))
    run.log_artifact(art)
    run.finish()
    print(f"uploaded {len(added)} files as p2-results@{stage}", flush=True)


if __name__ == "__main__":
    main()
