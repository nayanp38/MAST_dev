#!/usr/bin/env bash
# Sync study DB, run logs, and checkpoints back from the Colab VM.
# Safe to run anytime; run before the VM expires.
# Usage: bash scripts/colab/pull_results.sh <ssh-host> [remote-dir]
set -euo pipefail
HOST="${1:?usage: pull_results.sh <ssh-host> [remote-dir]}"
REMOTE_DIR="${2:-mast}"
cd "$(dirname "$0")/../.."

rsync -az "$HOST:~/$REMOTE_DIR/logs/optuna/" logs/optuna/
rsync -az "$HOST:~/$REMOTE_DIR/logs/pretrain/" logs/pretrain/
rsync -az "$HOST:~/$REMOTE_DIR/logs/nohup/" logs/nohup/
echo "pulled. study state:"
.venv/bin/python - <<'PY'
import optuna
from collections import Counter
s = optuna.load_study(study_name="stage1", storage="sqlite:///logs/optuna/stage1.db")
print(Counter(t.state.name for t in s.trials))
done = [t for t in s.trials if t.state.name == "COMPLETE"]
if done:
    best = max(done, key=lambda t: t.value)
    print(f"best so far: #{best.number} score={best.value:.2f} {best.params}")
PY
