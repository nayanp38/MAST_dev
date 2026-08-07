#!/usr/bin/env bash
# Launch (or resume) the Stage-1 HPO study detached on the Colab VM.
# Usage: bash scripts/colab/run_hpo_remote.sh <ssh-host> [n-trials] [remote-dir]
set -euo pipefail
HOST="${1:?usage: run_hpo_remote.sh <ssh-host> [n-trials] [remote-dir]}"
TRIALS="${2:-35}"
REMOTE_DIR="${3:-mast}"

ssh "$HOST" bash -s <<REMOTE
set -e
cd ~/$REMOTE_DIR
if pgrep -f 'hpo_stage1.py --trials' >/dev/null; then
  echo "study already running:"; pgrep -fl 'hpo_stage1.py --trials'; exit 0
fi
mkdir -p logs/nohup
nohup env PYTHONPATH=src python scripts/hpo_stage1.py --trials $TRIALS \
  > logs/nohup/hpo_stage1.log 2>&1 &
echo "launched pid \$!"
sleep 20
tail -n 3 logs/nohup/hpo_stage1.log
REMOTE
