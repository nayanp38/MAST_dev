#!/usr/bin/env bash
# Push code + minimal P2 artifacts + study state to the Colab VM.
# Usage: bash scripts/colab/sync_to_remote.sh <ssh-host> [remote-dir]
set -euo pipefail
HOST="${1:?usage: sync_to_remote.sh <ssh-host> [remote-dir]}"
REMOTE_DIR="${2:-mast}"
cd "$(dirname "$0")/../.."

ssh "$HOST" "mkdir -p ~/$REMOTE_DIR/{logs/nohup,logs/optuna,logs/pretrain,data/processed/corpus,data/processed/splits}"

# code
rsync -az --delete src scripts tests "$HOST:~/$REMOTE_DIR/"

# data (compiled artifacts only — no raw catalogs)
rsync -az data/processed/corpus/tokens_v1 "$HOST:~/$REMOTE_DIR/data/processed/corpus/"
rsync -az data/processed/corpus/scalars.parquet "$HOST:~/$REMOTE_DIR/data/processed/corpus/"
rsync -az data/processed/splits/ "$HOST:~/$REMOTE_DIR/data/processed/splits/"
rsync -az data/processed/label_table_v1.parquet \
      data/processed/visnir_dataset.parquet \
      data/processed/mithneos_demeo_per_spectrum.csv \
      "$HOST:~/$REMOTE_DIR/data/processed/"

# study state (resumable) + frozen proxy formula
rsync -az logs/optuna/ "$HOST:~/$REMOTE_DIR/logs/optuna/"

# W&B key
scp -q .env "$HOST:~/$REMOTE_DIR/.env"

echo "synced to $HOST:~/$REMOTE_DIR"
