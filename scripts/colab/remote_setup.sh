#!/usr/bin/env bash
# Install deps + verify CUDA on the Colab VM (idempotent, ~1 min).
# Usage: bash scripts/colab/remote_setup.sh <ssh-host> [remote-dir]
set -euo pipefail
HOST="${1:?usage: remote_setup.sh <ssh-host> [remote-dir]}"
REMOTE_DIR="${2:-mast}"

ssh "$HOST" bash -s <<REMOTE
set -e
cd ~/$REMOTE_DIR
pip install -q optuna wandb pyarrow 2>&1 | tail -1 || true
python - <<'PY'
import torch, sklearn, pandas, optuna, wandb, numpy
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| bf16:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
PYTHONPATH=src python - <<'PY'
# artifact sanity: token cache + manifests + probe set load
from mast.pretrain_data import CorpusTensors
ct = CorpusTensors(device="cpu", subset="holdout")
print("holdout records:", ct.n_total)
from mast import probe
d = probe._tokenize_probe_set()
print("probe set:", len(d["labels"]), "objects")
PY
REMOTE
echo "remote setup OK"
