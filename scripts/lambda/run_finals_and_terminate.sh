#!/usr/bin/env bash
# Corrected final pretraining runs only (schedule fix: cosine horizon =
# 150 epochs so annealing completes; see logs/pretrain_runs.md bug
# entry). Reads the winner from the completed Stage-1 study, runs
# seed 0 + seed 1, uploads artifacts, terminates the instance.
#
# Usage (on the instance): cd ~/mast && nohup bash scripts/lambda/run_finals_and_terminate.sh > runfinals.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."

MAX_SECONDS=$((10 * 3600))
status() { echo "$(date -u +%H:%M:%S) $*" | tee -a status.txt; }

cleanup() {
  code=$?
  status "EXIT trap (code=$code) — uploading + terminating"
  python3 scripts/lambda/upload_artifacts.py "finals-exit-$code" || true
  python3 scripts/lambda/self_terminate.py || status "TERMINATE FAILED"
}
trap cleanup EXIT
( sleep "$MAX_SECONDS" && status "WALL CAP" && kill -TERM $$ ) &
WATCHDOG=$!

read -r LR RHO GRAN <<< "$(python3 - <<'PY'
import optuna
s = optuna.load_study(study_name="stage1", storage="sqlite:///logs/optuna/stage1.db")
p = s.best_trial.params
print(p["lr"], p["mask_ratio"], p["granularity"])
PY
)"
status "winner: lr=$LR rho=$RHO gran=$GRAN — finals at 150 epochs"

for SEED in 0 1; do
  NAME=final_v2$([ "$SEED" = "1" ] && echo "_s1")
  status "stage: $NAME (seed $SEED)"
  PYTHONPATH=src python3 scripts/pretrain.py --run-name "$NAME" --subset full \
    --epochs 150 --lr "$LR" --mask-ratio "$RHO" --granularity "$GRAN" \
    --seed "$SEED" 2>&1 | tee "$NAME.log"
  python3 scripts/lambda/upload_artifacts.py "post-$NAME" || true
done

status "stage: finals done"
kill "$WATCHDOG" 2>/dev/null || true
exit 0
