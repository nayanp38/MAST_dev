#!/usr/bin/env bash
# Full overnight P2 sequence on a Lambda instance, self-terminating.
#
#   Stage-1 HPO (35 trials) -> final pretrain (seed 0) -> replicate
#   (seed 1) -> upload artifacts -> TERMINATE THE INSTANCE.
#
# On ANY exit (success, crash, or the wall-clock cap) the trap uploads
# whatever results exist to W&B and terminates the instance, so it
# cannot bill indefinitely. Status lines go to ~/mast/status.txt for
# the local monitor.
#
# Usage (on the instance):  cd ~/mast && nohup bash scripts/lambda/run_all_and_terminate.sh > runall.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."

MAX_SECONDS=$((16 * 3600))   # hard runaway cap
START=$(date +%s)
status() { echo "$(date -u +%H:%M:%S) $*" | tee -a status.txt; }

cleanup() {
  code=$?
  status "EXIT trap fired (code=$code) — uploading artifacts"
  python3 scripts/lambda/upload_artifacts.py "exit-code-$code" || true
  status "terminating instance"
  python3 scripts/lambda/self_terminate.py || status "TERMINATE FAILED — instance still running!"
}
trap cleanup EXIT

# watchdog: kill the whole process group at the cap
( sleep "$MAX_SECONDS" && status "WALL-CLOCK CAP HIT" && kill -TERM $$ ) &
WATCHDOG=$!

status "stage: hpo (35 trials)"
PYTHONPATH=src python3 scripts/hpo_stage1.py --trials 35 2>&1 | tee hpo.log
status "stage: hpo done"
python3 scripts/lambda/upload_artifacts.py "post-hpo" || true

# extract winner params from the study
read -r LR RHO GRAN <<< "$(python3 - <<'PY'
import optuna
s = optuna.load_study(study_name="stage1", storage="sqlite:///logs/optuna/stage1.db")
p = s.best_trial.params
print(p["lr"], p["mask_ratio"], p["granularity"])
PY
)"
status "winner: lr=$LR rho=$RHO gran=$GRAN"

status "stage: final pretrain (seed 0)"
PYTHONPATH=src python3 scripts/pretrain.py --run-name final --subset full \
  --epochs 600 --lr "$LR" --mask-ratio "$RHO" --granularity "$GRAN" \
  --seed 0 2>&1 | tee final.log
python3 scripts/lambda/upload_artifacts.py "post-final" || true

status "stage: replicate (seed 1)"
PYTHONPATH=src python3 scripts/pretrain.py --run-name final_s1 --subset full \
  --epochs 600 --lr "$LR" --mask-ratio "$RHO" --granularity "$GRAN" \
  --seed 1 2>&1 | tee final_s1.log

status "stage: all done"
kill "$WATCHDOG" 2>/dev/null || true
exit 0
