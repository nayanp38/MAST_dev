"""Stage-1 pretraining HPO (plan §3.2.1 compute-lean protocol).

Optuna + SuccessiveHalvingPruner (ASHA behavior, single worker),
30-40 trials on the frozen 25% subsample with an 80-epoch cap.
Search space (everything else frozen at base config):
    lr           log-uniform [1e-4, 3e-3]
    mask_ratio   {0.3, 0.45, 0.6, 0.75}
    granularity  {token, block, mixed}

Trial objective (FROZEN before trial 1; recorded in study user attrs):
    objective = probe_balanced_acc - ALPHA * (holdout_nll - NLL_REF) / NLL_SCALE
with ALPHA/NLL_REF/NLL_SCALE fixed from the three pilot runs
(scripts/pretrain.py --run-name pilot_*). Pruning signal: holdout NLL
per epoch.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/hpo_stage1.py [--trials 35] [--pilot]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import optuna

from mast.train_pretrain import Trainer, TrainConfig

STUDY_DIR = ROOT / "logs/optuna"
EPOCH_CAP = 80

# Frozen proxy-formula constants (set from pilots by --set-formula, then
# never changed; the study refuses to start without them).
FORMULA_PATH = STUDY_DIR / "stage1_proxy_formula.json"


def run_pilot(mask_ratio: float, run_name: str) -> dict:
    """Short pipeline-verification runs; their (NLL, probe) numbers also
    seed the frozen proxy-formula constants (rough calibration is all
    the formula needs — it only ranks trials)."""
    cfg = TrainConfig(run_name=run_name, subset="hpo25", epochs=8,
                      probe_every=8, mask_ratio=mask_ratio,
                      granularity="mixed", use_wandb=True)
    return Trainer(cfg).train()


def objective_factory(formula: dict):
    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        rho = trial.suggest_categorical("mask_ratio", [0.3, 0.45, 0.6, 0.75])
        gran = trial.suggest_categorical("granularity", ["token", "block", "mixed"])
        cfg = TrainConfig(
            run_name=f"hpo1/trial{trial.number:03d}", subset="hpo25",
            epochs=EPOCH_CAP, probe_every=0, lr=lr, mask_ratio=rho,
            granularity=gran, use_wandb=True, early_stop_patience=EPOCH_CAP,
        )
        trainer = Trainer(cfg)

        def cb(epoch: int, record: dict) -> bool:
            trial.report(record["holdout_nll"], step=epoch)
            return trial.should_prune()

        final = trainer.train(epoch_callback=cb)
        pruned = final["epochs_run"] < EPOCH_CAP and \
            final["epochs_run"] - trainer.best_epoch <= cfg.early_stop_patience
        score = (final["probe_balanced_acc"]
                 - formula["alpha"]
                 * (final["best_nll"] - formula["nll_ref"]) / formula["nll_scale"])
        trial.set_user_attr("probe_balanced_acc", final["probe_balanced_acc"])
        trial.set_user_attr("probe_macro_f1", final["probe_macro_f1"])
        trial.set_user_attr("best_nll", final["best_nll"])
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    return objective


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=35)
    ap.add_argument("--pilot", action="store_true",
                    help="run the three pilot configurations and exit")
    ap.add_argument("--set-formula", nargs=3, type=float, metavar=("ALPHA", "NLL_REF", "NLL_SCALE"),
                    help="freeze the proxy formula (once)")
    args = ap.parse_args()
    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    if args.pilot:
        for rho in (0.3, 0.45, 0.6):
            res = run_pilot(rho, f"pilot_rho{int(rho*100)}")
            print(f"pilot rho={rho}: {res}")
        return

    if args.set_formula:
        if FORMULA_PATH.exists():
            raise SystemExit("proxy formula already frozen — refusing to change")
        alpha, ref, scale = args.set_formula
        FORMULA_PATH.write_text(json.dumps(
            {"alpha": alpha, "nll_ref": ref, "nll_scale": scale,
             "definition": "probe_balanced_acc - alpha*(best_nll-nll_ref)/nll_scale"},
            indent=2) + "\n")
        print(f"frozen: {FORMULA_PATH.read_text()}")
        return

    if not FORMULA_PATH.exists():
        raise SystemExit("run pilots and freeze the proxy formula first "
                         "(--pilot, then --set-formula A R S)")
    formula = json.loads(FORMULA_PATH.read_text())

    study = optuna.create_study(
        study_name="stage1", direction="maximize",
        storage=f"sqlite:///{STUDY_DIR}/stage1.db", load_if_exists=True,
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=20, reduction_factor=3),
    )
    for k, v in formula.items():
        study.set_user_attr(k, v)
    done = len([t for t in study.trials
                if t.state in (optuna.trial.TrialState.COMPLETE,
                               optuna.trial.TrialState.PRUNED)])
    study.optimize(objective_factory(formula), n_trials=max(args.trials - done, 0))
    best = study.best_trial
    print(f"best trial #{best.number}: score={best.value:.3f} params={best.params} "
          f"attrs={best.user_attrs}")


if __name__ == "__main__":
    main()
