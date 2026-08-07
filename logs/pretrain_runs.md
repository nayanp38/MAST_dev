# Run Ledger — P2 Pretraining (MAST encoder)

Per CLAUDE.md documentation rules. One row per run; full metrics in W&B
project `mast-p2` (+ per-run `logs/pretrain/<run>/log.jsonl`).

**Goal:** pretrain the MAST encoder by masked reconstruction on the frozen
1.40M-record corpus (log: `p2_corpus_build.md`); select {lr, mask ratio,
granularity} via Stage-1 HPO (frozen proxy formula
`logs/optuna/stage1_proxy_formula.json`: bal-acc − 2.0·(NLL−1.6)/0.2);
end at Gate G1 (probe vs PCA features, ≥+3 macro-F1).

**Data:** `tokens_v1` cache; subsets per `corpus_holdout_k20_seed42`
(fold 0 = held-out 5%, folds 1–5 = HPO 25%). Probe: 476-object VIS+NIR
set on frozen b1 folds (disputed excluded from training folds).

| Run | Config | Data | Result | Notes |
|---|---|---|---|---|
| smoke0 | base, ρ=0.45 mixed, 2+2 ep | hpo25 | probe 27.4 F1 after 2 ep | pipeline + W&B + resume verified |
| pilot_rho30 | ρ=0.30 mixed, lr 1e-3, 8 ep | hpo25 | NLL 1.781, bal-acc 37.9 | pipeline verification; formula seeding |
| pilot_rho45 | ρ=0.45 mixed, lr 1e-3, 8 ep | hpo25 | NLL 1.586, bal-acc 37.1 | " |
| pilot_rho60 | ρ=0.60 mixed, lr 1e-3, 8 ep | hpo25 | NLL 1.600, bal-acc 37.8 | " |
| hpo1/trial000–034 | search {lr, ρ, granularity}, ≤80 ep, ASHA | hpo25 | **winner #30: lr 8.8e-4, ρ 0.45, mixed** — NLL 1.231, probe 38.2 bal-acc / 38.7 F1; top-5 all ρ=0.45+mixed | 35/35 complete, on Lambda A10, ~4.7 h |
| final, final_s1 | winner, ≤600 ep cap, patience 50 | full | **INVALID** — probe 26.9/26.6 | schedule bug: cosine horizon tied to the 600-ep cap while patience stopped at 69/103 ⇒ LR never annealed (still ~peak 8.8e-4 at stop; holdout NLL rose 1.23→1.42 after warmup). G1 preview with this checkpoint: 32.9 F1 vs PCA baseline 40.5 — meaningless FAIL. Fixes: finals use epochs=150 (cosine completes); final probe now evaluates best.pt; artifact upload made recursive (Lambda trial checkpoints were lost to a non-recursive glob — study DB survived). |
| final_v2, final_v2_s1 | winner, 150 ep, patience 50 | full | (pending) | corrected schedule; `scripts/lambda/run_finals_and_terminate.sh` |

Untrained-encoder probe baseline: 15.7 macro-F1 / 18.1 bal-acc.
**PCA+slope baseline (Gate G1 Arm B, measured): 40.5 macro-F1 / 41.2 bal-acc → G1 pass bar = 43.5 macro-F1.**
