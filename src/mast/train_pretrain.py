"""Pretraining loop for MAST (P2 plan W3).

Config-driven; device-portable (CUDA bf16 autocast / MPS+CPU fp32);
atomic checkpoints with full RNG state; held-out-NLL eval each epoch;
linear-probe hook every `probe_every` epochs; W&B + JSONL logging.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from mast import masking
from mast.model import MAST, MASTConfig, contrastive_loss, reconstruction_loss
from mast.pretrain_data import CorpusTensors

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    run_name: str = "dev"
    subset: str = "hpo25"            # 'full' | 'hpo25'
    device: str = "auto"
    seed: int = 0
    # optimization
    lr: float = 1e-3
    weight_decay: float = 0.03
    batch_size: int = 1024
    epochs: int = 80
    warmup_frac: float = 0.06
    min_lr: float = 1e-6
    early_stop_patience: int = 50
    # objective
    mask_ratio: float = 0.45
    granularity: str = "mixed"       # 'token' | 'block' | 'mixed'
    contrastive_weight: float = 0.1
    # eval
    probe_every: int = 10
    # infra
    checkpoint_every: int = 25
    out_dir: str = "logs/pretrain"
    wandb_project: str = "mast-p2"
    use_wandb: bool = True
    model: MASTConfig = field(default_factory=MASTConfig)


def get_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _amp(device: str):
    # bf16 autocast on CUDA cards that support it (Ampere+); fp32
    # everywhere else — T4/V100 (Colab free tier) and MPS run fp32,
    # which is fine for this launch-bound 2.7M-param model.
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = get_device(cfg.device)
        cfg.model.contrastive_weight = cfg.contrastive_weight
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.train_data = CorpusTensors(device=self.device, subset=cfg.subset)
        self.holdout = CorpusTensors(device=self.device, subset="holdout")
        self.model = MAST(cfg.model).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.steps_per_epoch = max(self.train_data.n_total // cfg.batch_size, 1)
        self.total_steps = self.steps_per_epoch * cfg.epochs
        self.rng = np.random.default_rng(cfg.seed)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.epoch = 0
        self.global_step = 0
        self.best_nll = math.inf
        self.best_epoch = -1

        self.out_dir = ROOT / cfg.out_dir / cfg.run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.out_dir / "log.jsonl", "a")
        self.wandb = None
        if cfg.use_wandb:
            _load_env()
            try:
                import wandb
                self.wandb = wandb.init(
                    project=cfg.wandb_project, name=cfg.run_name,
                    config=dataclasses.asdict(cfg), resume="allow",
                    id=cfg.run_name.replace("/", "-"),
                )
            except Exception as exc:  # offline dev shouldn't crash training
                print(f"[wandb disabled: {exc}]")

    # -- schedule ---------------------------------------------------------
    def _lr_at(self, step: int) -> float:
        cfg = self.cfg
        warmup = max(int(self.total_steps * cfg.warmup_frac), 1)
        if step < warmup:
            return cfg.lr * step / warmup
        t = (step - warmup) / max(self.total_steps - warmup, 1)
        return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * t))

    # -- steps ------------------------------------------------------------
    def _step(self) -> dict:
        cfg = self.cfg
        lr = self._lr_at(self.global_step)
        for group in self.opt.param_groups:
            group["lr"] = lr
        with_pairs = cfg.contrastive_weight > 0
        batch = self.train_data.sample_batch(cfg.batch_size, self.rng,
                                             with_pairs=with_pairs)
        mask = masking.make_mask(cfg.granularity, batch["valid"],
                                 batch["token_type"], cfg.mask_ratio, self.gen)
        mse_only = self.epoch < cfg.model.mse_warmup_epochs
        with _amp(self.device):
            out = self.model(batch, mask)
            losses = reconstruction_loss(out, batch, mask, cfg.model, mse_only)
            loss = losses["loss"]
            if with_pairs:
                pair_out = self.model(batch["pair"], mask=None)
                closs = contrastive_loss(out["pooled"], pair_out["pooled"],
                                         cfg.model.contrastive_temp)
                loss = loss + cfg.contrastive_weight * closs
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.global_step += 1
        return {"loss": float(loss.detach().cpu()),
                "nll": float(losses["nll"].cpu()), "lr": lr}

    @torch.no_grad()
    def eval_holdout(self) -> float:
        """Masked NLL on the frozen 5% held-out (fixed eval seed)."""
        self.model.eval()
        gen = torch.Generator(device=self.device).manual_seed(1234)
        total, count = 0.0, 0
        for source in self.holdout.sources:
            for batch in self.holdout.iter_all(source, 4096):
                mask = masking.make_mask(self.cfg.granularity, batch["valid"],
                                         batch["token_type"],
                                         self.cfg.mask_ratio, gen)
                out = self.model(batch, mask)
                losses = reconstruction_loss(out, batch, mask, self.cfg.model)
                n = int((mask & batch["valid"]).sum())
                total += float(losses["nll"].cpu()) * n
                count += n
        self.model.train()
        return total / max(count, 1)

    def run_probe(self) -> dict:
        from mast.probe import linear_probe
        return linear_probe(self.model, device=self.device)

    # -- checkpointing ----------------------------------------------------
    def save(self, name: str):
        path = self.out_dir / f"{name}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save({
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_nll": self.best_nll,
            "best_epoch": self.best_epoch,
            "cfg": dataclasses.asdict(self.cfg),
            "torch_rng": torch.get_rng_state(),
            "np_rng": self.rng.bit_generator.state,
        }, tmp)
        tmp.rename(path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_nll = state["best_nll"]
        self.best_epoch = state["best_epoch"]
        torch.set_rng_state(state["torch_rng"].cpu().to(torch.uint8))
        self.rng.bit_generator.state = state["np_rng"]

    def _log(self, record: dict):
        record = {"epoch": self.epoch, "step": self.global_step,
                  "time": time.time(), **record}
        self.log_file.write(json.dumps(record) + "\n")
        self.log_file.flush()
        if self.wandb is not None:
            self.wandb.log(record, step=self.global_step)

    # -- main loop --------------------------------------------------------
    def train(self, epoch_callback=None) -> dict:
        cfg = self.cfg
        while self.epoch < cfg.epochs:
            t0 = time.time()
            metrics = []
            for _ in range(self.steps_per_epoch):
                metrics.append(self._step())
            self.epoch += 1
            train_nll = float(np.mean([m["nll"] for m in metrics]))
            holdout_nll = self.eval_holdout()
            record = {"train_nll": train_nll, "holdout_nll": holdout_nll,
                      "lr": metrics[-1]["lr"],
                      "epoch_seconds": time.time() - t0}
            if holdout_nll < self.best_nll:
                self.best_nll = holdout_nll
                self.best_epoch = self.epoch
                self.save("best")
            if cfg.probe_every and self.epoch % cfg.probe_every == 0:
                probe = self.run_probe()
                record.update({f"probe_{k}": v for k, v in probe.items()})
            if self.epoch % cfg.checkpoint_every == 0:
                self.save("last")
            self._log(record)
            if epoch_callback is not None:
                stop = epoch_callback(self.epoch, record)
                if stop:
                    break
            if self.epoch - self.best_epoch > cfg.early_stop_patience:
                break
        self.save("last")
        final = {"best_nll": self.best_nll, "best_epoch": self.best_epoch,
                 "epochs_run": self.epoch}
        # final probe evaluates the BEST checkpoint, not the last epoch
        best_path = self.out_dir / "best.pt"
        if best_path.exists():
            state = torch.load(best_path, map_location=self.device,
                               weights_only=False)
            self.model.load_state_dict(state["model"])
        probe = self.run_probe()
        final.update({f"probe_{k}": v for k, v in probe.items()})
        self._log({"final": True, **final})
        if self.wandb is not None:
            self.wandb.summary.update(final)
            self.wandb.finish()
        return final
