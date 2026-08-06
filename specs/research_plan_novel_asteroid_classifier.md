# Research Plan: A Self-Supervised, Missing-Data-Native, Calibrated Architecture for Asteroid Taxonomic Classification

**Prepared:** August 6, 2026
**Companion document:** `ml_asteroid_classification_field_review.md` (cited throughout as "Review §N")
**Goal:** A peer-review-defensible paper demonstrating a scientifically significant advance over the 2024–2026 state of the art (Ge 2024; Tang 2025; Delbo 2026; Tinaut-Ruano 2026), built on capabilities the literature verifiably lacks (Review §4.7, §7).

---

## 0. Executive Summary

We will build **MAST** (Masked Asteroid Spectral Transformer): a modality-agnostic encoder pretrained by masked reconstruction on **~1.2 million unlabeled asteroid observations** (Gaia DR3 + SDSS + SkyMapper + MOVIS), fine-tuned hierarchically on the ~4,500-object labeled pool in **both** Bus-DeMeo and Mahlke taxonomies, and wrapped in **conformal calibration** with an explicit **cross-survey transfer benchmark** as the primary evaluation. Two alternative routes (a deep generative taxonomy model, and a physics-informed domain-adaptation pipeline) are specified to the same depth, with quantitative decision gates for pivoting. The plan includes exact hyperparameter search spaces, a 14-item ablation matrix designed to attribute every claimed performance gain to a specific architectural choice, and a pre-registered statistical protocol.

The novelty claim rests on four verified absences in the literature (Review §4.7): no self-supervised pretraining on unlabeled survey spectra, no conformal/calibrated uncertainty, no deep missing-data-native multimodal fusion, and no cross-survey transfer evaluation standard. Each is individually precedented in adjacent astronomy (stellar/galaxy spectra foundation models; TNO calibrated inference, Lin et al. 2026) — which makes the plan *feasible* — and collectively absent in asteroid taxonomy — which makes it *novel*.

---

## 1. Scientific Hypotheses (falsifiable, pre-registered)

Every claim in the eventual paper maps to one of these. Each has a quantitative success criterion and a designated ablation that isolates it.

- **H1 (pretraining).** Masked-reconstruction pretraining on unlabeled survey spectra improves type-level macro-F1 by ≥3 points and rare-class (n<25) recall by ≥10 points over an identical architecture trained from scratch, at every labeled-data fraction ≤100%. *Tested by ablation A1–A2.*
- **H2 (fusion).** Attention-masked multimodal fusion (spectra + albedo + phase slope, missing values handled natively) outperforms (a) spectra-only by ≥5 points balanced accuracy — the Ge et al. (2024) multimodal dividend was +7.8 on 6 collapsed classes; we predict ≥5 on the harder type-level task — and (b) the same inputs with mean/k-NN imputation by ≥2 points. *Tested by A3–A4.*
- **H3 (transfer).** The pretrained encoder cuts the in-domain→cross-survey performance drop (Review §6.5: ~30 points, Sullivan 2023) by at least half relative to supervised baselines evaluated under the identical protocol. *Tested by benchmark B2.*
- **H4 (calibration).** Mondrian (class-conditional) conformal prediction achieves 90%±2 empirical coverage on both in-domain and shifted test sets, while raw softmax confidence is demonstrably miscalibrated under shift (ECE > 0.10). *Tested by benchmark B4, ablation A11.*
- **H5 (degeneracy, secondary).** The fused model's X-complex resolution (E/M/P) matches Mahlke's albedo-GMM assignments at ≥85% agreement when albedo is present, and — critically — abstains (wide conformal sets) rather than guesses when albedo is missing. *Tested by targeted evaluation E5.*

If H1 fails at the decision gate (§8), the paper pivots to Route B or C, both of which remain publishable on different grounds (see §4).

---

## 2. Data Engineering Plan

Data work is half the paper's defensibility. Referees in this field (Review §6) attack data handling before architecture.

### 2.1 Corpora

**Unlabeled pretraining corpus (~1.2M observation-records):**

| Source | Records | Content | Handling |
|---|---|---|---|
| Gaia DR3 SSO reflectance (Galluccio 2023) | 60,518 | 16-band, 0.374–1.034 µm | Drop bands 1–2 and 15–16 (documented systematics); carry per-band flags and S/N as inputs, not filters — the model should *learn* quality weighting. Keep the S/N>20 subset (36,566) tagged for high-quality experiments. |
| SDSS colors (Sergeyev & Carry 2021 recomputation) | ~1,036,000 obs / 379,714 objects | ugriz | Convert to reflectance colors; propagate photometric errors; multiple epochs per object retained as separate records (natural data augmentation + rotational-variability signal). |
| SkyMapper (Sergeyev 2022) | ~880,000 obs / 205,515 objects | uvgriz | Same treatment; overlaps with SDSS objects give free cross-instrument pairs (used in contrastive variant, A1c). |
| MOVIS (Popescu 2018) | 53,447 | Y, J, Ks NIR colors | NIR anchor for the wavelength-coordinate encoding. |
| Auxiliary scalars via SsODNet/rocks | — | pV albedo (NEOWISE/AKARI/IRAS), phase-curve G or G1/G2, H | Attached to any record when available (~40–60% coverage); always with uncertainties. |

**Labeled fine-tuning pool (object-level deduplicated):**

- Mahlke et al. (2022) compilation: 6,038 observations / 4,526 asteroids, 0.45–2.45 µm + albedo — primary labeled set (both Mahlke classes and, where derivable, Bus-DeMeo labels via `classy`).
- DeMeo et al. (2009) 371 + MITHNEOS (~1,000 NEOs) — canonical Bus-DeMeo labels.
- Gaia DR3 labeled overlap: the ~2,653 S/N≥50 objects with MP3C literature classes (Delbo 2026's reference set) — reserved primarily for the **transfer test set**, never for training in the transfer benchmark.
- S3OS2 (820, visible) — additional visible-only labeled data; tests partial-input classification.

### 2.2 Label harmonization

Maintain a three-way label table per object: (Bus-DeMeo class, Bus-DeMeo complex, Mahlke class + probability). Rules, documented in an appendix: subclass→complex mapping fixed a priori (following DeMeo 2009 Table); conflicting multi-source labels resolved by (1) VIS+NIR over VIS-only provenance, (2) most recent, (3) else flagged "disputed" and excluded from training but *kept in the test set* as a labeled-noise stress subset. Expected disputed rate ~5–10% given the ~30% cross-catalog disagreement (Sullivan 2023) concentrated in photometric labels.

### 2.3 Preprocessing and nuisance-parameter policy

- **Common representation:** each observation = a set of (λ_eff, Δλ, reflectance, uncertainty) quadruplets + scalar tokens; bandpass widths taken from published filter curves (SVO Filter Profile Service) and, for Gaia, per-band effective widths of the BP/RP dispersion. No resampling to a fixed grid — the wavelength–bandwidth positional encoding (§3.1) makes grids unnecessary. This single design choice is what lets one model ingest Gaia 16-band, SDSS 5-point, MOVIS 3-point, and 200-point ground spectra as *physically commensurable* measurements rather than same-named features.
- **Normalization:** reflectance normalized at 0.55 µm where covered, else at the flux-weighted band nearest 0.55 µm with an offset token recording the pivot.
- **Slope policy (Review §6.2 — slope is informative but untrustworthy):** dual-stream. Stream 1: slope-removed shape (spline detrend, DeMeo-style). Stream 2: the removed slope as a *scalar token with inflated uncertainty* (floor of 0.5%/µm systematic per Marsset 2020) plus phase angle as a covariate token when known. Ablation A9 quantifies the cost/benefit.
- **Phase reddening:** where phase angle is recorded, add it as a token; also used in physics-based augmentation (§4.3).
- **Orbital elements: excluded from the primary model.** Ge et al. (2024) used orbit type; a referee can argue the model then learns *where* an asteroid is, not *what it is made of* (shortcut learning via dynamical families). We include orbits only in leakage ablation A10, evaluated with family-held-out splits, to make this critique quantitative rather than rhetorical — a defensive asset.

### 2.4 Splits (the most attackable part of any prior paper — over-engineer this)

- **Object-level deduplication everywhere:** all observations of one asteroid live on one side of every split (Mahlke: 549/2,125 objects have multiple spectra; random observation-level splits leak).
- **B1 in-domain:** 10-fold object-stratified CV on the labeled pool; inner 20% of each training fold for tuning/early stopping. Test folds untouched by any tuning (contrast with Penttilä 2021's acknowledged full-data tuning).
- **B2 cross-survey transfer (primary benchmark):** train on ground-based VIS+NIR labeled pool only → test on Gaia DR3 labeled overlap (never trained on); and the reverse; and ground-spectra→SDSS-colors of the same objects. Reported as (in-domain score, transfer score, gap).
- **B3 rare-class few-shot:** for classes with 10–50 members, episodes with K∈{5,10,20} training shots, rest tested; 20 episodes per configuration.
- **B4 calibration/OOD:** conformal calibration split carved from B1 training side; OOD suite = cometary spectra, spectra of icy TNO/Centaur interlopers, lab meteorite spectra outside asteroid distribution, and corrupted-Gaia simulations.
- **Family-blocked variant:** repeat B1 with entire dynamical families held out (e.g., train without Vesta family, test V-types) — the strongest available proxy for "does it generalize compositionally or memorize populations."

---

## 3. Route A (Primary): MAST — Masked Asteroid Spectral Transformer

### 3.1 Architecture

**Tokenization (the core novel component for this field):**
- *Spectral tokens:* each measurement is a quadruplet (λ_eff, Δλ, R, σ) → token = MLP(R, σ) + FourierFeatures(log λ_eff, **log Δλ**). The positional coordinate is two-dimensional: effective wavelength *and* bandpass width (filter FWHM, published for every survey; for Gaia, the per-band effective width of the BP/RP dispersion, which varies across the 16 bands). Rationale: a photometric value is the integral of the spectrum against a transmission curve, not a point sample — a broad SDSS r token reporting "no 0.7 µm dip" and a narrow band reporting the same are different evidence, and bandwidth varies *within* instruments (Gaia band widths; SPHEREx R≈35–130), so it must be a per-token property, not a provenance fact. Encoding it positionally also keeps resolution out of the instrument embedding, which the Route C adversarial arm (A14) deliberately strips — otherwise removing instrument identity would also blind the model to spectral resolution and confound that ablation. Generalized form (ablation A7): encode the full transmission curve T(λ) sampled on a coarse grid through a small MLP (handles asymmetric filters, red leaks, airmass-dependent throughput). Adjacent-channel patching (patch size p ∈ {1, 2, 4}, tuned) for dense ground spectra to keep sequences short.
- *Scalar tokens:* albedo, slope, phase angle, H each as value+uncertainty embedded tokens with a learned modality embedding.
- *Missingness:* absent modalities are simply absent tokens (attention never sees them) — no imputation, no mask value. A learned [MISSING-m] summary token per absent modality lets the model reason about *what it doesn't know* (used by the abstention head, H5).
- *Instrument/provenance embedding:* learned embedding per source catalog (Gaia/SDSS/SkyMapper/MOVIS/SpeX/ground-VIS) added to every token — lets the model absorb inter-instrument systematics explicitly rather than confounding them with composition; can be adversarially removed in Route C.

**Encoder:** pre-norm Transformer; base config d_model=192, 6 layers, 6 heads, GELU, DropPath 0.1, LayerScale init 1e-4; ~2.7M parameters. Deliberately small: our unlabeled corpus is ~10⁶ short sequences, labeled pool ~10³·⁷ — a 100M-parameter model would be indefensible (Review §6.5 overfitting critiques). Model-size sweep in A13.

**Pretraining objective:** masked reconstruction with a heteroscedastic head — mask a fraction ρ of tokens (spectral patches and scalars alike), predict (μ, σ_pred) for each masked value, minimize Gaussian NLL weighted by measurement uncertainty. Rationale: MSE reconstruction on noisy Gaia bands teaches the model to reproduce noise; NLL with predicted variance teaches it to *know its uncertainty per wavelength* — which feeds H4/H5. Auxiliary objective (weight 0.1): cross-epoch consistency — two observations of the same object (SDSS epochs; Gaia vs SkyMapper) pulled together in embedding space (light contrastive term; full contrastive is variant A1c).

**Fine-tuning heads:**
- *Hierarchical classifier:* head 1 predicts complex (C/S/X/end-member, 4-way); head 2 predicts class conditioned on complex (masked softmax). Loss = λ_c·CE(complex) + λ_t·CE(type), λ tuned. Matches the demonstrated recoverability hierarchy (Review §6.3) and gives graceful degradation for poor inputs.
- *Dual-taxonomy output:* separate linear heads for Bus-DeMeo and Mahlke labels off the same pooled embedding, trained jointly where both labels exist. Sidesteps the two-taxonomy schism (Review §2.2) and doubles the paper's audience.
- *Prototypical option for rare classes:* replace head 2's linear layer with distances to learned class prototypes (episodic training on B3) — designed for the few-shot regime; ablation A6c.
- *Abstention/OOD score:* max conformal set size + embedding KDE density (Lin et al. 2026 template).

### 3.2 Hyperparameter optimization — exact plan

**Tooling:** Optuna with ASHA early stopping (grace period 20 epochs, reduction factor 3), all trials logged to a public W&B project (reproducibility asset). Tuning data: inner-validation folds only (§2.4). Two stages:

**Stage 1 — pretraining HPO (~150 trials, proxy metric = masked-reconstruction NLL on held-out unlabeled 5% + linear-probe balanced accuracy on a fixed small labeled probe set; the probe prevents optimizing reconstruction at the cost of linearly-separable structure):**

| Hyperparameter | Search space | Prior/notes |
|---|---|---|
| Learning rate (AdamW) | log-uniform [1e-4, 3e-3] | peak, cosine decay to 1e-6 |
| Warmup fraction | {0.03, 0.06, 0.1} | |
| Batch size | {256, 512, 1024} | with lr scaled √(batch/256) |
| Weight decay | {0.01, 0.03, 0.05} | |
| Mask ratio ρ | {0.3, 0.45, 0.6, 0.75} | expect optimum 0.45–0.6 for short sequences (16-band Gaia ≠ 196-patch images; extreme MAE-style 0.75 likely too destructive) |
| Masking granularity | {per-token, per-modality-block, mixed 50/50} | per-modality-block teaches cross-modal inference (predict albedo from spectrum) — expected winner for H2/H5 |
| Patch size p (dense spectra) | {1, 2, 4} | |
| d_model | {96, 128, 192, 256} | |
| Depth | {4, 6, 8} | |
| Heads | {4, 6, 8} | constrained d_model/heads ≥ 24 |
| DropPath | {0.0, 0.1, 0.2} | |
| Contrastive auxiliary weight | {0, 0.05, 0.1, 0.3} | |
| Epochs | up to 600, early stop patience 50 on proxy metric | |

**Stage 2 — fine-tuning HPO (~250 trials per benchmark configuration, metric = inner-val type-level macro-F1):**

| Hyperparameter | Search space | Notes |
|---|---|---|
| Learning rate | log-uniform [1e-5, 1e-3] | |
| Layer-wise lr decay | {0.65, 0.75, 0.85, 1.0} | 1.0 = uniform |
| Adaptation mode | {linear probe, full fine-tune, LoRA r∈{4,8,16}} | LoRA included because the labeled set is tiny; freezing may win |
| Epochs / early stop | ≤200, patience 25 | |
| Class-imbalance loss | {CE, focal γ∈{1,2}, logit-adjusted τ∈{0.5,1.0}, LDAM+DRW} | logit adjustment is the theoretically cleanest for long-tail; LDAM the empirical strong baseline |
| Label smoothing | {0, 0.05, 0.1} | interacts with calibration — record ECE per setting |
| λ_c : λ_t (hierarchy weights) | {1:1, 1:2, 2:1, complex-annealed} | |
| Modality dropout p_mod | {0, 0.1, 0.2, 0.3} | randomly drop albedo/slope tokens in training → robustness to missing modalities at test (key for H2/H5) |
| Mixup / manifold mixup α | {off, 0.2, 0.4} | within-complex only, to avoid nonsensical cross-complex interpolants — a physics-motivated restriction referees will like |
| Augmentation suite (§4.3) | {off, noise-only, physics-full} | |
| Seed protocol | best config rerun ×10 seeds | all reported numbers = mean ± std over seeds |

**Compute budget:** pretraining run ≈ 2–4 h on one A100 (10⁶ short sequences, 2.7M params); full Stage 1 ≈ 4–8 GPU-days with ASHA; Stage 2 ≈ 3–5 GPU-days; ablations ≈ 10 GPU-days; total **< 1 GPU-month on a single A100** (or ~2 months on a consumer 4090). State this in the paper — accessibility is a selling point against foundation-model skepticism.

### 3.3 Planned architecture adjustments (pre-specified contingencies)

Pre-registering these prevents post-hoc-looking changes:

1. **If short Gaia sequences underuse attention** (16 tokens; attention may be overkill — diagnosable via attention-entropy collapse): swap encoder for a hybrid Conv1D-front-end (2 conv blocks, kernel 3) + 4 transformer layers, as in Tang 2025's ASC-Net front end; or test an MLP-Mixer variant. Reported as A13b regardless of trigger.
2. **If heteroscedastic NLL destabilizes early training** (known σ-collapse failure mode): β-NLL parameterization (weight gradient by σ^2β, β=0.5) or warm-start 20 epochs with MSE.
3. **If cross-modal block masking makes albedo reconstruction trivially dominate:** per-modality loss normalization by average NLL.
4. **If the hierarchical head underperforms flat softmax** (possible — hierarchy can propagate complex-level errors): switch primary to flat + hierarchy as auxiliary loss only; A5 quantifies.
5. **If LoRA/linear-probe beats full fine-tuning** (likely at n≈4,500): make parameter-efficient adaptation the headline configuration and reframe MAST as a reusable frozen encoder — arguably a *stronger* paper (a community-usable "asteroid spectral embedding").
6. **Tokenizer stress case:** if Fourier wavelength encoding interpolates poorly across the VIS/NIR gap, add learned per-instrument wavelength calibration offsets (2 params/instrument).

---

## 4. Alternative Routes (specified to pivot-ready depth)

### 4.1 Route B: Deep generative taxonomy — "MCFA, but deep"

**Positioning:** if H1 fails (pretraining gains marginal), the field's appetite shifts from "better classifier" to "better taxonomy." Route B directly extends Mahlke's MCFA (Review §2.2) — the linear latent model — with a nonlinear, uncertainty-aware successor. Publishable even without a classification-accuracy win, because its contribution is *taxonomic structure discovery* plus OOD detection.

**Model:** a mixture-prior VAE over the same token representation:
- Encoder = the MAST tokenizer + transformer (shared code); amortized inference with missing-data-aware set encoding (Partial-VAE / set-transformer pooling — no imputation).
- Latent: z ∈ R^d, d ∈ {4, 8, 16} (d=4 replicates Mahlke's interpretable dimensions: slope, 1-µm band, 2-µm band, albedo; d=8/16 test whether more structure exists).
- Prior: mixture of K Gaussians, K ∈ {20, 35, 50} (Mahlke used 50 components → 17 classes), or a VampPrior variant; component-to-class mapping by the same many-to-one merging protocol Mahlke documented.
- Decoder: heteroscedastic, per-modality.
- Losses: ELBO with β-annealing (β: 0→1 over 100 epochs, capacity-controlled); optional light supervision (5–20% labels, semi-supervised M2-style) to anchor class identity without dictating boundaries.

**Key experiments:** (i) does the deep latent space reproduce the 17 Mahlke classes unsupervised, and does it *split or merge* any of them with statistical support (Bayes factors between K settings)? A data-driven verdict on disputed classes (S-subtypes; Z; P-in-C-complex) would be scientifically significant regardless of ML metrics. (ii) Calibrated posterior class probabilities under missing modalities vs MCFA's. (iii) OOD via marginal-likelihood ranking.

**HPO:** lr [1e-4, 1e-3]; β schedule {linear, cyclical}; K and d as above; free-bits {0, 0.5, 1.0} nats; 100 trials, model selection by held-out ELBO + cluster stability (ARI across seeds ≥ 0.8 required — clustering papers die on seed instability).

### 4.2 Route C: Physics-informed transfer pipeline

**Positioning:** cheapest, most conservative route; also its components are *harvestable* into Route A (its augmentation and domain-adaptation modules are Route A ablation arms A12/A14). Standalone only if both A and B underdeliver; then framed as "closing the survey-transfer gap with physics."

**Components:**
1. **Physics-based augmentation** replacing the field's statistically naive PCA-noise cloning (critiqued in Review §6.1): (a) space-weathering transforms via Hapke-model npFe⁰ coating curves applied to olivine/pyroxene-bearing classes; (b) phase-reddening slope perturbations drawn from the empirical 0.067%/100nm/deg relation (Sanchez 2012) over α ∈ [0°, 30°]; (c) instrument response resampling: project ground spectra through Gaia/SDSS/LSST bandpasses with realistic per-band noise draws — generating *paired* (ground, survey) views of the same object.
2. **Domain-adversarial alignment:** gradient-reversal head predicting source catalog from the embedding (λ_DANN ∈ {0.01, 0.1, 0.3, 1.0} annealed), plus CORAL loss variant {on/off} — trained on the paired views from (1c).
3. **Classifier:** gradient boosting (XGBoost: max_depth {4,6,8}, η {0.03, 0.1}, subsample {0.7, 0.9}, colsample {0.6, 0.8}, min_child_weight {1, 5}, 2,000 trees early-stopped) on engineered features (band centers/depths/areas via polynomial fits, slope, albedo, PCs) AND the MAST embedding — an honest test of "do we even need deep learning" that doubles as ablation A1d.

### 4.3 Route interaction and the augmentation suite

The physics augmentation (C1) is used inside Route A fine-tuning as the "physics-full" arm of the augmentation hyperparameter. The paired ground/survey views from C1c also power the contrastive pretraining variant (A1c). Routes are thus not mutually exclusive; they are an ordered portfolio sharing one codebase.

---

## 5. Evaluation Protocol

### 5.1 Benchmarks (all pre-registered before final training)

- **B1 — In-domain:** 10-fold object-level CV on the labeled pool. Primary metric: **type-level macro-F1** (Bus-DeMeo, all classes with n≥5; report per-class table). Secondary: balanced accuracy, MCC, complex-level metrics. This forces honest comparison against Klimczak 2021's 76.8% balanced accuracy — the field's real number — instead of collapsed-class headlines.
- **B2 — Cross-survey transfer (headline benchmark):** as §2.4. Metric: transfer macro-F1 and the **transfer gap**. Baseline anchor: Sullivan's documented 92→58 collapse; Delbo 2026's per-class Gaia agreement (S>92 … K 59) reproduced under our protocol for comparability.
- **B3 — Few-shot rare classes:** episodic; metric = rare-class mean recall at K shots.
- **B4 — Calibration & OOD:** ECE (15-bin, equal-mass), reliability diagrams, Brier score; conformal marginal and class-conditional coverage at α=0.1 with average set size; OOD AUROC on the §2.4 suite.
- **B5 — Scientific sanity (Tinaut-Ruano-style):** apply the final model to all 60,518 Gaia DR3 objects; check dynamical-family purity (Vesta→V, Trojans→primitive ≥99%, Hungaria→E fraction), heliocentric S→C gradient, NEO Q/S ratio vs Binzel 2019. These population-level checks are what planetary-science referees actually trust.

### 5.2 Baselines (all reimplemented under identical splits — the comparison table no prior paper has)

1. PCA + decision tree (canonical DeMeo 2009 pipeline, via `classy`).
2. Penttilä 2021 FFNN (30 hidden, 5-vote ensemble) — faithful reimplementation.
3. Klimczak-style XGBoost and MLP on PCs + slope.
4. PCA + KDE posteriors (Delbo 2026).
5. MCFA (Mahlke's released code).
6. Tang 2025 ASC-Net (reimplemented; their augmentation *disabled* and *enabled* — quantifying how much of 95.7% was augmentation).
7. k-NN color classifier (Popescu/Erasmus style) for photometric inputs.

### 5.3 Statistical protocol

- 10 seeds per final configuration; mean ± std everywhere; no single-split numbers anywhere in the paper.
- Pairwise significance: exact McNemar on pooled CV predictions for classifiers on identical folds; paired bootstrap (10⁴ resamples, object-level) for macro-F1 deltas; Holm–Bonferroni across the baseline family.
- Pre-registered primary comparisons (to avoid garden-of-forking-paths critique): (i) MAST vs Penttilä-FFNN on B1 macro-F1; (ii) MAST vs best non-pretrained ablation on B2 transfer gap; (iii) coverage validity on B4. Everything else labeled exploratory.
- Effect-size floor for claiming "significant improvement" in the abstract: ≥3 points macro-F1 **and** p<0.01 **and** consistent sign across ≥8/10 seeds.

---

## 6. Ablation Matrix (attributing every gain)

Each row: 5 seeds × relevant benchmarks; ~40 configurations total ≈ 6 GPU-days.

| ID | Ablation | Isolates | Compared configurations |
|---|---|---|---|
| A1 | Pretraining objective | H1 | none (scratch) / masked-NLL / masked-MSE / contrastive (A1c) / MAST-embedding→XGBoost (A1d) |
| A2 | Pretraining corpus size | H1 scaling | 0% / 10% / 25% / 50% / 100% of unlabeled corpus; also Gaia-only vs +SDSS vs +all |
| A3 | Modalities | H2 | spectrum-only / +albedo / +slope / +phase / all |
| A4 | Missing-data mechanism | H2 | native attention-masking / mean imputation / k-NN imputation / MCFA-style FA imputation |
| A5 | Head structure | — | hierarchical / flat / flat+hierarchy-as-auxiliary |
| A6 | Imbalance strategy | rare classes | plain CE / focal / logit-adjusted / LDAM+DRW / prototypical head (A6c) |
| A7 | Wavelength/bandpass encoding | tokenizer novelty | Fourier-(λ, Δλ) / Fourier-λ only (bandwidth ablated) / full transmission-curve MLP encoding / index positional / fixed-grid-resample (the field's default). Targeted probe: Ch-vs-C discrimination from broadband-only inputs, the case where bandwidth awareness should matter most (0.7 µm feature dilution) |
| A8 | Reconstruction loss | — | heteroscedastic NLL / MSE / β-NLL |
| A9 | Slope policy | §2.3 | dual-stream / slope-in / slope-removed-entirely |
| A10 | Orbit-feature leakage | shortcut learning | +orbital elements, evaluated on random split VS family-blocked split (the gap between those two numbers IS the leakage measurement) |
| A11 | Calibration method | H4 | raw softmax / temperature scaling / conformal marginal / Mondrian conformal |
| A12 | Augmentation | Route C harvest | off / Gaussian-noise / PCA-clone (field standard) / physics-full |
| A13 | Capacity | overfitting critique | d_model×depth grid {96×4, 192×6, 256×8, 384×8}; plus conv-hybrid (A13b) |
| A14 | Domain adaptation | H3 | DANN on/off / CORAL on/off / instrument-embedding removed |

**Reporting rule:** the paper's architecture figure gets one number per component = the drop when ablated, so the "novel architecture" claim is component-wise quantified — the strongest possible answer to "is this just a bigger model."

---

## 7. Defense Dossier: Anticipated Referee Objections and Prepared Responses

1. **"Novelty — transformers already applied (Tang 2025)."** Tang is supervised, fixed-grid, single-instrument, augmentation-dependent, uncalibrated. Our contribution is not "a transformer" but the pretraining paradigm + heterogeneous-instrument tokenization + calibration + transfer protocol; A1/A7 isolate each. We reimplement Tang under our splits (baseline 6).
2. **"Your labels are circular (Review §6.4)."** Conceded in a dedicated discussion section, and mitigated three ways: dual-taxonomy heads (not privileging one human clustering), Route-B unsupervised structure analysis of the same latent space, and B5 population-level physical checks that do not depend on label fidelity. We claim improved *label reproduction and transfer*, explicitly not improved *mineralogy* — with meteorite-spectrum evaluation (Burbine 2024 set) as an appendix bridge.
3. **"Small labeled data + big model = overfitting."** 2.7M params, LoRA/linear-probe arms, A13 capacity sweep, nested CV with untouched test folds, 10-seed reporting. The pretraining corpus is the point: the model that sees only 4,500 labeled objects is the *baseline*, not our method.
4. **"Gaia systematics invalidate pretraining data."** Band-drop policy per Tinaut-Ruano 2023/2026; per-band flags and S/N as model inputs; instrument embeddings absorb residual systematics; A14 shows removing them hurts transfer. We do not use the NUV artifact region for F/B claims.
5. **"Synthetic augmentation inflates results (as you yourselves critique)."** Physics-grounded only, ablated (A12), and *never applied to test data*; headline numbers reported with augmentation off as well.
6. **"Why believe the calibration under shift? Conformal guarantees assume exchangeability."** Correct — stated honestly: marginal validity is guaranteed in-domain; under shift we *measure* empirical coverage (B4) rather than claim it, and use weighted conformal (likelihood-ratio reweighting by embedding density) as a mitigation arm of A11.
7. **"Orbit features would have helped — why exclude?"** A10 quantifies exactly what they add and how much of it is dynamical-family leakage; we include the family-blocked evaluation most prior work lacks.
8. **"Where is LSST/SPHEREx?"** The tokenizer ingests their formats natively — including their spectral resolution, via the (λ_eff, Δλ) positional coordinate, so a novel instrument is interpreted correctly on first contact *within the (λ, Δλ) region covered in training*. We demonstrate on simulated LSST colors (Penttilä 2022 protocol). Scoping stated plainly: LSST falls inside our trained wavelength–bandwidth support; SPHEREx beyond 2.45 µm does not — no encoding fixes extrapolation into unobserved spectral territory, and full SPHEREx readiness requires its 1–2.5 µm overlap data plus retraining, claimed as the successor paper, not this one.

---

## 8. Timeline, Decision Gates, and Deliverables

| Phase | Months | Work | Gate |
|---|---|---|---|
| P1 | 1–2 | Data engineering (§2), label harmonization, baseline reimplementations (§5.2), split freeze + pre-registration of §1/§5.3 | Baselines reproduce literature numbers ±2 pts → proceed |
| P2 | 2–4 | MAST pretraining + Stage-1 HPO; linear-probe tracking | **Gate G1:** linear probe beats PCA features by ≥3 macro-F1 → Route A confirmed. Else → Route B primary, Route C harvest |
| P3 | 4–6 | Fine-tuning + Stage-2 HPO; Route B pilot in parallel (shared encoder) | **Gate G2:** H1 criterion met on inner val → freeze architecture |
| P4 | 6–8 | Full ablation matrix (§6); B2/B3/B4 benchmarks | |
| P5 | 8–9 | B5 Gaia-wide application; catalog production (60,518 classifications with conformal sets) | |
| P6 | 9–12 | Paper (target: **A&A** primary; PSJ alternate), code + weights + catalog release (Zenodo DOI, `classy` integration PR), response-to-referees buffer | |

**Deliverables:** (1) the paper; (2) open-source MAST code + pretrained weights; (3) a value-added catalog — probabilistic dual-taxonomy classifications with conformal sets for all usable Gaia DR3 asteroids — the artifact that guarantees citations regardless of methodological reception; (4) documented readiness for Gaia DR4 (Dec 2026: 436,551 spectra — a 7× pretraining corpus arriving exactly when this project matures, enabling an immediate high-impact follow-up).

**Risk register (top 3):** (i) Gaia DR4 slips or scoops us via a Gaia-consortium classifier → mitigation: our transfer/calibration contributions are survey-agnostic; (ii) pretraining gains are real but <3 pts → reframe around calibration + missing-data + catalog (still exceeds field standard); (iii) rare-class few-shot fails → report honestly; H1's rare-class clause is a secondary, not the headline.

