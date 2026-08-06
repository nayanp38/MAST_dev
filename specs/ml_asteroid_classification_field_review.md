# Machine Learning for Asteroid Spectral Classification in the Bus-DeMeo Taxonomy: A Field Review

**Prepared:** August 6, 2026
**Purpose:** Foundation document for the design of a novel, scientifically defensible ML architecture for asteroid taxonomic classification, targeting eventual peer-reviewed publication.

---

## 1. Introduction and Scope

Asteroid taxonomic classification — the assignment of asteroids to spectral classes that serve as proxies for surface composition — sits at the intersection of planetary spectroscopy and, increasingly, machine learning. The field is in a period of rapid transition: the classical labeled datasets (a few hundred to a few thousand visible/near-infrared spectra) are being dwarfed by survey data at the 10⁴–10⁶ scale (Gaia DR3 today; LSST, SPHEREx, and Gaia DR4 imminently), and the methods used to classify them are evolving from PCA-plus-decision-tree pipelines toward probabilistic, deep, and multimodal architectures.

This review covers (i) the taxonomy itself and its lineage, (ii) the data landscape, (iii) the ML methods literature through mid-2026, (iv) recurring methodological challenges, and (v) the open gaps that a novel architecture could credibly fill. Every quantitative claim was checked against a primary source during preparation; items that could not be fully verified are flagged inline.

---

## 2. Taxonomy Foundations

### 2.1 The lineage: Tholen → Bus-Binzel → Bus-DeMeo

**Tholen (1984)** built the first widely adopted modern taxonomy from the Eight-Color Asteroid Survey (ECAS; Zellner, Tholen & Tedesco 1985), covering 0.31–1.06 µm, combined with albedo. Its 14 classes (A, B, C, D, E, F, G, M, P, Q, R, S, T, V) were derived by cluster analysis on principal components of the ECAS colors. Critically, the E/M/P split is defined *only* by albedo — spectrally these classes are degenerate and collapse to "X" without it. This albedo dependence, dropped by later taxonomies and now being reinstated, is a recurring theme in the modern ML literature.

**Bus & Binzel (2002)** used SMASSII CCD spectra of 1,447 asteroids (~0.44–0.92 µm) to define a 26-class, feature-based visible-wavelength taxonomy via PCA on slope-removed spectra. Albedo was deliberately excluded (visible spectra only), which introduced the X-complex degeneracy into the standard system. Numerous intermediate subclasses (Sa, Sk, Sl, Sq, Sr, Cb, Cg, Cgh, Ch, Xc, Xe, Xk, Ld, ...) were introduced.

**DeMeo, Binzel, Slivan & Bus (2009)** — the **Bus-DeMeo taxonomy**, the field standard — extended the Bus system into the near-infrared using 371 asteroids with combined visible + NIR spectra over **0.45–2.45 µm** (SMASS + IRTF/SpeX). The method: spline-fit and slope-remove each spectrum, resample to 41 channels at 0.05 µm, run PCA, then apply a decision tree / flowchart in principal-component space with visual checks of diagnostic features (0.49, 0.7, 0.9, 1 µm bands). The result is **24 classes** (adding Sv; dropping visible-only Bus subclasses Sl, Sk, Ld), with a "w" notation for high-slope (weathered) variants. The taxonomy and its 24 class-mean spectra are archived at the PDS Small Bodies Node.

Two later refinements matter for ML work. **Binzel et al. (2019)**, in the MITHNEOS NEO results paper, added the **Xn class** and codified NIR-only (0.85–2.45 µm) classification; the SMASS online classifier now implements both a VIS+NIR mode and an IR-only mode over 25 classes. Note: there is **no standalone "DeMeo 2022 IR-only taxonomy" paper** — a common miscitation. DeMeo et al. (2022, Icarus 380, 114971) is the asteroid–meteorite spectral connection study (~500 asteroid spectra vs. >1,000 RELAB meteorite spectra); the IR-only classification machinery lives in Binzel et al. (2019) and the SMASS tool.

### 2.2 The Mahlke, Carry & Mattei (2022) taxonomy: the data-driven alternative

Mahlke, Carry & Mattei (2022, A&A 665, A26) rebuilt asteroid taxonomy bottom-up with a **mixture of common factor analyzers (MCFA)** — a probabilistic latent-variable clustering model with 4 latent dimensions and 50 Gaussian components, trained on 2,983 quality-controlled observations of 2,125 asteroids (spectra 0.45–2.45 µm resampled to 53 wavelengths, **plus visual albedo**). Its three defining properties, each a direct response to a Bus-DeMeo limitation:

1. **Native missing-data handling** (maximum-likelihood factor analysis, no imputation) — enables classification of partial observations (VIS-only, NIR-only, no-albedo), exactly the data regime of Gaia, SPHEREx, and NEO surveys.
2. **Albedo as a first-class feature** — resolves the X-complex degeneracy, restoring Tholen's E (~0.5), M (~0.15), P (~0.05) split via a Gaussian mixture in albedo, and moves P into the C-complex.
3. **Probabilistic output** — per-object class probabilities rather than hard labels, with "diffuse" flags for ambiguous objects.

The result is **17 classes in three complexes** (C, M, S), including a new very-red **Z class**, with S-subclasses (Sa/Sq/Sr) deliberately abandoned as artificial partitions of a continuum. Final classifications were published for 6,038 observations of 4,526 asteroids — nearly tenfold the Bus-DeMeo reference set. The accompanying `classy` Python package has made this taxonomy operationally convenient, and recent Gaia-era papers (Pentikäinen et al. 2026) are beginning to adopt Mahlke classes as training labels, making it a genuine competitor system rather than an academic exercise.

**Design implication for our work:** the field now has two live taxonomies. A new architecture should either classify in both, or make its class scheme a configurable output layer, and must engage with the Mahlke critique of Bus-DeMeo (arbitrary subclass boundaries, no missing-data path, no albedo) rather than treating Bus-DeMeo as unquestioned ground truth.

---

## 3. The Data Landscape

### 3.1 Labeled spectral datasets (the training pool)

| Dataset | Size | Coverage | Notes |
|---|---|---|---|
| SMASS/SMASSII (Bus & Binzel 2002) | 1,447 asteroids | ~0.44–0.92 µm | Visible only; classical label source |
| S3OS2 (Lazzaro et al. 2004) | 820 asteroids | ~0.49–0.92 µm | Tholen + Bus classes |
| Bus-DeMeo reference set (DeMeo et al. 2009) | 371 asteroids | 0.45–2.45 µm | THE canonical labeled set |
| MITHNEOS (Binzel et al. 2019; Marsset et al. 2022) | ~1,000+ NEOs; +491 NIR spectra of 420 NEOs in 2022 | 0.8–2.45 µm (SpeX), merged with visible | NEO-focused; slope systematics up to several %/µm between nights (Marsset et al. 2020, ApJS 247, 73) |
| Mahlke et al. (2022) compilation | 6,038 observations / 4,526 asteroids | 0.45–2.45 µm + albedo | Largest curated VIS+NIR labeled pool |

The critical structural fact: **every supervised spectral classifier in the literature ultimately trains on the same ~500–750 unique VIS+NIR objects** (DeMeo 2009 + MITHNEOS), or modest extensions thereof. This is the field's data ceiling, and it is both small and severely imbalanced (S-types ~42%; O-types n=2).

### 3.2 Large survey datasets (the classification targets — and the unlabeled pool)

- **SDSS Moving Object Catalog:** MOC4 contains 471,569 moving-object detections, with 220,101 matched to 104,449 unique asteroids (Ivezić et al. 2001; Parker et al. 2008). Sergeyev & Carry (2021, A&A 652, A59) re-mined all SDSS releases to obtain **1,036,322 observations of 379,714 unique objects**. Five photometric bands (ugriz) — effectively ~2 useful color dimensions; supports complex-level classification only (DeMeo & Carry 2013, Icarus 226, 723; the famous compositional map paper is DeMeo & Carry **2014, Nature 505, 629** — the "2013 Nature" citation seen in some drafts is wrong).
- **Gaia DR3 reflectance spectra** (Galluccio et al. 2023, A&A 674, A35): **60,518 SSOs**, mean spectra in **16 bands spanning 0.374–1.034 µm**, with per-band quality flags. Known systematics that any ML pipeline must handle: artificial NUV reddening from solar-analog choice (Tinaut-Ruano et al. 2023, A&A 669, L14), unexplained taxonomy-dependent artificial reddening at 0.7–1.0 µm, unreliable extreme bands (standard practice: drop them), and a spurious feature near 0.65 µm at the BP/RP junction. Delbo et al. (2026) recommend an S/N > 20 cut, leaving ~36,566 usable objects.
- **MOVIS** (VISTA-VHS NIR colors; Popescu et al. 2016, 2018): 53,447 SSOs; 18,265 with both (Y−J) and (J−Ks); 6,496 confidently classified into nine NIR-compatible superclasses.
- **Albedo catalogs:** WISE/NEOWISE (>100,000 main-belt albedos/diameters, Masiero et al. 2011; ~20–25% typical 1σ albedo uncertainty), AKARI AcuA (5,120 asteroids; updated to 8,097 values for 5,170 objects by Alí-Lagoa et al. 2018), IRAS, Spitzer.

### 3.3 Imminent data sources (the window of opportunity)

- **Rubin/LSST:** First Look June 2025 (>11,000 new asteroids discovered during commissioning); the **10-year survey formally began June 30, 2026**. Expected: millions of main-belt asteroids with ugrizy colors. **No LSST-data taxonomy paper exists yet** — only simulation studies (Penttilä et al. 2022; Klimczak et al. 2022, 2023).
- **SPHEREx:** launched March 2025; all-sky **0.75–5.0 µm** spectral survey (R≈35–130), expected to observe ~100,000+ small bodies. The 1–2.5 µm coverage directly addresses the taxonomy's NIR requirement, and the 3-µm region adds hydration diagnostics beyond any existing taxonomy. **No SPHEREx taxonomy paper exists yet**; supporting infrastructure is appearing (SOLO lightcurve monitoring, Lim et al. 2026, arXiv:2602.08037).
- **Gaia DR4:** will contain mean reflectance spectra for **436,551 SSOs** (7× DR3), plus epoch spectra; expected December 2026 (date from ESA communications — reconfirm before citing).
- **JWST:** no systematic taxonomy program, but 1–5 µm small-body spectroscopy is revealing surface types beyond the Bus-DeMeo inventory (e.g., a new high-albedo Trojan surface type, arXiv:2506.19922).

### 3.4 Tools

`classy` (Mahlke) classifies in Mahlke, Bus-DeMeo, and Tholen systems and aggregates public spectra; `rocks`/SsODNet (Berthier et al. 2023) resolves identities and best-estimate parameters; M4AST (Popescu et al. 2012) does template curve-matching; the SMASS online tool implements the canonical Bus-DeMeo PCA + flowchart.

---

## 4. The Machine Learning Literature

### 4.1 Precursors

The lineage begins with **Howell, Merényi & Lebofsky (1994, JGR 99, 10847)** — neural networks (unsupervised clustering + supervised categorization) on 52-color + ECAS data in the Tholen system. A random-forest study exists in the Chinese literature (Huang et al. 2016/2017). These predate Bus-DeMeo and modern sample sizes but establish that the ANN-taxonomy idea is three decades old; novelty claims must be positioned carefully.

### 4.2 Supervised classification of VIS+NIR spectra (the Helsinki/Poznań core)

**Penttilä, Hietala & Muinonen (2021, A&A 649, A46)** — often miscited as "Penttilä et al. 2020" from its DPS abstract — is the modern baseline: a single-hidden-layer feed-forward network (30 hidden neurons, tanh, softmax over 11 collapsed Bus-DeMeo classes, 5-network voting ensemble) on 586 VIS+NIR spectra (DeMeo 2009 + MITHNEOS, resampled to 200 points). Leave-one-out accuracy: **90.6%** full range, **86.5%** restricted to a Gaia-like 0.45–1.05 µm window. Rare classes were handled by collapsing to 11 classes and PCA-based synthetic augmentation (200 samples/class). The authors themselves flag that hyperparameters were tuned on all the data — a leakage risk — and call for re-evaluation on real Gaia data.

**Klimczak et al. (2021, Front. Astron. Space Sci. 8, 767885)** ran the most systematic algorithm comparison: multinomial logistic regression, Gaussian naive Bayes, SVM, gradient boosting (XGBoost), and MLP on 504 asteroids (classes with <10 members dropped), in both a 12-type and a 4-complex scheme, with 5-fold stratified CV repeated 10× and **balanced accuracy** as the headline metric. Best (MLP on PCA features): **93.2% accuracy / 90.0% balanced accuracy at complex level; 82.9% / 76.8% at type level**. The gap between those two numbers is the honest measure of how much collapsed-class schemes inflate reported performance. Feature analysis: slope plus reflectance near 0.65, 0.9, 1.05, 1.1 µm carry most information.

**Luo et al. (2023/2024, AJ 167, 13)** applied a small ANN ensemble to 834 SMASS II visible spectra (10 collapsed Bus classes), 92% test accuracy, aimed at China's CSST — evidence the "shallow ensemble on few hundred spectra" recipe has become commodity.

### 4.3 Simulated-survey studies (pre-computing the future)

**Penttilä, Fedorets & Muinonen (2022, Front. Astron. Space Sci. 9, 816268)** convolved the 586-spectrum set with LSST ugrizy filters: **85.8%** (11 classes) rising to **90.1%** with a simplified 7-class merge. **Klimczak et al. (2022, A&A 667, A10)** did the same across 11 filter systems (27,500 models): NIR filter sets (VISTA Y/J/H/Ks) beat visible sets (~93% vs ~80–90% complex-level balanced accuracy) because they sample the 1 and 2 µm silicate bands; LSST+VISTA combined reached ~96%/90%. Both are upper bounds — no noise, no rotational variability, and spectrum "cloning" inflates estimates (~20% drop without it). **Klimczak et al. (2023, AJ 166, 230)** inverted the question into ML-driven filter design: 5 optimally chosen bands recover 85% complex-level balanced accuracy.

### 4.4 Photometric-survey classification at scale

- **Sergeyev & Carry (2021)**: probabilistic complex assignment for >1M SDSS observations by intersecting each observation's 3D color-uncertainty ellipsoid with Bus-DeMeo class volumes — probabilistic but not learned.
- **Sergeyev et al. (2022, A&A 658, A109)**: same approach on SkyMapper (880,528 observations of 205,515 SSOs); confusion matrix against 1,683 spectrally classified objects shows rare classes (A, K, L, Q) leaking into S.
- **Popescu et al. (2018)**: MOVIS NIR colors with dual-method (probabilistic model + 3-NN with Monte Carlo error resampling) agreement requirement, ~95% leave-one-out accuracy on reference objects — an early, clean example of consensus-based confidence.
- **Erasmus et al. (2017 AJ; 2018 ApJS 237, 19; 2019 ApJS 242, 15)**: k-NN on VRI colors (KMTNet-SAAO) for NEOs and >2,000 main-belt asteroids in 4 complexes (S/C/X/D), with Monte Carlo resampling of color uncertainties.
- **Roh et al. (2022, A&A 664, A51)**: Gaussian mixture models (infinite/unsupervised + finite/semi-supervised K=12) in SDSS 3D color space; X-type consensus assignment essentially failed — an honest negative result on featureless-class separation in broadband data.
- **Colazo et al. (2022, A&A 666, A77)**: fuzzy C-means on phase-corrected absolute colors of 9,481 asteroids, 4 complexes.
- **Choi et al. (2023, PSJ 4, 49)**: KMTNet griz photometry of 6,793 asteroids; 2D criteria + semi-supervised GMM; again X vs. C poorly separated.

### 4.5 The Gaia DR3 era (2023–2026): the field's center of gravity

Four 2024–2026 papers define the current state of the art:

**Tinaut-Ruano et al. (2026, A&A 711, A167)** — the largest spectroscopy-based taxonomy ever: **14,042 asteroids** classified from Gaia DR3 spectra + albedos via an iterative, hierarchical PCA + GMM soft-clustering pipeline, each iteration targeting specific features. Custom 13-class scheme (A, B, C, D, E, F, G, K, L, M, P, S, V): the NUV coverage revives Tholen's F and G classes; albedo eliminates X in favor of E/M/P. Validation is by template comparison and dynamical-family sanity checks (99.4% of Trojans primitive; 61% of Hungarias E-type) — no supervised accuracy metric exists because there is no train/test split.

**Delbo et al. (2026, Front. Astron. Space Sci. 13, 1774478)** — supervised counterpart: PCA (3 components) + per-class multivariate KDE posteriors, trained on 2,653 S/N≥50 Gaia spectra with literature labels, applied to all 60,518. Per-class agreement: S >92%, V 99%, A 92%, D 91%, B 82%, C 72%, X 71%, L 66%, K 59% — a crisp quantification of which classes survive the Gaia wavelength range and which dissolve.

**Pentikäinen et al. (2026, A&A 707, A132)** — LDA + k-NN fusing Gaia spectra (418–770 nm), photometric phase slopes, and albedos, trained on 328 Mahlke-labeled asteroids; classifies 1,668 new objects in Mahlke classes. Albedo was the single most influential feature; C vs. P remained inseparable; M reached only ~44% agreement.

**Ge et al. (2024, A&A 692, A100)** — the most architecturally ambitious: "AsterRF," a weighted ensemble of a dual-branch attention network (spectral branch + physical/orbital-parameter branch fused by self-attention) and a random forest, on Gaia DR3 spectra + 5 selected parameters (H, G, albedo, orbit type, aphelion). **92.2%** accuracy over 6 merged classes — **+7.8 points over spectrum-only**, the cleanest published quantification of the multimodal dividend. A follow-up (Ge et al. 2025, ApJS, "AadRF") jointly inverts type + albedo + diameter for 58,168 asteroids. Caveats: heavy Gaussian-perturbation augmentation (4,961 real → 15,000 samples) and 10-fold CV on the *augmented* set, which risks train/test contamination by near-duplicate clones.

### 4.6 Deep architectures

**Tang et al. (2025, AJ 169, 201; arXiv:2502.16458)** is the first (and so far only) transformer-based asteroid spectral classifier: ASC-Net (1D-conv encoder + multi-head self-attention with skip connections and a "differential module" emphasizing slope/peak structure), plus an autoencoder-transformer for olivine/pyroxene composition and an attention-conditioned albedo regressor. Reported: **95.69%** on an 11-class collapsed Bus-DeMeo task (vs. baselines: PCA+SVM 84.5%, RF 85.1%, ANN 90.9%, ResNet-50 92.5%, vanilla transformer 91.7%). Important caveats: training data heavily inflated by PCA+noise synthetic augmentation (to 8,000 samples), and the evaluation is a single random split. The composition module signals the emerging "mineralogy-first" direction (see §6.4).

**Korda et al. (2023, A&A 669, A101)** — 1D CNN regressing modal/chemical mineral composition (olivine, ortho/clinopyroxene) directly from reflectance spectra of laboratory minerals and meteorites, within ~10 percentage points on test data; applied to S/Q/V/A asteroids. Not taxonomy, but the standard reference for the composition-regression alternative.

On the meteorite side: **Dyar et al. (2023, Icarus, 115718)** classified 1,422 RELAB meteorite spectra at ~92% (logistic regression) and mapped classes onto asteroids; **Burbine et al. (2024, PSJ 5, 10.3847/PSJ/ad57b6)** pushed ~1,500 meteorite spectra through the Bus-DeMeo pipeline, confirming V↔HED and S/Q↔ordinary-chondrite links but demonstrating that featureless classes (D/X) match both CM chondrites *and* irons — the definitive recent statement that Bus-DeMeo classes are not mineralogically unique.

### 4.7 What does NOT exist (verified negative results, as of August 2026)

These absences were checked by multiple targeted searches and are the field's open lanes:

1. **No self-supervised / contrastive / masked-autoencoder pretraining** on unlabeled asteroid survey spectra, despite ~10⁵–10⁶ unlabeled spectra sitting in Gaia/SDSS and an explicit call for masked autoencoders in the adjacent TNO literature (Lin et al. 2026, arXiv:2604.23840).
2. **No conformal prediction or formal calibration analysis** in any asteroid-taxonomy paper.
3. **No published taxonomy from real LSST or SPHEREx data** (both now flowing).
4. **No ML fusion of polarimetric phase curves** with spectra, despite polarimetry being the textbook complement for albedo/X-complex degeneracies.
5. **Only one transformer paper** (Tang et al. 2025), and no graph/set-based architectures for heterogeneous multi-instrument data.
6. **No dedicated MITHNEOS/NEO-specific deep classifier** (Marsset et al. 2022 is debiasing, not ML).
7. **No UMAP/t-SNE/modern-manifold rediscovery study** of taxonomy structure (the unsupervised role is filled by GMM/MCFA work).

---

## 5. Comparative Summary of Key Supervised Results

| Paper | Data (n) | Method | Classes | Validation | Headline result |
|---|---|---|---|---|---|
| Penttilä+ 2021 | 586 VIS+NIR spectra | FFNN ensemble | 11 (collapsed BD) | LOO-CV | 90.6% (full), 86.5% (Gaia range) |
| Klimczak+ 2021 | 504 spectra | LR/NB/SVM/XGB/MLP | 12 types / 4 complexes | 5-fold ×10 | 76.8% BAcc types; 90.0% BAcc complexes (MLP) |
| Penttilä+ 2022 | 586 → LSST colors | FFNN | 11 / 7 | LOO-CV | 85.8% / 90.1% |
| Klimczak+ 2022 | 752 (×10 clones) → 11 filter sets | 5 algorithms | types + complexes | 5×5-fold | VISTA ~93%/85% BAcc; Gaia 3-band 81%/63% |
| Luo+ 2023 | 834 SMASS II | ANN ensemble | 10 (Bus) | 4:1 split | 92% |
| Popescu+ 2018 | 371 refs → MOVIS | prob. model + 3-NN | 9 NIR groups | LOO | ~95% |
| Roh+ 2022 | 4,213 SDSS | GMM (semi-sup.) | 7 | unsupervised | no global accuracy; X failed |
| Ge+ 2024 | 4,961 Gaia (→15k aug.) | attention fusion + RF | 6 merged | 10-fold (aug.) | 92.2% (+7.8 vs. spectrum-only) |
| Tang+ 2025 | SMASS II + VisNIR (aug. to 8k) | transformer (ASC-Net) | 4 / 11 | held-out split | 94.6% / 95.7% |
| Delbo+ 2026 | 2,653 train / 60,518 applied | PCA + KDE posterior | 9 | literature agreement | S>92% … K 59% |
| Pentikäinen+ 2026 | 328 train (Mahlke labels) | LDA + k-NN | Mahlke classes | 1,000 randomized runs | S 92%, Ch 85%, M ~44% |

**Reading this table honestly:** the 90–96% headlines all come from collapsed class schemes (≤11 classes), tiny in-domain test sets, and/or synthetic augmentation. The two numbers that best represent *deployed* difficulty are Klimczak's 76.8% balanced accuracy at type level and the Sullivan (2023, LJMU thesis) transfer result — models at 91–92% on held-out spectra dropping to **~58%** on real SDSS data. That ~30-point in-domain-to-survey gap is the field's real frontier, not the last 3 points of LOO accuracy.

---

## 6. Methodological Challenges (What a Defensible Paper Must Confront)

### 6.1 Class imbalance

S-types are ~42% of the canonical labeled set; O-types have two members. The literature's three coping strategies each carry a documented failure mode: (a) **class merging** dissolves precisely the rare classes of scientific interest; (b) **synthetic oversampling** (PCA-noise, SMOTE-ENN, VAE) overfits — Sullivan (2023) showed a VAE-balanced model at 92% on spectra collapsing to 58% on real SDSS; (c) **dropping rare classes** (<5 or <10 members) simply redefines the problem. Tinaut-Ruano's hierarchical iterative clustering (isolate dominant classes early, target rare taxa in later layers with tailored features) is the most interesting structural response. **No paper demonstrates few-shot rare-class recognition that survives survey transfer.** Ge et al. (2024) state the consensus: more real rare-class samples are the actual bottleneck.

### 6.2 Spectral degeneracies and physical confounders

- **X-complex:** E/M/P are spectrally identical and separated only by albedo (~0.5/~0.15/~0.05). Every albedo-free VIS classifier structurally fails here (Delbo: X 71%; Roh: X failed; Ge: X worst class). Every method that recovers E/M/P injects albedo (Mahlke, Tinaut-Ruano, Pentikäinen, Ge).
- **S-subclass continuum:** Sa/Sq/Sr boundaries are fuzzy partitions of a continuum; Mahlke abandoned them outright, and Penttilä's S/Q confusion is persistent. A hard-label architecture inherits this arbitrariness; a probabilistic/ordinal treatment does not.
- **Space weathering & phase reddening are entangled slope effects:** phase reddening at 30–120° mimics 0.1–1.3 Myr of space-weathering reddening (Sanchez et al. 2012), and Gaia spectra carry phase-dependent slope changes plus *artificial* NUV reddening exceeding the true F/B class separation (Tinaut-Ruano et al. 2023) — causing real, documented misclassifications (Polana members typed B instead of F). Instrumental slope systematics of several %/µm exist even in the SpeX training data (Marsset et al. 2020). **Slope is therefore the least trustworthy feature in the entire pipeline**, yet it consistently ranks among the most informative (Klimczak 2021) — a tension no published classifier resolves explicitly.

### 6.3 Wavelength-coverage mismatch

The taxonomy is defined on 0.45–2.45 µm; the big surveys deliver 0.35–1.05 µm. Quantified costs of losing the NIR: −4 to −5 accuracy points at complex level (Penttilä 90.6→86.5; 90.6→85.8), but *qualitative collapse* at subclass/rare-class level (Sullivan: every A, B, K, Q object reassigned at broadband resolution; Penttilä: all T-types lost; Delbo: K 59%, L 66%). Conversely, Klimczak (2022) showed NIR filter sets (sampling the 1 and 2 µm bands) beat any visible set, and Gaia's NUV added back the Tholen F/G distinction (Tinaut-Ruano 2026). The field has quietly bifurcated: complex-level taxonomy for VIS surveys, class-level fidelity only with NIR and/or albedo. SPHEREx (0.75–5 µm) is the first survey that can heal this split — and has no classifier built for it yet.

### 6.4 Label noise and epistemic circularity

Training labels descend from a human clustering built on small VIS-heavy samples. Documented symptoms: 549/2,125 asteroids with repeat spectra land in different classes across observations (Mahlke 2022); ~30% of objects get different classes between SDSS and spectral catalogs (Sullivan 2023); unsupervised clustering finds only the C- and S-complex *cores* to be robust density structures. Meteorite ground truth cuts both ways: same taxon / different mineralogy and vice versa (Burbine et al. 2024; DeMeo et al. 2022). Mahlke's own P-class reassignment leaned on meteorite/IDP interpretation — an acknowledged mineralogy↔taxonomy feedback loop. A supervised classifier scoring 90% is faithfully reproducing this imperfect clustering, not exceeding it. The two live responses: rebuild the label space from data (Mahlke), or bypass taxonomy for direct composition regression (Tang 2025; Korda 2023). A defensible new paper must state clearly which game it is playing.

### 6.5 Validation and uncertainty quantification

The dominant weakness. In-domain random splits (85–96%) are systematically optimistic versus cross-survey transfer (~58%, Sullivan). Best current practices, scattered across papers but never combined: nested CV with balanced accuracy/MCC (Klimczak 2022); ensemble class-probability outputs with disagreement flags (Penttilä); KDE posteriors with S/N-based reliability cuts (Delbo 2026); mixture-model probabilities with diffuse flags (Mahlke); dual-method consensus (Popescu 2018). Missing everywhere: **calibration testing** (does 80% confidence mean 80% accuracy?), **conformal/coverage guarantees**, and **explicit domain-shift evaluation** (train on SMASS, test on Gaia, report the drop). The nearest template is in the TNO literature: Lin et al. (2026) verify ~95% empirical coverage of posterior credible intervals and detect OOD objects via posterior-variance inflation.

---

## 7. The Opportunity Map: Where a Novel Architecture Can Win

Synthesizing §4.7 and §6, the gaps rank as follows by scientific payoff × feasibility:

**Tier 1 — the structural gap.** *Self-supervised pretraining on the unlabeled survey corpus.* ~10⁵ Gaia DR3 spectra (10⁶ SDSS observations; soon 4×10⁵ Gaia DR4) are unlabeled; every published classifier trains supervised on ≤~15,000 (usually ≤~600 unique) labeled objects. A masked-autoencoder or contrastive encoder pretrained on unlabeled spectra, fine-tuned few-shot on Bus-DeMeo/Mahlke labels, attacks imbalance, label scarcity, and transfer *simultaneously* — and is explicitly called for but unbuilt. This is the single clearest "first paper to do X" lane.

**Tier 2 — the credibility gap.** *Uncertainty that survives review.* Calibrated probabilistic outputs + conformal prediction sets + an explicit cross-survey transfer benchmark (train SMASS/MITHNEOS → test Gaia DR3 labeled overlap) would instantly exceed the field's validation standard. Cheap to add, high defensibility value, and directly responsive to the documented 30-point transfer gap.

**Tier 3 — the physics gap.** *Heterogeneous multimodal fusion with native missing-data handling.* The multimodal dividend is proven (+7.8 pts, Ge 2024; albedo decisive in Mahlke/Pentikäinen), but no architecture yet ingests an arbitrary subset of {VIS spectrum, NIR spectrum, 16-band Gaia spectrum, ugrizy colors, albedo, phase-curve G, polarimetry} with attention-based masking rather than imputation. Mahlke's MCFA does missing-data linearly; nobody has done it deep. Polarimetry fusion is entirely virgin territory.

**Tier 4 — the timing gap.** *Be first on new data.* No classifier exists for SPHEREx (which uniquely restores NIR coverage) or real LSST photometry; Gaia DR4 (Dec 2026) will 7× the spectral sample. An architecture published with demonstrated readiness for these streams (via the missing-data machinery of Tier 3) gets cited by every downstream application paper.

**Tier 5 — framing choices that strengthen a paper.** Treat slope/phase/weathering as explicit nuisance parameters (invariance or augmentation grounded in Sanchez/Tinaut-Ruano systematics) rather than trusting slope as a feature; output both Bus-DeMeo and Mahlke labels from one latent space (sidesteps the two-taxonomy schism); consider a hierarchical output (complex → class) matching the demonstrated recoverability hierarchy; report balanced accuracy and per-class metrics at *type* level, never only collapsed-class accuracy.

**A candidate synthesis** (our working direction, to be refined): a spectral foundation-style encoder pretrained via masked reconstruction on all public unlabeled asteroid spectra/colors (Gaia DR3 + SDSS + MOVIS), with modality-agnostic tokenization so VIS/NIR/photometry/albedo/phase enter as maskable tokens; fine-tuned hierarchically on Mahlke + Bus-DeMeo labels with class-balanced few-shot objectives; wrapped in conformal calibration and evaluated on an explicit cross-survey transfer protocol. Each component is individually precedented (in astronomy at large) and collectively absent (in this field) — the definition of a strong novelty claim that is still defensible.

---

## 8. Known Correction List (for citation hygiene in the eventual paper)

Verified during this review; propagating any of these errors would be embarrassing at referee stage:

1. The compositional-map paper is **DeMeo & Carry 2014, Nature 505, 629** — not 2013 (the 2013 paper is Icarus 226, 723).
2. "Penttilä et al. 2020" is the DPS abstract; the refereed paper is **2021, A&A 649, A46**.
3. There is **no standalone DeMeo 2022 IR-only taxonomy paper**; DeMeo et al. 2022 (Icarus 380, 114971) is the meteorite-connection study; IR-only classification is Binzel et al. 2019 + the SMASS tool.
4. "Testing the Bus-DeMeo Taxonomy Using Meteorite Spectra" is **Burbine, Khanani, Kumawat, Hussain, Wallace & Dyar 2024, PSJ 5** (some secondary sources misattribute the first author).
5. Delbo et al. 2026 is **Frontiers in Astronomy and Space Sciences vol. 13**, article 1774478 (last-author surname appears as both "Milton" and "Minton" across versions — check the version of record).
6. Mahlke et al. 2022 defines **17 classes in 3 complexes**; enumerate the full class list from their Table before citing (our extraction confirmed 17/3 but captured only 16 names directly).
7. Gaia DR4 timing (Dec 2026) and LSST yield predictions are from announcements/secondary sources — reconfirm at submission time.

---

## 9. Core Reference List

**Taxonomy foundations:** Tholen 1984 (PhD thesis, U. Arizona) · Zellner, Tholen & Tedesco 1985, Icarus 61, 355 · Bus & Binzel 2002, Icarus 158, 106 & 146 · DeMeo, Binzel, Slivan & Bus 2009, Icarus 202, 160 · Binzel et al. 2019, Icarus 324, 41 · Mahlke, Carry & Mattei 2022, A&A 665, A26 (arXiv:2203.11229)

**Datasets/surveys:** Lazzaro et al. 2004, Icarus 172, 179 (S3OS2) · Ivezić et al. 2001, AJ 122, 2749 & Parker et al. 2008, Icarus 198, 138 (SDSS MOC) · DeMeo & Carry 2013, Icarus 226, 723; 2014, Nature 505, 629 · Galluccio et al. 2023, A&A 674, A35 (Gaia DR3) · Tinaut-Ruano et al. 2023, A&A 669, L14 · Masiero et al. 2011, ApJ 741, 68 · Alí-Lagoa et al. 2018, A&A 612, A85 · Popescu et al. 2016, A&A 591, A115; 2018, A&A 617, A12 (MOVIS) · Marsset et al. 2020, ApJS 247, 73; 2022, AJ 163, 165 · Berthier et al. 2023 (SsODNet) · Popescu et al. 2012, A&A 544, A130 (M4AST)

**ML classification:** Howell, Merényi & Lebofsky 1994, JGR 99, 10847 · Penttilä et al. 2021, A&A 649, A46 · Penttilä et al. 2022, Front. Astron. Space Sci. 9, 816268 · Klimczak et al. 2021, Front. Astron. Space Sci. 8, 767885; 2022, A&A 667, A10; 2023, AJ 166, 230 · Luo et al. 2023, AJ 167, 13 · Sergeyev & Carry 2021, A&A 652, A59 · Sergeyev et al. 2022, A&A 658, A109 · Erasmus et al. 2017, AJ 154, 162; 2018, ApJS 237, 19; 2019, ApJS 242, 15 · Roh et al. 2022, A&A 664, A51 · Colazo et al. 2022, A&A 666, A77 · Choi et al. 2023, PSJ 4, 49 · Ge et al. 2024, A&A 692, A100; 2025, ApJS (10.3847/1538-4365/adefe1) · Tang et al. 2025, AJ 169, 201 (arXiv:2502.16458) · Korda et al. 2023, A&A 669, A101 · Tinaut-Ruano et al. 2026, A&A 711, A167 · Delbo et al. 2026, Front. Astron. Space Sci. 13, 1774478 (arXiv:2602.22816) · Pentikäinen et al. 2026, A&A 707, A132 · Sullivan 2023, LJMU PhD thesis (DOI 10.24377/LJMU.t.00021526)

**Meteorite/mineralogy side:** DeMeo et al. 2022, Icarus 380, 114971 · Dyar et al. 2023, Icarus, 115718 · Burbine et al. 2024, PSJ 5 (10.3847/PSJ/ad57b6) · Sanchez et al. 2012, Icarus (arXiv:1205.0248)

**Adjacent/enabling:** Lin et al. 2026, arXiv:2604.23840 (TNO calibrated Bayesian spectra) · Lim et al. 2026, arXiv:2602.08037 (SOLO/SPHEREx) · Ivezić et al. 2022, Icarus (simulated SPHEREx asteroid spectra)
