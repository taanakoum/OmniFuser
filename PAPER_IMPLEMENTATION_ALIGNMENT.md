# OmniFuser paper-to-code alignment

**Paper reviewed:** `Literature/2511.01320v1.pdf` (arXiv:2511.01320v1).
**Code reviewed:** the maintained `omnifuser/` package, `scripts/` and
`tests/` (state of 2026-10-14).
**Scope:** Nonastreda only. MATWI remains deferred.

This document compares the paper with the code. It is organized as follows:

| Section | Contents |
|---|---|
| **Paper fidelity review** | The authoritative item-by-item verdicts, the fixes, behavioral checks, and the paper's own inconsistencies |
| **Equation and architecture map** | Where and how each equation is implemented |
| **Decision chronology** | When each choice was made |
| **Theorems** | What the code does and does not validate |
| **Training, data and metric decisions** | The protocol choices the paper leaves open |
| **Blocks 12 command-line integration** | The architecture flags |
| **Verification boundary** | What is and is not checked here |

The full module-level reference is [`IMPLEMENTATION_AUDIT.md`](IMPLEMENTATION_AUDIT.md).

The paper does not specify enough detail for a bit-for-bit reproduction.
Where an activation, preprocessing step or operator detail is missing from the
paper, the code's choice is labelled a choice, never an exact match.

## Paper fidelity review (2026-10-14)

This review checks every item that the paper states (or leaves open) against
the maintained code. Verdicts:

- **Match**: the code does what the paper prints.
- **Match + choice**: the paper leaves a detail open and the code fills it.
- **Deviation (flag)**: the default differs from the printed text, and a flag
  selects the printed reading.
- **Forced**: the printed text is internally inconsistent, so no reading
  satisfies all of it.
- **Not verifiable**: cannot be judged from code or this data.

### Architecture

| # | Paper item | Code (default) | Verdict |
|---|---|---|---|
| A1 | Eq. 1: two-layer value MLP to \(d\), fixed sine/cosine positions, learned cycle/batch/relative-position tables | `TemporalEmbedding`; GELU between layers; run → cycle, block → batch, 16 position bins → relative | Match + choice |
| A2 | Eq. 2: depthwise temporal conv, kernel = stride \(s_l\), then \(W^{(l)}\), zero padding | `RTDBlock.depthwise` + `mix` | Match |
| A3 | MRTE: RTD blocks stacked, \(Z^r=\sum_l \tilde U^{(l)}\) | Cascaded blocks (each downsamples the previous), outputs summed | Match + choice |
| A4 | Eq. 3: \(T=\hat H PQ\), \(S=\hat H-T\), \(U=T+S\) | `decomp_mode=branch`: separate trend/seasonal FFNs, then sum (Fig. 5) | Deviation (flag `--decomp-mode literal`). As printed, \(U=T+S=\hat H\), so the low-rank split does nothing. Fig. 5 draws one FFN per component. Fig. 13's re-LR-MA ablation changes the MSE, which also implies the split matters. |
| A5 | FFD restores length \(T_l\to T\) | Linear over time + channel MLP | Match + choice |
| A6 | Eq. 4: lightweight EfficientNet (Fig. 4 lists the B0 stages) | torchvision EfficientNet-B0, 128×384 input, random init in all main runs; `--pretrained` (ImageNet weights plus ImageNet input normalization) is an optional comparison run | Match + choice |
| A7 | GAT over the \(T\) frame features, fully connected temporal graph | `GraphAttentionLayer` ×4, 4 heads, residual + LayerNorm + ELU | Match + choice |
| A8 | Positional and periodic embeddings added after the GAT | `ImageBranch.forward` | Match |
| A9 | Eq. 5: \(S=ZW'\), Frobenius-orthogonal \(P\) | `_decompose` | Match |
| A10 | Text after Eq. 5: "\(Z=P+S\), \(P\perp S\)"; Eq. 7 defines \(S=\Pi(Z)\) | Eq. 5 gives \(Z=P+cS\) with \(c=\langle Z,S\rangle_F/\lVert S\rVert_F^2\) | Forced. `--exact-split` uses \(cS\) so that \(Z=P+S\) holds exactly (Eq. 7 reading); off by default. |
| A11 | No attention before the shared/private split | `pre_attn=False` | Match |
| A12 | Private path \(\hat P=\mathrm{MHA}(P)\) | `private_attn=plain` (changed today; was `residual_ln`) | Match (residual + LayerNorm remains a flag) |
| A13 | Eq. 10: \(\alpha_t=\mathrm{softmax}(f(S_t))\in\mathbb R^k\), \(L=\alpha^T S\) | `als_mode=k_colnorm` (changed today; was `time`): printed weights; each landmark divided by its weight mass | Deviation in scale only (flag `--als-mode k_sum`). The printed sum grows with \(T\) and saturates Eq. 11. |
| A14 | Eq. 11: cross messages use the *other* modality's landmarks, \(/\sqrt d\), \(V_L=\alpha^T S W_v\) | `CrossModalFusionLayer.forward` | Match |
| A15 | Eq. 16: concatenate cross and private features | `torch.cat((cross, private))` | Match |
| A16 | Eq. 17: \(W_l\in\mathbb R^{2d\times 2d}\), depthwise Conv1d, learnable \(\alpha\) initialized to 0.5 | global gate \(2d\to d\); depthwise conv (kernel 3) then \(2d\to d\); \(\alpha=\sigma(0)=0.5\) | Forced: Eq. 18 multiplies the gate with \(d\)-wide features, so the gate must be \(d\) wide |
| A17 | Eq. 18: \(U=G\odot H+(1-G)\odot\hat P\) | same | Match |
| A18 | Eq. 19: \(W_3\,\phi(W_2[U^r,U^i])\), \(2d\to r\to d\), GELU | `project` (with biases) | Match + choice |
| A19 | Eq. 20: \(n=2\) recursive rounds re-injecting \(Z^m+\mathcal P_m(\hat Z)\) | separate C2F weights per round (`--share-c2f` for one \(G\)) | Match + choice (one \(G\) in Eq. 20, "Layer 1..n" in Fig. 4) |
| A20 | Eqs. 21-22: \(\hat y_t=f_{reg}(\hat Z,t)=W_r\hat z_T+b_r\) | `direct`: lead-indexed \(W_r,b_r\) on \(\hat z_T\) | Deviation (flag `--prediction-head paper_literal`). The printed right-hand side drops \(t\), making all 96 leads identical. |
| A21 | Eq. 23: argmax of softmax | `classification_report` | Match |
| A22 | Modality dropout in training, masking at inference (Sec. VI-B4) | `modality_dropout`; every fold evaluates sensor-/image-masked inputs | Match + choice (rate not given) |

### Training, data, and evaluation

| # | Paper item | Code / protocol | Verdict |
|---|---|---|---|
| B1 | Adam, lr \(10^{-4}\), batch 32, dropout 0.1 | same | Match |
| B2 | \(d=32\), \(L=4\), \(s_l=3\), \(r=8\), GAT 4 layers × 4 heads, \(k=4\), \(n=2\) | `ModelConfig` defaults (test `test_defaults_select_the_printed_paper_readings`) | Match |
| B3 | Loss weighting | MSE + cross-entropy, 1:1 | Match + choice (not stated) |
| B4 | Epochs, checkpoint selection | 30 epochs, best validation loss | Match + choice (not stated) |
| B5 | Metrics: MSE, MAE; Accuracy, Precision, Recall, F1 | all, plus weighted variants, AUC, RMSE, per lead/channel | Match |
| B6 | Horizons 12-96 | one \(P=96\) model, cumulative leads 1..h | Match + choice (paper may train one model per horizon) |
| B7 | Image replicated over every sensor reading of the **same** pass | protocol `--image-lag 0` (same pass's photo); `--image-lag 1` (previous pass's photo) is an experimental run | Match. Caveat: copying the shown photo's label scores 0.92-0.99 cumulative accuracy at 1 s steps, above the paper's own OmniFuser (EDA.md §7), so accuracy partly measures reading the label off the input. The lag-1 run quantifies this. |
| B8 | Input length \(T\), time step, normalization, data split | \(T=96\); 1 s mean pooling; train-tool z-score; 10-fold leave-one-tool-out | Not verifiable (paper is silent); choices recorded in the EDA |
| B9 | Fig. 13 ablations | 9 variants (`--ablation`) | Match + choice (replacement operators unspecified) |
| B10 | Figs. 7-8 sensitivity | 8 axes in `run_experiment_suite.py --study sensitivity` | Match |
| B11 | Comparative baselines (Tables IV-V) | not implemented; persistence/last-value baselines reported | Not implemented |
| B12 | ~\(10^{11}\) FLOPs per iteration | measured: forward \(2.46\times10^{10}\) per batch of 32 windows (EfficientNet is 98 %), so about \(7.4\times10^{10}\) forward+backward with one photo per window and \(1.5\times10^{11}\) with two | Match. `torch.utils.flop_counter` overcounts grouped-conv backward by about the group count, so its raw \(10^{12}\) is an artifact. |
| B13 | 7.26 GB memory | needs a GPU | Not verifiable here |

**Tally:**

| Verdict | Count | Items |
|---|---|---|
| Match or Match + choice | 27 | All items not listed below |
| Deviation (flag) | 2 | A4, A20 |
| Deviation in scale only | 1 | A13 |
| Forced | 2 | A10, A16 |
| Not verifiable | 2 | B8, B13 |
| Not implemented | 1 | B11 |

Every deviation has a flag that restores the printed reading.

### Fixes made in this review

- **`private_attn` default → `plain`** (A12): \(\hat P=\mathrm{MHA}(P)\) as
  printed.
- **`als_mode` default → `k_colnorm`** (A13): keeps the printed per-token
  softmax; previously normalized over time.
- **`exact_split` option** (A10): selects the Eq. 7 reading \(Z=P+S\).
- **`prediction_head` default `direct` everywhere** (A20): the CLI and
  `train_lightning.py` defaulted to `lead_conditioned` with horizon 24, while
  the suite used `direct`/96. All entry points now use `direct`, horizon 96.
- **`ImageFeatureEncoder` rejects unknown encoder names**: it previously fell
  back to the compact CNN without an error.
- **w/o-ALS uses fixed segment-mean landmarks**: the old version averaged
  every landmark over all time steps, which made them identical. Results
  produced before this fix are not comparable.
- **The hybrid gate is no longer computed in w/o-HG.**

Folds trained before these default changes have a different `config` and are
retrained automatically, never reused.

### Behavioral checks

These checks confirm that the code behaves as the paper describes. They are
not a reproduction of the paper's scores.

- **Compute.** One batch of 32 windows with EfficientNet-B0 at 128×384 costs
  \(2.46	imes10^{10}\) forward FLOPs. That is about \(7\)-\(15	imes10^{10}\)
  forward+backward, depending on whether a window shows one or two photos. The
  paper reports ~\(10^{11}\) (B12).
- **Learning capacity.** The model was trained for 400 Adam steps (lr
  \(10^{-3}\)) on 32 fixed real windows (1 s steps, \(T=P=96\), lag 1, compact
  encoder). Every configuration memorizes them:

  | Configuration | Force MSE (z-scored) | Class loss | Accuracy |
  |---|---|---|---|
  | Paper-reading defaults | 0.93 → 0.18 | 1.11 → 0.023 | 0.35 → 0.996 |
  | Previous defaults (`time`, `residual_ln`) | → 0.17 | → 0.023 | 0.999 |
  | `--exact-split` | → 0.19 | → 0.019 | 1.000 |

  The default changes do not reduce the model's capacity to fit.
- **A capped 8-epoch run is not a learning check.** 64 windows at batch 16 is
  only 32 Adam steps at lr \(10^{-4}\), and the loss barely moves. Use the
  uncapped pilot in the runbook to judge training.
- **Missing-modality masking.** A masked modality's input provably cannot
  change the predictions
  (`tests/test_model.py::test_masked_modality_input_cannot_influence_predictions`).

### Paper inconsistencies (not fixable in code)

- **Eq. 3 cancels:** \(U=T+S=\hat H\).
- **Eq. 5 vs. \(Z=P+S\):** Eq. 5 gives \(Z=P+cS\), not \(Z=P+S\).
- **Gate width:** Eq. 17's \(2d\times2d\) gate cannot multiply Eq. 18's
  \(d\)-wide features.
- **Eqs. 21-22:** the left side \(f_{reg}(\hat Z,t)\) takes \(t\); the right
  side drops it.
- **The symbol \(r\) has three meanings:** RTD rank (Eq. 3), the fusion
  bottleneck (Eq. 19), and the refinement-step index (Eq. 20).
- **\(k\):** the text says \(k=2\) gives the lowest error in Fig. 7, yet the
  configuration uses \(k=4\).
- **Fig. 13's base MSE (0.5749) matches no Table IV figure:**
  - Mudestreda mean: 0.709
  - MATWI mean: 0.102
  - Mean of the two datasets: 0.405
- **Table IV disagrees with itself:** OmniFuser's MATWI MSE at \(h=84\) is
  0.141 in the upper block and 0.142 in the lower block.
- **Table V disagrees with itself:** OmniFuser's precision at \(h=12\) is
  0.890 in the upper block and 0.889 in the lower block.
- **Dataset identity:** the text calls the dataset "Mudestreda", but its
  reference [18] is the *Nonastreda* data paper.

## Equation and architecture map

| Paper location | Implementation | Verdict and exactness boundary |
|---|---|---|
| Eq. 1, temporal embedding | `TemporalEmbedding` in [`omnifuser/models.py`](omnifuser/models.py): force-value MLP, sinusoidal position encoding, and three learned metadata tables. | The additive structure is implemented. The standard \(10000^{-2i/d}\) sine/cosine frequencies and GELU between the two value-MLP layers are choices because the paper says predefined frequencies/two linear layers but does not specify those formulas/activation. Table sizes and metadata meanings are also Nonastreda choices; see below. |
| Eq. 2, resolution-wise depthwise temporal convolution | `RTDBlock` and `MultiResolutionTemporalExtractor` in [`omnifuser/models.py`](omnifuser/models.py). | The grouped depthwise Conv1d, kernel=stride, channel projection, and zero-padding to a complete final window follow the stated operator. Padding convention for lengths not divisible by stride is an implementation choice. The multiblock cascade is a choice; Eq. 2's notation can also be read as parallel blocks. |
| Eq. 3, trend and seasonal terms | Low-rank `down @ P @ Q` and residual subtraction in `RTDBlock`. | The factorization and subtraction are represented. **The printed \(U=T+S\) cancels algebraically** because \(S=H-T\). Default `branch` applies separate feed-forward layers to the terms, following Fig. 5; `literal` exposes the equation's identity behavior. The paper does not resolve this contradiction. |
| Text after Eq. 3, restore the temporal length | `time_up` followed by GELU and a channel feed-forward in `RTDBlock`. | The output length is restored, but a learned linear map over the time dimension plus the channel MLP is a choice: the paper does not specify the internals of the independent restoration feed-forward. |
| Eq. 4 and Sec. V-B, image representation | `ImageFeatureEncoder` and `ImageBranch` in [`omnifuser/models.py`](omnifuser/models.py). | EfficientNet is implemented with torchvision EfficientNet-B0 by default. B0, image resize to 128x384, random rather than pretrained initialization, and the projection from 1280 features to \(d\) are decisions because the paper calls the backbone lightweight EfficientNet without fixing all these choices. `--pretrained` opts into ImageNet weights and then normalizes the [0, 1] pixels with the ImageNet mean and standard deviation, which those weights expect; random initialization leaves the pixels unchanged. |
| Sec. V-B, image temporal graph attention | `GraphAttentionLayer` and `ImageBranch` in [`omnifuser/models.py`](omnifuser/models.py). | A fully connected temporal graph and learned attention are implemented. The additive source/target score, LeakyReLU slope, residual connection, ELU, LayerNorm, and projection details are not fully specified in the paper and are choices. The stated layer/head counts are defaults. |
| Sec. VI-A, image-to-pass association | `make_tool_timelines`, `WindowDataset`, and `collate_windows` in [`omnifuser/data.py`](omnifuser/data.py); image gathering in `ImageBranch`. | A pass image is represented at every force sample in its pass and selected for each timestep. The implementation indexes and gathers each unique photo; it does not duplicate image files or materialize one image tensor per force row. `image_lag=0` is the paper's same-pass association; `image_lag=1` (the protocol) shows the preceding pass's photo. With lag 0, copying the photo's label already scores 0.92-0.99 cumulative accuracy at 1 s steps (`EDA.md` §7). The protocol nevertheless uses lag 0, the paper's association, and runs lag 1 as an experiment that measures this effect. Acquisition timestamps are absent, so photo timing relative to a pass is not verified. |
| Eq. 5 and Sec. V-C, shared/private information separation | `_decompose` in `CrossModalFusionLayer` in [`omnifuser/models.py`](omnifuser/models.py). | The shared projection and Frobenius-orthogonal private residual follow the equation, including the projection across the complete sequence-feature matrix. The stabilizing denominator clamp is a numerical detail. Eq. 5 gives \(Z=P+cS\), not the stated \(Z=P+S\). `--exact-split` uses the projection \(cS\) as the shared part, so \(Z=P+S\) holds exactly (the Eq. 7 reading). It is off by default. |
| Eq. 10, adaptive landmark weighting | `_landmarks` in `CrossModalFusionLayer`. | Default `als_mode=k_colnorm` (since 2026-10-14): the printed per-token softmax over the \(k\) landmarks, with each landmark column renormalized to a weighted mean. The renormalization changes only landmark magnitude, not direction. Without it, the printed \(L=\alpha^T S\) is a sum over time whose norm grows with \(T\) and saturates Eq. 11's softmax. `k_sum` is the unnormalized printed form; `time` (the previous default) normalizes over time instead, which departs from the printed per-token softmax. |
| Eq. 11, bidirectional cross-modal landmark attention | `CrossModalFusionLayer.forward` in [`omnifuser/models.py`](omnifuser/models.py). | Each modality attends over projected landmarks/values from the other modality. The selected attention scaling and its tensor normalization follow the implemented reading; no separate learned cross-attention block is silently substituted. |
| Eqs. 16-19, hybrid local/global gate and low-rank fusion | `_hybrid_gate` and `CrossModalFusionLayer` in [`omnifuser/models.py`](omnifuser/models.py). | The global/local gated mixture and low-rank bottleneck are implemented. The gate width \(2d\to d\) is forced: Eq. 17's \(2d\times2d\) cannot multiply Eq. 18's \(d\)-wide features. The depthwise kernel size 3 and the MLP biases are choices. The mixture \(\alpha=\sigma(0)=0.5\) matches the paper's stated initialization. |
| Eq. 20, progressive C2F recursion | `OmniFuser.forward` and `--share-c2f` in [`omnifuser/models.py`](omnifuser/models.py) and [`omnifuser/cli.py`](omnifuser/cli.py). | The next round re-injects original streams plus learned projections of the prior fused stream. Separate weights are the default, matching the stacked layers in the architecture figure; `--share-c2f` tests the single-\(G\) equation reading. |
| Eqs. 21-23, prediction heads | `OmniFuser.forward` in [`omnifuser/models.py`]. | Default `direct` (since 2026-10-14 in every entry point): one \(W_r^{(t)}, b_r^{(t)}\) per lead applied to \(\hat z_T\) only. This is Eq. 21 with the lead argument the notation \(f_{reg}(\hat Z, t)\) carries. `paper_literal` drops \(t\) as the right-hand side is printed and repeats one output over all leads. `lead_conditioned` adds a learned lead code. |

## Decision chronology

The code locations and reasons are in the matrices above. Their recorded
selection points are:

- **Earlier implementation:** choices such as GELU, 16 relative-position
  bins, the RTD restoration MLP, EfficientNet-B0, GAT residual/normalization,
  and the unweighted MSE-plus-cross-entropy loss were already part of the
  implementation. The Block 12 review documented them as choices on
  **2026-10-11**; the supplied history does not establish a separate creation
  date for each one.
- **2026-10-11, Blocks 9-11 decisions:** Nonastreda's one-second pooling
  (`pool=1000`) and preceding-image lag (`image_lag=1`) were selected for the
  main diagnostic arm after comparing persistence baselines. `pool=300`,
  lag 0, and the alternative forecast heads remain comparison arms.
  **Superseded 2026-10-18:** the protocol now uses the paper's lag 0, and
  lag 1 is the experimental arm. These
  selections and their numeric rationale are recorded in
  `Literature/omnifuser_part2_status_final.md` and
  `Literature/omnifuser_alignment_review_final.md` (local review notes); they
  were inferences from the available dataset diagnostics, not values specified
  in the paper. `EDA.md` §6-7 re-derives both from the real data.
- **2026-10-11, Blocks 12 integration:** the paper's parameter/sensitivity
  dimensions were exposed as CLI flags. The defaults remain the existing
  `ModelConfig` values so existing runs are not silently reconfigured; the
  flags make controlled ablations possible. Positive-dimension, dropout, and
  attention-divisibility guards were added at the same time so invalid
  experiments fail before model construction.
- **2026-10-14, paper fidelity review:** the defaults were switched to the
  printed readings (`private_attn=plain`, `als_mode=k_colnorm`; `direct` head
  and horizon 96 at every entry point). Also added: `--exact-split`, the Fig. 13
  ablations, the Figs. 7-8 sensitivity mapping, missing-modality evaluation,
  and the EDA. Lightning AI commands live in
  [`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md); no GPU run has been performed
  yet.

## Theoretical statements versus executable code

Theorem 1 follows the paper's information-processing argument from a
deterministic shared projection. The code implements the projection and
Frobenius-orthogonal residual in Eq. 5, but does not estimate mutual
information or validate the theorem's information-theoretic conclusion on
data.

Theorem 2 (Eqs. 12-15) states an expected Frobenius-error bound for a
CUR-style approximation under a data-dependent landmark selection condition.
The model implements ALS-weighted landmarks and proxy cross-attention, but it
does not implement a CUR estimator, compute the expectation/bound, or test the
theorem's assumptions. The bound and its assumptions are therefore **not
validated by the implementation or its model tests**. The attention and
landmark shape/parity tests establish code behavior only; they are not a proof
or empirical verification of Theorem 2.

## Training, data, and metric decisions

| Item | Paper evidence | Current setting or options | When/why selected |
|---|---|---|---|
| Dataset identity | The paper reports Mudestreda; the supplied project data is Nonastreda. | Nonastreda only, loaded by [`omnifuser/data.py`](omnifuser/data.py). | Scope retained from the user's instruction to defer MATWI and train on Nonastreda. Paper scores are plausibility references, not like-for-like reproduction results. |
| Time step | No unambiguous pooling/aggregation interval is stated. | The `--pool` library default is 1; the protocol uses `--pool 1000` (1 s at the recorded 1 kHz). | `EDA.md` §6: at 1 s the last-value baseline scores z-scored MSE 0.58/0.65/0.86 at P=12/48/96, inside the paper's range (OmniFuser 0.467/0.686/0.902). At 1 ms it is 23-29× easier. An earlier diagnostic, which used separate windows per horizon, matched only P=96. This is an inference from difficulty matching, not a paper value. |
| Image timing | Same-pass image replication is described; acquisition time for Nonastreda is not confirmed. | The library default and the protocol use `--image-lag 0` (paper reading). `--image-lag 1` is an experimental runbook run (§14). | `EDA.md` §7: with lag 0, copying the shown photo's label scores 0.92-0.99 cumulative accuracy, above the paper's OmniFuser. Lag 1 lowers this to about 0.80, so the model must forecast; the experimental run measures the difference. |
| Label and period codes | Paper describes class labels and cycle/batch/relative-position context but does not define a direct Nonastreda mapping. | `image_label` by default; run number -> cycle table, block number -> batch table, 16-bin position-within-record -> relative table. | Selected from the provided dataset fields and prior inspection; class names and the physical semantics/cardinalities of paper \(R\)/\(B\) are not fully established. |
| Window protocol | Paper tables report forecast horizons 12 through 96; input length and split are not stated. | 10-fold leave-one-tool-out: test tool, the next tool cyclically for validation, 8 for training. Protocol: \(T=96\), \(P=96\), training stride 24, evaluation stride 96. That gives 1,127-1,213 training and 28-51 test windows per fold. | Tool grouping keeps every window of a test tool out of training. This is the requested protocol, not a paper-exact split. |
| Force preprocessing | The paper does not define this Nonastreda-specific cleanup. | Exact duplicate force arrays are consolidated by default (`--dedup-identical-forces True`; `False` keeps all 512 records; `--eval-dedup-identical-forces` sets the validation/test data separately, and `--study deduplication` compares the two settings on the same deduplicated test windows), selecting the maximum-flank-wear annotation; span trimming is on by default and can be disabled. Pooling is per record; incomplete trailing blocks are dropped/reported. | Chosen from supplied guide/data audit. The precise envelope/threshold and dedup policy are documented in [`IMPLEMENTATION_AUDIT.md`](IMPLEMENTATION_AUDIT.md); they are not paper formulas. |
| Loss | No complete scalar objective and weighting for the two tasks is specified. | `multitask_loss` uses force MSE + class cross-entropy with 1:1 weighting. Alignment/orthogonality diagnostics are recorded but not added to total loss. | Simple equal weighting was selected because the paper does not give coefficients. It is not a reproduction of an unstated objective. |
| Optimizer/training lifecycle | Adam and learning rate \(10^{-4}\) are paper defaults; number of epochs/checkpoint selection are not fully prescribed. | Adam at \(10^{-4}\); best validation-loss checkpoint by default; resumable epoch-boundary checkpoints; gradient clipping and optional accumulation. | Best-checkpoint selection/resume/clipping are reliability choices for finite compute, not claims about the authors' exact training run. |
| Experiment metrics | Classification and regression tasks are reported; details such as class averaging can be ambiguous. | Accuracy, macro and weighted precision/recall/F1, ROC/AUC; MAE/MSE/RMSE in normalized and physical force units, per channel and horizon; baselines and per-fold/out-of-fold summaries. | Both common averaging conventions are retained instead of guessing which table convention was used. |

## Blocks 12 command-line integration

The architecture settings in the supplied patch are now available through the
installed `omnifuser train` command and both training scripts because they
share `add_protocol_arguments`:

`--decomp-mode`, `--rtd-layers`, `--rtd-stride`, `--rtd-rank`,
`--gat-layers`, `--gat-heads`, `--fusion-layers`, `--landmarks`,
`--fusion-rank`, and `--dropout`.

Unset flags preserve the configuration defaults and are not forwarded as
overrides. Invalid positive dimensions, dropout outside `[0, 1)`, and GAT
head/model-width incompatibility are rejected at `ModelConfig` construction.
See [`omnifuser/config.py`](omnifuser/config.py) and
[`tests/test_block12_integration.py`](tests/test_block12_integration.py).

## Verification boundary

The focused integration tests and model-summary count check validate the
maintained code using real PyTorch in this workspace. The supplied
`omnifuser_blocks12/tests/test_block12_fullmodel.py` is not runnable unchanged
because it imports earlier reference/test assets absent from the supplied
Blocks 12/9-11 directories and the root test tree. The NumPy parity figures in
`omnifuser_blocks12/BLOCKS_12_NOTES.md` therefore remain reported patch-author
evidence, not independently re-run parity tests here.

The repository's capped CPU smoke tests do not validate CUDA throughput,
memory, or training quality. Before production training, confirm CUDA in the
Lightning AI Studio, run one uncapped fold using the intended EfficientNet
settings and persistent storage, inspect epoch time/peak memory and validation
metrics, and only then launch the ten-fold run. The executable commands are in
[`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md). The complete record of what was
verified (static checks, 127 tests, real-data runs of every
entry point and flag) is in `IMPLEMENTATION_AUDIT.md` §10.
