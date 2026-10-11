# OmniFuser paper-to-code alignment

**Paper reviewed:** `Literature/2511.01320v1.pdf` (arXiv:2511.01320v1).
**Code reviewed:** the maintained root `omnifuser/` package, `scripts/`,
`tests/`, README, implementation audit, and the supplied Blocks 12 notes and
diff.
**Scope:** Nonastreda only. MATWI remains deferred.

## How to read the verdicts

- **Implemented as specified** means the code follows the equation or
  experimental instruction that the paper actually states. It does not mean
  the full training run or reported benchmark has been reproduced.
- **Choice** marks a paper ambiguity that required an implementation decision.
- **Variant** is a runnable alternative to the paper-literal reading.
- **Dataset/protocol adaptation** identifies a choice imposed by the available
  Nonastreda files or the requested tool-held-out experiment.

The paper does not specify enough implementation details to claim a
bit-for-bit reproduction. The exact-match column below is deliberately narrow:
where activation functions, preprocessing, or operator details are absent
from the paper, they are not called exact.

## Equation and architecture map

| Paper location | Implementation | Verdict and exactness boundary |
|---|---|---|
| Eq. 1, temporal embedding | `TemporalEmbedding` in [`omnifuser/models.py`](omnifuser/models.py): force-value MLP, sinusoidal position encoding, and three learned metadata tables. | The additive structure is implemented. The standard \(10000^{-2i/d}\) sine/cosine frequencies and GELU between the two value-MLP layers are choices because the paper says predefined frequencies/two linear layers but does not specify those formulas/activation. Table sizes and metadata meanings are also Nonastreda choices; see below. |
| Eq. 2, resolution-wise depthwise temporal convolution | `RTDBlock` and `MultiResolutionTemporalExtractor` in [`omnifuser/models.py`](omnifuser/models.py). | The grouped depthwise Conv1d, kernel=stride, channel projection, and zero-padding to a complete final window follow the stated operator. Padding convention for lengths not divisible by stride is an implementation choice. The multiblock cascade is a choice; Eq. 2's notation can also be read as parallel blocks. |
| Eq. 3, trend and seasonal terms | Low-rank `down @ P @ Q` and residual subtraction in `RTDBlock`. | The factorization and subtraction are represented. **The printed \(U=T+S\) cancels algebraically** because \(S=H-T\). Default `branch` applies separate feed-forward layers to the terms, following Fig. 5; `literal` exposes the equation's identity behavior. The paper does not resolve this contradiction. |
| Text after Eq. 3, restore the temporal length | `time_up` followed by GELU and a channel feed-forward in `RTDBlock`. | The output length is restored, but a learned linear map over the time dimension plus the channel MLP is a choice: the paper does not specify the internals of the independent restoration feed-forward. |
| Eq. 4 and Sec. V-B, image representation | `ImageFeatureEncoder` and `ImageBranch` in [`omnifuser/models.py`](omnifuser/models.py). | EfficientNet is implemented with torchvision EfficientNet-B0 by default. B0, image resize to 128x384, random rather than pretrained initialization, and the projection from 1280 features to \(d\) are decisions because the paper calls the backbone lightweight EfficientNet without fixing all these choices. `--pretrained` opts into ImageNet weights. |
| Sec. V-B, image temporal graph attention | `GraphAttentionLayer` and `ImageBranch` in [`omnifuser/models.py`](omnifuser/models.py). | A fully connected temporal graph and learned attention are implemented. The additive source/target score, LeakyReLU slope, residual connection, ELU, LayerNorm, and projection details are not fully specified in the paper and are choices. The stated layer/head counts are defaults. |
| Sec. VI-A, image-to-pass association | `make_tool_timelines`, `WindowDataset`, and `collate_windows` in [`omnifuser/data.py`](omnifuser/data.py); image gathering in `ImageBranch`. | A pass image is represented at every force sample in its pass and selected for each timestep. The implementation indexes and gathers each unique photo; it does not duplicate image files or materialize one image tensor per force row. `image_lag=0` is the same-record association; `image_lag=1` is a preceding-record causal-timing ablation. Acquisition timestamps are absent, so photo timing relative to a pass is not verified. |
| Eq. 5 and Sec. V-C, shared/private information separation | `_decompose` in `CrossModalFusionLayer` in [`omnifuser/models.py`](omnifuser/models.py). | The shared projection and Frobenius-orthogonal private residual follow the equation, including the projection across the complete sequence-feature matrix. The stabilizing denominator clamp is a numerical detail. |
| Eq. 10, adaptive landmark weighting | `_landmarks` in `CrossModalFusionLayer`. | The default `als_mode=time` normalizes over time to form temporal weighted averages. The printed equation can instead be read as a softmax over landmarks (`k_sum`); `k_colnorm` is another normalized option. This is an explicit notation/scale ambiguity, not an exact claim. |
| Eq. 11, bidirectional cross-modal landmark attention | `CrossModalFusionLayer.forward` in [`omnifuser/models.py`](omnifuser/models.py). | Each modality attends over projected landmarks/values from the other modality. The selected attention scaling and its tensor normalization follow the implemented reading; no separate learned cross-attention block is silently substituted. |
| Eqs. 16-19, hybrid local/global gate and low-rank fusion | `_hybrid_gate` and `CrossModalFusionLayer` in [`omnifuser/models.py`](omnifuser/models.py). | The global/local gated mixture and low-rank bottleneck are implemented. Gate width, depthwise kernel size 3, sigmoid mixture initialized to 0.5, and MLP biases are choices where the paper's displayed dimensions do not fully constrain a runnable layer. |
| Eq. 20, progressive C2F recursion | `OmniFuser.forward` and `--share-c2f` in [`omnifuser/models.py`](omnifuser/models.py) and [`omnifuser/cli.py`](omnifuser/cli.py). | The next round re-injects original streams plus learned projections of the prior fused stream. Separate weights are the default, matching the stacked layers in the architecture figure; `--share-c2f` tests the single-\(G\) equation reading. |
| Eqs. 21-23, prediction heads | `OmniFuser.forward` in [`omnifuser/models.py`]. | `paper_literal` repeats the final-token output over the horizon, matching the printed head equations. `direct` predicts each lead with separate output weights; `lead_conditioned` uses a learned horizon code. Both are comparison variants because the plotted trajectories vary by lead while the printed head expression drops the lead argument. |

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
  lag 0, and the alternative forecast heads remain comparison arms. These
  selections and their numeric rationale are recorded in
  [`omnifuser_part2_status_final.md`](omnifuser_part2_status_final.md); they
  were inferences from the available dataset diagnostics, not values specified
  in the paper.
- **2026-10-11, Blocks 12 integration:** the paper's parameter/sensitivity
  dimensions were exposed as CLI flags. The defaults remain the existing
  `ModelConfig` values so existing runs are not silently reconfigured; the
  flags make controlled ablations possible. Positive-dimension, dropout, and
  attention-divisibility guards were added at the same time so invalid
  experiments fail before model construction.
- **Lightning AI preparation in this review:** the one-fold then ten-fold
  command sequence in `README.md` uses the previously selected comparison
  values, persistent output storage, and an explicit CUDA availability check.
  It is operational guidance, not evidence that a Lightning AI GPU has already
  been exercised.

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
| Time step | No unambiguous pooling/aggregation interval is stated. | `--pool` default is 1. Prior task diagnostics selected pool 1000 (one second at the recorded 1 kHz) as the main comparison and 300 as an ablation. | The one-second interval brought the 96-step persistence MSE near the paper's range; it did not match the 12-step range. It is an inference from baseline diagnostics, not a paper value. The flag is exposed but the chosen experiment value is not forced as a global default. |
| Image timing | Same-pass image replication is described; acquisition time for Nonastreda is not confirmed. | `--image-lag 0` is paper-literal; `--image-lag 1` is the causal timing ablation. | Lag 1 was selected as the main forecast comparison to avoid an image captured after its target pass; lag 0 remains the direct paper-alignment arm. |
| Label and period codes | Paper describes class labels and cycle/batch/relative-position context but does not define a direct Nonastreda mapping. | `image_label` by default; run number -> cycle table, block number -> batch table, 16-bin position-within-record -> relative table. | Selected from the provided dataset fields and prior inspection; class names and the physical semantics/cardinalities of paper \(R\)/\(B\) are not fully established. |
| Window protocol | Paper tables report forecast horizons 12 through 96; the exact 8/1/1 split is not specified as a universal protocol. | The CV implementation holds one tool out for test, the next numeric tool for validation, and trains on the remaining eight. Training horizon/stride are independently configurable; use 96/24 to score paper horizons. | Tool grouping prevents windows from the same tool appearing in train and test. This is the requested evaluation protocol, not a paper-exact split. |
| Force preprocessing | The paper does not define this Nonastreda-specific cleanup. | Exact duplicate force arrays are consolidated by default, selecting the maximum-flank-wear annotation; span trimming is on by default and can be disabled. Pooling is per record; incomplete trailing blocks are dropped/reported. | Chosen from supplied guide/data audit. The precise envelope/threshold and dedup policy are documented in [`IMPLEMENTATION_AUDIT.md`](IMPLEMENTATION_AUDIT.md); they are not paper formulas. |
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
[`README.md`](README.md).
