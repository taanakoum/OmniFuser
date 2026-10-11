# OmniFuser alignment review of the VS Code repo (revision 5: after your modified code, the Blocks 6-8 check of the C2F layer, the Blocks 9-11 check of recursion, heads, objective and scores, and the Block 12 check of the front end and the whole model)

Date: 2026-10-11. Revision 1 was written from IMPLEMENTATION_AUDIT.md and README.md only. Revision 2 adds a line-by-line read of the code you sent
(`cli.py`, `config.py`, `data.py`, `models.py`, `train.py`, `evaluation.py`, `cross_validation.py`, `scripts/train_local.py`, `scripts/train_lightning.py`) and a patch for the open items (`omnifuser_alignment_patch.zip`, see its PATCH_NOTES.md).
Checked against the paper text (2511.01320v1, Sec. IV-VI, Eq. 1-23) and Figs. 4-6.

**What was and was not checked.** I read all nine files and ran your data loader on the real `forces_xyz_raw.mat` and the real label CSVs (the photos were replaced by links to three real photos, so image content did not enter any number).
I did not run any PyTorch code, yours or mine: PyTorch cannot be installed where I work. The patch's PyTorch parts (models, training loop, resume, cross-validation reuse) were re-read line by line and have 8 tests that you need to run (`python tests/test_alignment_patch.py`).
Statements about the paper were checked against the PDF.

## 0. Status of every finding against your modified code

| # | Finding (revision 1) | In your modified code | Patch |
|---|---|---|---|
| A1 | time step is not 1 ms | **decided (rev 4)**: your diagnostics put the paper's P = 96 level at 1 s steps; main arm `--pool 1000`, ablation `--pool 300` | `--pool Q`, `scripts/task_diagnostics.py` |
| A2 | future label = label of the shown photo | **decided (rev 4)**: `--image-lag 1` main arm, 0 as the paper-literal second arm (lag 0 persistence 91-99.99 %, lag 1 77-79 %) | `--image-lag 1`; `baselines.persistence_class_accuracy` in every fold JSON |
| A3 | horizons 12...96 | open (P = 24 only); rev 4: `paper_horizons` now come with per-horizon baselines, `scripts/compare_with_paper.py` | `test.paper_horizons` (leads 1..h, h = 12, 24, ..., 96); train with `--horizon 96` |
| B1 | self-attention before the split | open: `self_r`/`self_i` + LayerNorm in every C2F layer | `pre_attn` (default off) |
| B2 | ALS softmax over time | open | `als_mode` = `time` / `k_sum` / `k_colnorm` |
| B3 | heavy `lead_conditioned` head; Eq. 21 drops t | open; rev 4 recommends `direct` as the aligned arm (closest to Eq. 21-22), `lead_conditioned` as comparison | `prediction_head="direct"` |
| B4 | modality dropout missing | open | `modality_dropout`, `drop_modality=` at inference |
| B5 | shared or separate C2F weights per round | a separate C2F layer per round (Fig. 4, Layer 1..n); Eq. 20 writes one G, so rev 4 adds the other reading | `share_c2f` (`--share-c2f`), default off |
| B6 | Eq. 5 gives Z = P + cS | unchanged (same as paper) | none |
| B7 | private path P_hat = MHA(P) (Sec. V-C step 2) | **new in revision 3**: your layer computes LayerNorm(P + dropout(MHA(P))) | `private_attn` = `residual_ln` (default, unchanged) / `plain` (printed text) |
| D1 | Eq. 3 is a no-op (U = T + S = H) | **checked (rev 5)**: with `decomp_mode="literal"` the low-rank factors cannot change the output (< 1e-12 on your block); default `branch` (Fig. 5 paths) makes the decomposition matter | `--decomp-mode {branch,literal}` |
| D2 | MRTE blocks in cascade or parallel | **checked (rev 5)**: cascade 96 -> 32 -> 11 -> 4 -> 2; Eq. 2 literally reads H^r in every block, which with s_l = 3 would repeat one resolution; no flag for the parallel reading | none |
| D3 | meaning of c, b, p (Eq. 1) | **checked (rev 5)**: run number, block number, 16-bin position inside the pass; c is a clock inside each tool | none (period-table ablation not available) |
| D4 | front end numerically equal to the paper's equations | **checked (rev 5)**: embedding, RTD, MRTE, GAT, image branch and the whole forward agree with a NumPy reference to 1e-13 (float64, stand-in for torch) | `tests/test_block12_fullmodel.py` |
| D5 | architecture sizes (L, s, r, GAT layers / heads, n, k) only editable in code | **new in rev 5** | `--rtd-layers --rtd-stride --rtd-rank --gat-layers --gat-heads --fusion-layers --landmarks --fusion-rank --dropout` |
| C1 | batch size 32 | open (2 local, 16 CV) | defaults 32, `--accumulation-steps` |
| C2 | pretrained EfficientNet | open (`weights=None`) | `pretrained`, `freeze_bn` |
| C3 | last epoch tested | open | `--select-best` (default on) |
| C4 | no resume | open | per-epoch resume file; finished CV folds are reused |
| E | duplicate force records | **fixed in your code**: exact-equality dedup over all blocks within a tool/run (keeps the larger flank wear, ties to the lower block): 378 records, the same count I measured | none |
| new | pass trimming (`_nonastreda_span_bounds`, window 20, reference = median of first 5 %, threshold 0.5) | removes about 1.6 % of samples (309,140 vs 314,088 timeline windows at stride 120) | none; default on |

## 1. Task and time scale

### A1. The paper's time step is not a 1 kHz sample
Naive forecasts of the real force file at 1 kHz (T = 96, P = 24, z-scored with global mean/std, untrimmed): predict 0 gives 1.002, repeat the last value 0.017, repeat the last 15 samples (66.7 Hz period) 0.004.
The paper's Table IV (Mudestreda) lists 0.47-0.70 at horizon 12 and 0.90-1.37 at horizon 96 for nearly all models, OmniFuser included (0.467 / 0.902). At 1 ms a trivial predictor beats the paper's best model by a factor of 27 to 117.

Measured with your loader (trim on, dedup on; non-overlapping windows, T = 96; z-scored with all records):

| pool | step | windows (P = 12 / 96) | last-value MSE | labels change in horizon | persistence acc lag 0 | persistence acc lag 1 |
|---|---|---|---|---|---|---|
| 1 | 1 ms | 193,212 / 163,477 | 0.018 / 0.036 | 0.00 % / 0.02 % | 1.0000 / 0.9999 | 0.794 / 0.789 |
| 10 | 10 ms | 147,440 / 38,626 | 0.029 / 0.085 | 0.03 % / 0.20 % | 0.9999 / 0.9990 | 0.792 / 0.794 |
| 100 | 0.1 s | 30,815 / 3,848 | 0.093 / 0.260 | 0.23 % / 1.98 % | 0.9987 / 0.9909 | 0.794 / 0.793 |
| 300 | 0.3 s | 10,205 / 1,271 | 0.165 / 0.319 | 0.69 % / 6.14 % | 0.9960 / 0.9671 | 0.792 / 0.791 |
| 1000 | 1 s | 2,990 / 369 | 0.215 / 0.777 | 2.51 % / 20.60 % | 0.9867 / 0.9116 | 0.788 / 0.772 |

Revision 1 said that around 1 s the naive baselines come near the paper's band. That holds for P = 96 (0.78 against 0.90-1.37) but not for P = 12 (0.22 against 0.47-0.70). So the step is probably 1 s or coarser. This is an inference from baselines and figure axes (Figs. 1-3 and 10 use an "Iteration" axis of about 20 steps; Fig. 10 spans 24 h); the paper states neither the step nor the pooling (mean, RMS and energy are all possible).
**Change:** `--pool Q` (block mean per record before windowing); calibrate Q so that last-value and linear baselines fall in the paper's Table IV/V band, and report Q. The window count falls to a few hundred to tens of thousands per run.

### A2. The "future" class is the label of the photo that is in the input
With the photo of the sample's own pass (paper wording: "replicate each image across all sensor readings within the same pass"), 99.995-99.996 % of 1 ms windows have all target labels equal to the label of the shown photo, so accuracy measures reading the label off the input and the GAT sees T identical nodes (uniform attention). This is a faithful reading of the paper, not a code error; it makes the metric uninformative about prognosis at fine steps.
With the photo of the previous pass the persistence share is 79.5 % (lag 1), i.e. about 20 % of windows need real prognosis.
**Change:** keep `image_lag=0` as the paper-literal arm, add `image_lag=1` as the causal arm, and report the persistence baseline next to accuracy.

### A3. Horizons
The paper reports P in {12, 24, ..., 96}; the repo evaluates P = 24 only. Train with `--horizon 96`; `test.paper_horizons` scores leads 1..h for each h.

## 2. Architecture

| # | Item | Paper | Your code | Verdict | Patch |
|---|---|---|---|---|---|
| B1 | Attention before the split | Sec. V-C: Z goes straight into S = Z W', P (Eq. 5); Figs. 4 and 6 show Information Separation first | MHA + LayerNorm on both full streams before the split, every layer | **Divergence** | `pre_attn=False` |
| B2 | ALS weights (Eq. 10) | alpha_t = softmax(f(S_t)) in R^k: softmax over the k landmarks per token; L = alpha^T S | softmax over time per landmark | Interpretation that departs from the printed equation; stable | `als_mode` |
| B3 | Heads (Eq. 21-22) | y_t = f_reg(Z, t) = W_r z_T + b_r; the right side drops t | `paper_literal`, `lead_conditioned` | Extension, documented | `direct` |
| B4 | Modality dropout | Sec. VI-B4; rate not given | absent | Missing | `modality_dropout` |
| B5 | C2F weights across the n = 2 rounds | Eq. 20 writes one G; Fig. 4 stacks layers | separate layer per round | Defensible | none |
| B6 | Eq. 5 | Z = P + S claimed; the equation gives Z = P + cS, c = <Z,S>_F / (|S|_F^2 + eps) | literal Eq. 5 | Same as the paper | none |
| B7 | Private path | P_hat = MHA(P); Fig. 6 draws MHA without residual or normalisation | LayerNorm(P + dropout(MHA(P))) | Addition to the text | `private_attn` |

Aligned with the paper text: Eq. 1; two-layer MLP to d; Eq. 2 depthwise conv with kernel = stride; RTD `branch` mode = Fig. 5; image branch EfficientNet -> GAT over T nodes -> then position and period (Sec. V-B); GAT 4 x 4; landmark queries from the other modality (Eq. 11); Eq. 16-20; gate starting at 0.5; low-rank 2d -> r -> d; d = 32, L = 4, s = 3, r = 8, K = 4, n = 2, Adam 1e-4, dropout 0.1; metrics.
Open in the paper itself: Fig. 7 bars (k = 2, d = 16, heads = 16, r = 4) vs the text (k = 4, d = 32, heads = 4, r = 8); Eq. 17 W_l is 2d x 2d but the gate must be d wide; Eq. 3 cancels (U = T + S); `f_reg(Z, t)` takes t but Eq. 21 drops it.

Without column normalisation the printed Eq. 10 gives landmark norms that grow with T (19, 64, 267 for T = 24, 96, 384 in a toy check), so the logits S L^T / sqrt(d) saturate; that is why `k_colnorm` exists.

## 3. Training protocol (the paper is silent on epochs, scheduler, checkpoint choice and split)

Batch 32 (paper) with gradient accumulation for small GPUs; `pretrained` and `freeze_bn` (the paper does not say whether EfficientNet is pretrained); best-validation checkpoint; per-epoch resume; 8/1/1 leave-one-tool-out is your protocol, not the paper's.
Found while reading `train.py`: `float()` calls on GPU tensors in every step (a synchronisation per step) and JPEG decoding of 3100 x 1000 photos at 13-61 ms each with `num_workers=0`, which dominates step time. The patch keeps running sums on the device and decodes every photo once per fold (`ImageBank`, about 56 MB).

## 4. Choices to document, not to fix

Period-table mapping (run -> cycle, block -> batch, 16 position bins; sizes 128 / 8 / 16) is a guess, since what R and B mean physically is not in any file I have; untied period tables in the two branches; RTD cascade of four stride-3 blocks; gate kernel 3 and 2d -> d shape; loss = MSE + CE, unweighted (alignment and orthogonality terms are computed and logged but are not in the total); whole records kept (each record is one steady cut, verified); `discover_tools` requires exactly 10 tools.

## 5. Data

* "Mudestreda" in the paper cites reference [18], titled "Nonastreda multimodal dataset ...", by the same authors as the supplied data: very probably the same dataset (inference from the citation; the Data in Brief page was blocked for me). Scores remain non-comparable until A1-A3 are settled.
* MATWI is forecasting-only in the paper. Your repo does not cover MATWI.

## 6. What to run next

1. `python tests/test_alignment_patch.py` on your machine (expect 17 passed). Send me the output if anything fails.
2. `python scripts/task_diagnostics.py --data-dir Nonastreda_Data --pools 1 10 100 300 1000 --horizons 12 96 --out diag.json`; this needs no PyTorch and reproduces the table above on your machine.
3. One smoke fold with every new flag, then the experiment order in PATCH_NOTES.md section 4.

## 7. Blocks 6-8: second pass over the C2F layer (Eq. 5, 10-11, 16-19)

Full write-up: BLOCKS_6_8_NOTES.md in `omnifuser_blocks678.zip`. Method: a NumPy reference written from the paper (`tests/c2f_reference.py`), 25 tests (`tests/test_c2f_blocks678.py`) that load the weights of `CrossModalFusionLayer` into it, run on a NumPy stand-in for the torch calls the layer uses (24 passed, 1 skipped; largest difference 1.3e-12 in float64) and 11 deliberate breaks of the layer, all caught. Not yet run on real PyTorch.

* Eq. 5, Eq. 11, Eq. 16-19 are implemented as printed, with these choices: gate width d via a Linear after each branch (Eq. 17 prints 2d x 2d), depthwise kernel 3, biases in the low-rank map, `clamp_min(1e-8)` instead of `+ 1e-8` (relative effect 3e-12).
* Eq. 5 gives Z = P + cS (c != 1) and P is orthogonal to S only as a whole matrix, not per token; the logged `orthogonality_loss` is zero up to rounding.
* B7 (new): residual + LayerNorm on the private path is not in the text.
* Eq. 10: at initialisation (synthetic tokens, d = 32, k = 4) the printed form (`k_sum`) gives landmark norms of 5 / 20 / 80 times the token norm at T = 24 / 96 / 384 and attention entropy 0.01 (uniform: 1.386), i.e. a saturated softmax; `time` and `k_colnorm` give near-uniform attention (entropy 1.36-1.38). Ablate all three.
* Open in the paper: Fig. 9's landmark indices (0, 4, ..., 20) do not fit k = 4; Fig. 7 prefers k = 2.

## 8. Blocks 9-11: recursion and heads (Eq. 20-23), objective and protocol, scores (Tables IV-V)

Full write-up: BLOCKS_9_11_NOTES.md in `omnifuser_blocks911.zip`. Method as in Blocks 6-8: a NumPy reference from the paper (`tests/blocks9_11_reference.py`), 38 tests (`tests/test_blocks_9_11.py`; 35 run on a NumPy stand-in for torch and torch-free, 3 need autograd), 22 deliberate breaks all caught. Largest difference between `OmniFuser.forward` (front end replaced by fixed streams) and the reference over 18 head x round x sharing combinations: 1.2e-12. Not yet run on real PyTorch.

* Eq. 20 is implemented as printed (original streams + linear map of the previous round); one G or one layer per round is a choice: `share_c2f` (22,570 parameters per layer; sharing removes them from the second round).
* Eq. 21-22 as printed give a constant forecast over the leads (t dropped); `direct` writes the dependence on t out and is the closest reading; your default `lead_conditioned` is a shared nonlinear map of [z_T, e_t].
* Loss = MSE + CE (not stated in the paper); all stated hyper-parameters are the defaults (tested); clipping 1.0, epochs 5, best-validation selection are additions.
* Fixed: gradient accumulation divided the last shorter group by `accumulation_steps`; the CV summary reported MSE / MAE in force units while Tables IV-V are z-scored (new `paper_horizons_mean_std`, `scripts/compare_with_paper.py`).
* Added: weighted precision / recall / F1 next to macro (the paper does not say which), per-horizon baselines, absent classes counted as 0 in macro averages (documented).
* Decisions: `image_lag` 1 main (0 second), `pool` 1000 main (300 ablation), train with P = 96 and stride 24, score prefixes 12 ... 96.
* Caveat: your repo contains edits I have not seen (`pool_dropped_samples` in `inspect`; pooled diagnostics 2-5 % off mine for pool >= 100). Send changed files if the diff does not apply.

## 9. Block 12: front end (Eq. 1-4), the whole model, parameters, checkpoints

Full write-up: BLOCKS_12_NOTES.md in `omnifuser_blocks12.zip`. Method as before: a NumPy reference from the paper (`tests/frontend_reference.py`, independent of `omnifuser`), 38 tests (`tests/test_block12_fullmodel.py`; 34 run on a NumPy stand-in for torch, 4 need real autograd or torchvision), 38 deliberate breaks all caught. Largest difference between the real modules and the reference: embedding 4e-16, MRTE 3e-15, GAT 6e-16, whole model at T = 96 1.4e-13. Not yet run on real PyTorch.

* Eq. 1 and the image branch are implemented as printed; activation (GELU), positional-encoding formula, GAT score form and the residual / ELU / LayerNorm around each GAT layer are choices. c, b, p = run, block, 16-bin position inside the pass.
* Eq. 3 cancels when applied literally (U = H). Default `branch` follows Fig. 5. Experiment: `--decomp-mode literal`.
* MRTE is a cascade (lengths 96, 32, 11, 4, 2); Eq. 2 literally reads H^r in every block.
* Gathering photo features by index is the same computation as the paper's replication over the pass (tested).
* Parameters (d = 32, n = 2): temporal embedding 6,048; MRTE 53,600; image GAT part 9,472; C2F 22,570 per round; refinement 2,048; heads 3,046 (`lead_conditioned`, P = 24) / 19,008 (`direct`, P = 96). With EfficientNet-B0 the model has 4,167,894 / 4,183,856 parameters, 97 % in the CNN.
* Added (additive): the architecture flags above; `scripts/model_summary.py` (per-stage parameters and shapes, `--check` against the dimension count).
* Caveat unchanged: your repo contains edits I have not seen; its only change to an existing file is `omnifuser/cli.py` (`review/blocks12.diff`). The three autograd tests and the torchvision count test are yours to run.
