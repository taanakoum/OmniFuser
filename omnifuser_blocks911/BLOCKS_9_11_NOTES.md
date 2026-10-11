# Blocks 9-11: recursive refinement and heads, training objective and protocol, scoring

Checked against the paper's Sec. V-C (Eq. 20), V-D (Eq. 21-23), Sec. VI-A (setup, metrics) and Tables IV-V, and against your `OmniFuser.forward`, `multitask_loss`, `train.py`, `evaluation.py` and `cross_validation.py`.
Builds on the alignment patch and the Blocks 6-8 bundle; no training default changed.

## 0. Decisions you asked me to take

| Question | Decision | Why (from your `diag` output) |
|---|---|---|
| `image_lag` | **1 for the main arm, 0 as the paper-literal second arm** | At lag 0 the label of the shown photo is the target label for 99.99 % of 1 ms windows and still 91 % at 1 s steps with P = 96, so "predict the label of the photo you see" scores 91-99.99 % and the paper's 82-93 % accuracy would be *below* that trivial baseline at P = 12 and P = 96, the two horizons your diagnostics cover. At lag 1 the same baseline scores 77-79 %, and the paper's numbers sit above it. Lag 1 is also the causal reading if, as the paper states for MATWI (assumed here for Nonastreda), the photo is taken after the pass: the photo of the *same* pass does not exist while that pass is being cut. |
| Label column | `image_label` (as you said) | |
| Time step (`--pool`) | **1000 (1 s) main, 300 as an ablation** | Last-value MSE at lag-independent 1 s steps: 0.226 (P = 12), 0.807 (P = 96), the closest to the paper's level at P = 96 (0.90). At 0.3 s it is 0.170 / 0.328, at 0.1 s 0.095 / 0.265. No pool reproduces the paper's whole MSE-versus-horizon curve (its P = 12 value, 0.467, is about twice our persistence value at 1 s), so judge runs by the gap to the baselines, not by the third decimal. |
| Training horizon | **P = 96**, report the prefixes 12 ... 96 (`paper_horizons`) | A model trained with P = 24 cannot be scored at 36 ... 96. The paper's MSE is not monotone in h (0.917 at 84, 0.902 at 96), which suggests one model per horizon; that protocol costs 8x and is the second step. |
| Training stride | **`--stride 24`** (overlapping windows) | At 1 s steps and P = 96 your diagnostics (stride P = 96) count 369 windows over all tools; the default training stride T + P = 192 would give about half of that (about 190). Stride 24 gives about 37.1 M / (1000 x 24) = 1,500 (my arithmetic from your sample counts, not a measured number). Evaluation windows keep stride max(stride, P) = 96. |
| Head | **`direct`** as the paper-aligned arm, `lead_conditioned` (your default) as the comparison, `paper_literal` only to show the failure | See G3. |

## 1. Install and run

1. Commit your state. Copy `omnifuser/`, `tests/`, `scripts/` from this zip over your repo (Windows line endings). Changed: `config.py`, `models.py`, `cli.py`, `evaluation.py`, `train.py`, `cross_validation.py`. New: `tests/blocks9_11_reference.py`, `tests/test_blocks_9_11.py`, `scripts/compare_with_paper.py`. Diff against the Blocks 6-8 state: `review/blocks911.diff`. **If you changed any of the six files after the earlier patches** (your pooled diagnostics differ from mine by 2-5 % for pool >= 100 and your `inspect` output has a field I never wrote, `pool_dropped_samples`, so I know your repo contains edits I have not seen), do not copy them; run `git apply --ignore-whitespace review/blocks911.diff`, and send me the files that fail to apply.
2. `python tests/test_blocks_9_11.py` (or `python -m pytest tests/test_blocks_9_11.py`). Expected with PyTorch: **38 passed, 0 skipped, 0 failed**.
3. Earlier suites must still pass: `python tests/test_alignment_patch.py` (17) and `python tests/test_c2f_blocks678.py` (25).

## 2. What was run and what was not

| Part | Status |
|---|---|
| NumPy reference for Eq. 20-23, loss, metrics (`tests/blocks9_11_reference.py`) written from the paper's text | **Run.** |
| Your real `OmniFuser.forward` (fusion layers, refinement maps, three heads, modality masking) against that reference, front end replaced by fixed streams so only Eq. 20-23 are tested | **Run on a NumPy stand-in for the torch calls** (float64, eval): 18 combinations of head x rounds (1, 2, 3) x `share_c2f` plus 12 layer-option sets plus T = 96 at the paper's d, k, r, n. Largest difference 1.2e-12. **Not run on real PyTorch.** |
| `multitask_loss`, `classification_report`, `cumulative_horizon_report`, `regression_report`, `baseline_report`, ROC AUC with ties, absent classes, CV summary, `compare_with_paper.py` | **Run** (these use NumPy only; one test uses the CV module and needs a torch import). |
| Windows, leave-one-tool-out folds, normaliser fitted on training tools only, target strictly after the input | **Run** on synthetic Nonastreda-shaped folders (torch-free). |
| 3 tests that need autograd: gradients reach every parameter (3 heads x shared / separate), gradient accumulation equals one larger batch including a shorter last group, a tiny model overfits three samples | **Not run here (no autograd).** The overfit test was replicated in NumPy with hand-written gradients and Adam: final loss 2-8 % of the initial one for 6 seeds, threshold in the test 15 %. |
| Are the tests sensitive? | I broke your code on purpose in 22 ways (round 2 without the skip, wrong refine map, no recursion, head on first / mean token, direct head reshaped wrongly, lead code first, classification head reading the force head, drop_modality swapped, modality dropout dropping both, loss = MSE + 0.5 CE, alignment loss in the total, L1 force loss, CE times P, macro precision from recall, uniform weights, AUC without ties, cumulative MSE at one lead, accuracy over all leads, two baseline errors, accumulation divisor = N): every one made at least one test fail. |
| Front end (Eq. 1-4, MRTE/RTD, image branch) | **Not part of Blocks 9-11.** Block 12 compares it with the paper; no numerical parity has been done for it. |

## 3. Paper -> code map

| Paper | Equation | Your code | Reference | Match |
|---|---|---|---|---|
| Z^(1) = G(Z^r, Z^i) | Eq. 20 | first iteration of the loop in `OmniFuser.forward` | `refine` | yes |
| Z^(k) = G(Z^r + P^r Z^(k-1), Z^i + P^i Z^(k-1)) | Eq. 20 | `original_sensor + refine_sensor[k-1](fused)`, same for the image stream; `refine_*` = Linear(d, d, no bias) | `refine` | yes; P^r, P^i are "lightweight projection layers" in the paper, shape and bias not given |
| One G in every round | Eq. 20 | separate `CrossModalFusionLayer` per round (default), one shared layer with `share_c2f` | `refine(layers=[g]*n)` | **choice**, see G2 |
| y_t = W_r z_T + b_r | Eq. 21 | `paper_literal` (printed, constant over t), `direct` (one W_r^(t), b_r^(t) per lead), `lead_conditioned` | `head_*` | see G3 |
| o_t = f_cls(z_T) in R^3 | Eq. 22 | same three heads | `head_*` | see G3 |
| s_t = argmax softmax(o_t) | Eq. 23 | `argmax` of the softmax probabilities in `evaluation.py` | `cumulative_classification` | yes |
| Loss | not stated | `multitask_loss` = MSE(force) + CE(class), weights 1 : 1; alignment and orthogonality only logged | `multitask_loss` | **choice**, see H1 |
| d = 32, L = 4, s = 3, r = 8, GAT 4 x 4, k = 4, n = 2, Adam 1e-4, batch 32, dropout 0.1, alpha_0 = 0.5 | Sec. VI-A3 | `ModelConfig` defaults, `train_fold` defaults, `gate_mix` = 0 | AST + defaults test | yes |
| MSE, MAE (z-scored), Acc / Pre / Rec / F1, horizons 12 ... 96 | Sec. VI-A2, Tables IV-V | `cumulative_horizon_report`, `classification_report` | `cumulative_*`, `class_scores` | yes; macro vs weighted undecided, see K1 |

## 4. Findings

### Block 9 (Eq. 20-23)

**G1 Recursion is implemented as printed.** Round k reads the original streams plus a linear map of the previous round's whole (T x d) output. Tested for n = 1, 2, 3, with and without sharing. With zero refinement maps and a shared G every round repeats round one (reference and model); with separate layers they differ.

**G2 One G or one layer per round.** The paper writes one symbol G ("the full C2F module") for all rounds; Fig. 4 stacks Layer 1 ... n. Your default is one layer per round. New option `--share-c2f` ties them. In the test configuration (d = 32, k = 4, r = 8, n = 2) one layer has 22,570 parameters, the refinement maps 2 x 1,024; sharing removes 22,570 (49,658 -> 27,088 without the front end). I cannot tell which is meant; ablate. Changing `share_c2f` changes the stored `ModelConfig`, so old fold JSONs are rerun once.

**G3 The printed head cannot make a trajectory.** Eq. 21 is `f_reg(Z, t) = W_r z_T + b_r` with W_r in R^{d x C}: t is an argument but does not appear on the right, so every lead gets the same vector. That contradicts Fig. 10's trajectories. `paper_literal` is exactly that (198 parameters); a numerical check in the tests shows the best constant forecast has MSE equal to the variance of the target over the leads. `direct` writes the dependence on t as one weight block per lead (linear in z_T; 19,008 parameters at P = 96) and is the closest reading of Eq. 21-22. Your default `lead_conditioned` shares a nonlinear map of [z_T, e_t] across leads (5,350 parameters at P = 96); it is a reasonable model but not what the equation says. Run `direct` and `lead_conditioned` against each other.

**G4 Only z_T is used.** The heads see one d-vector per sample (checked through the parameter shapes). z_T is the last row of the final C2F output; since the cross-modal and private paths mix all time steps, it summarises the window, but a mean over time would be a different model (the mutation "head reads the mean" is caught).

**G5 Modality dropout.** Paper Sec. VI-B4 adopts it, rate not given. Yours drops sensor with probability p/2 and image with p/2, never both; tested for the rate and for the exclusion; default off. `drop_modality` at inference is the same as feeding a zero stream (tested).

### Block 10 (objective and protocol)

**H1 Loss.** The paper names both tasks and gives no loss or weights. MSE on z-scored force plus cross-entropy over all (sample, lead) pairs, weights 1 : 1, both means over their elements, so the value does not change with P. At initialisation MSE is about 1 and CE about ln 3 = 1.10, so the weights are balanced by luck of scaling, not by design. Alignment and orthogonality terms are logged and do not enter the total (tested by setting them to huge values).

**H2 Hyper-parameters.** Every value the paper states is the default (tested). Present but not in the paper: gradient clipping at 1.0, best-validation-epoch selection, 5 epochs (a smoke setting, use more), no scheduler, no weight decay (tested: Adam with `lr` only).

**H3 Gradient accumulation (fixed).** Each batch loss was divided by `accumulation_steps`, so with `n_batches` not divisible by `accumulation_steps` the last, shorter group was applied with a gradient that was too small (for example 7 batches, accumulation 3: the last step used one third of its gradient). Now each loss is divided by the size of its own group (`_accumulation_group_size`). For `accumulation_steps = 1` (default) nothing changes.

**H4 Protocol.** Folds: test tool, next tool in the cycle as validation, the other eight for training (tested for 10 tools); the normaliser is fitted on the training tools only (tested against `mean` / `std` of the stacked training forces and against leakage from the test tool). Targets start exactly after the input window and never overlap it (tested for both lags). The paper does not describe its split, so numbers are not comparable in detail.

**H5 Horizon and stride.** See section 0. `cumulative_horizon_report` silently skips horizons larger than the model's P; `compare_with_paper.py` prints only the horizons present.

### Block 11 (scores)

**K1 Macro or weighted.** The paper says "Accuracy, Precision, Recall, F1" without an averaging rule. `classification_report` and `cumulative_horizon_report` now return both (`*_macro`, `*_weighted`); weighted recall equals accuracy by construction (tested). Use `--average` in the comparison script to switch.

**K2 Units.** Tables IV-V are z-scored. Your fold JSON had z-scored cumulative scores under `paper_horizons`, but the 10-fold summary (`cv_mean_std`, `out_of_fold`) reports MSE / MAE in original force units, which cannot be compared with the paper. New: `paper_horizons_mean_std` in the CV summary and `scripts/compare_with_paper.py`, which reads the fold JSONs.

**K3 Baselines per horizon.** `baseline_report` now also returns `by_horizon` (last-value MSE / MAE, zero prediction, persistence accuracy over leads 1..h), so each model score has its baseline at the same h. Top-level keys are unchanged.

**K4 Absent classes.** A class with no samples in a test tool has precision = recall = 0 and enters the macro average as 0 (and its AUC is `None`). Tested. If a fold's test tool lacks a class (check `per_class.support`), its macro scores are lowered by that rule.

**K5 AUC with ties** equals the rank-sum (Mann-Whitney) value with average ranks, also when probabilities are rounded to few distinct values (tested).

**K6 Cumulative MSE** over leads 1..h equals the mean of the per-lead MSEs of `regression_report.by_horizon_normalized` (tested).

## 5. Your code vs the paper, in one table

| Item | Paper | Your code | Verdict |
|---|---|---|---|
| Recursion Eq. 20 | residual re-injection of Z^(k-1) through P^r, P^i | same | matches |
| C2F weights across rounds | one G | one layer per round (default) | choice; `--share-c2f` |
| Heads Eq. 21-22 | W_r z_T + b_r, t dropped | `lead_conditioned` default; `direct`, `paper_literal` available | choice; use `direct` as the aligned arm |
| z_T only | yes | yes | matches |
| Loss | not stated | MSE + CE | choice |
| Hyper-parameters of Sec. VI-A3 | listed | all equal | matches |
| Epochs, scheduler, clipping, checkpoint rule | not stated | 5, none, 1.0, best validation | choices |
| Split | not stated | leave-one-tool-out, cyclic validation tool | choice |
| Metrics | MSE, MAE, Acc, Pre, Rec, F1 | all, macro and weighted | matches; averaging undecided |
| Table IV-V units | z-scored | `paper_horizons` z-scored; CV summary was not | fixed |
| Image timing | "replicate each image across the readings within the same pass; updated images in subsequent passes" | `image_lag` 0 (own pass) or 1 (previous pass) | lag 0 is the literal reading; lag 1 chosen as main (section 0) |
| Gradient accumulation | not stated | divisor error in the last group | fixed |

## 6. Run plan (same fold, seed, pool, lag, stride within each comparison)

Fold T10, 5 epochs first (it takes minutes), then more epochs once the numbers move. Replace the paths; `--data-dir` is your Nonastreda folder.

```
python -m omnifuser.cli train --data-dir Nonastreda_Data --test-tool T10 --output-dir runs/a_lag1_pool1000_direct ^
  --pool 1000 --image-lag 1 --horizon 96 --stride 24 --prediction-head direct --epochs 30 --device cuda
python scripts/compare_with_paper.py runs/a_lag1_pool1000_direct
```
Then change one thing at a time: `--image-lag 0`; `--pool 300`; `--prediction-head lead_conditioned` and `paper_literal`; `--share-c2f`; `--als-mode k_colnorm` / `k_sum`; `--private-attn plain`; `--pretrained --freeze-bn`. Read `last-value` and `persistence` next to every model number: a model that does not beat them has learned nothing beyond "the force and the label stay". When one configuration is chosen, run all ten folds with `scripts/train_lightning.py` (same flags) and then `compare_with_paper.py` on the output folder (`--by pooled_by_windows` gives the window-weighted mean).

## 7. Side effects

`ModelConfig` gained `share_c2f` earlier in this series; fold JSONs written before it no longer match in `run_cross_validation` and are rerun once. `baseline_report` has a new optional argument and a new key `by_horizon`; `_summarize_folds` tolerates older fold files that lack the weighted metrics. No default changed except the accumulation divisor in the last short group.
