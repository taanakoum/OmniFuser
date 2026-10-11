# Block 12: the front end (Eq. 1-4), the whole model, parameters and checkpoints

Checked against the paper's Sec. V-A (temporal embedding, MRTE / RTD), V-B (image branch), VI-A (alignment of photos and force), VI-B3 (hyper-parameters, Figs. 7-8) and against your `TemporalEmbedding`, `RTDBlock`, `MultiResolutionTemporalExtractor`, `GraphAttentionLayer`, `ImageFeatureEncoder`, `ImageBranch` and `OmniFuser`.
Builds on the alignment patch and the Blocks 6-8 and 9-11 bundles. **No model code was changed**; the only edit to existing code is additive (command-line flags for the architecture sizes, section 5).

## 1. Install and run

1. Commit your state. Copy `tests/`, `scripts/` and `omnifuser/cli.py` from this zip over your repo (Windows line endings). New: `tests/frontend_reference.py`, `tests/test_block12_fullmodel.py`, `scripts/model_summary.py`. Changed: `omnifuser/cli.py` only. Diff against the Blocks 9-11 state: `review/blocks12.diff`. **If you changed `cli.py` yourself after the earlier patches** (I know your repo contains edits I have not seen), do not copy it; run `git apply --ignore-whitespace review/blocks12.diff` and send me the hunks that fail.
2. `python tests/test_block12_fullmodel.py` (or `python -m pytest tests/test_block12_fullmodel.py`). Expected with PyTorch and torchvision: **38 passed, 0 skipped, 0 failed**; without torchvision 37 passed, 1 skipped (`..._efficientnet_b0_backbone_has_the_torchvision_parameter_count`). Four tests need real autograd or torchvision and could not run where I worked (section 2).
3. `python scripts/model_summary.py --horizon 96 --prediction-head direct --check` builds the model with EfficientNet-B0, runs one random batch and prints parameters and output shape per stage. Expected last lines: `total 4,183,856` and `implied by the dimensions: 4,183,856 (match)`. With `--image-encoder compact --horizon 24` the total is 128,482.
4. Earlier suites must still pass: `python tests/test_alignment_patch.py` (17), `python tests/test_c2f_blocks678.py` (25), `python tests/test_blocks_9_11.py` (38).

## 2. What was run and what was not

| Part | Status |
|---|---|
| NumPy reference of Eq. 1-4 and of the whole forward pass (`tests/frontend_reference.py`), written from the paper's text without importing `omnifuser` | **Run.** Checked against hand values (positional encoding), an explicit-loop convolution, algebraic identities (Eq. 3 cancels when applied literally, rank of the trend, row-stochastic and permutation-equivariant attention), and the parameter count. |
| Your real modules against that reference: embedding (4 kinds of period codes, clipping, missing codes), one RTD block (T = 96, 24, 25, 26, 7; both decomposition modes), the whole MRTE (strides 1-4, 1-4 blocks, T = 10 ... 96), the compact CNN, a graph-attention layer and its attention matrix, the image branch (non-decreasing and arbitrary indices), and the **complete model** from raw force, photos, indices and period codes to the heads (3 heads x rounds 1-2 x shared / separate C2F, layer options, drop_modality, T = 96 at the paper's dimensions) | **Run on a NumPy stand-in for the torch calls** (float64, eval mode, your code unmodified). Largest differences: embedding 4e-16, MRTE (T = 96, L = 4) 3e-15, compact CNN 2e-16, GAT 6e-16, whole model at T = 96 between 4e-14 and 1.4e-13. **Not run on real PyTorch.** |
| Your batches (`WindowDataset` -> `collate_windows` on a synthetic Nonastreda folder, `image_lag` 1) through the model and the reference, plus the loss | **Run** on the stand-in; shapes and values agree. |
| Image indexing equals the paper's replication of the photo over the pass | **Run**: gathering per-photo features by index gives the same output as running the CNN on T replicated copies (1e-9). |
| Batch independence (a sample alone equals the same sample in a batch), determinism in eval mode, a photo no index points at is ignored, float32 within 2e-4 of the float64 reference, malformed inputs rejected, `state_dict` round trip (positional encodings are not stored, BatchNorm statistics are) | **Run** (the `torch.save` / `torch.load` leg only where those exist: it will run on your side). |
| Parameter counts: 16 configurations (heads, rounds, sharing, `pre_attn`, `private_attn`, T = 24 / 50 / 96, d = 16 / 32, RTD depth and stride 1-3, landmarks, ranks, GAT depth, `decomp_mode`) equal the numbers implied by the dimensions alone, per part and in total | **Run.** Paper settings with the compact CNN: 128,482. |
| `scripts/model_summary.py` (shapes via forward hooks, `--check`) | **Run** on the stand-in with the compact CNN; the EfficientNet line is for you. |
| Architecture flags `--decomp-mode --rtd-layers --rtd-stride --rtd-rank --gat-layers --gat-heads --fusion-layers --landmarks --fusion-rank --dropout`: unset flags change nothing (the stored `ModelConfig` stays the same), set flags reach `ModelConfig` and the built model, through `omnifuser train`, `train_local.py` and `train_lightning.py` | **Run** (with training stubbed out). |
| 3 tests that need autograd: gradients reach every parameter of the whole model (3 heads), dropout is active in train mode and not in eval mode, the whole model overfits three samples (Adam, 300 steps, loss below 40 % of the initial one) | **Not run here.** For the overfit test I replicated the head alone in NumPy with hand-written gradients and Adam on the z_T of this model: 23 % of the initial loss with the same settings (the three z_T are nearly parallel at initialisation, cosine 0.99, which is why it takes hundreds of steps); training the other weights as well should only help, but this is the test I am least sure of on your machine. |
| 1 test that needs torchvision: the `features` of `efficientnet_b0` have 4,007,548 parameters (5,288,548 minus the 1,281,000 of the 1000-class classifier, from memory of torchvision's numbers), the projection 40,992, and the whole model matches the dimension count | **Not run here.** If it fails on the number alone, the number is mine to correct, not a model fault. |
| Are the tests sensitive? | I broke your code on purpose in 38 ways and re-ran the file after each: window padded on the left or not padded, W^(l) dropped, seasonal sign, trend = full signal, trend and seasonal feed-forwards swapped, `decomp_mode` ignored, no GELU before the channel feed-forward, wider channel feed-forward, `time_up` to the wrong length, MRTE keeping only the last block / halving / adding a skip, positional encoding omitted or with base 1000 or sin / cos swapped, period tables swapped or dropped, value MLP without activation, index clamp removed, missing code = row 1, GAT with source / target swapped, softmax over the wrong axis, slope 0.01, no ELU / no residual / no LayerNorm in the image branch, image positional encoding omitted, always the first photo, wrong table for a code, index check removed, ReLU instead of SiLU in the CNN, codes swapped when calling the embedding or the image branch, MRTE skipped, image stream zeroed, streams exchanged. Every one made at least one test fail (several by raising an error rather than by a number). |

## 3. Paper -> code map (front end)

| Paper | Equation | Your code | Reference | Match |
|---|---|---|---|---|
| H^r_t = MLP(x_t) + PE[t] + TE_c(c_t) + TE_b(b_t) + TE_p(p_t) | Eq. 1 | `TemporalEmbedding` | `sensor_embedding`, `positional_encoding` | yes; activation and PE formula are choices (F1), the meaning of c, b, p is a mapping (F2) |
| Depthwise temporal convolution, kernel = stride = s_l, zero padding, then channel projection W^(l) | Eq. 2 | `RTDBlock.depthwise` (Conv1d, groups = d, no bias), right zero padding, `mix` (Linear, no bias) | `depthwise_strided` | yes; T_l = ceil(T / s_l) |
| T = H P Q, S = H - T, rank r | Eq. 3 | `low_rank_left`, `low_rank_right`, `seasonal = down - trend` | `rtd_block` | yes |
| U = T + S | Eq. 3 | `decomp_mode="literal"`; default `"branch"` feeds trend and seasonal through their own feed-forwards (Fig. 5) | `rtd_block(decomp_mode=...)` | **cancels as printed**, see F3 |
| U-tilde = FFD(U) in R^(T x d) | text after Eq. 3 | `time_up` (Linear over time T_l -> T), GELU, `channel_ffn` | `rtd_block` | **choice**, F4 |
| Z^r = sum_l U-tilde^(l) | text | `MultiResolutionTemporalExtractor.forward` | `mrte` | yes; each block reads the previous block's downsampled output, F5 |
| f_t = EffNet(I_t) | Eq. 4 | `ImageFeatureEncoder` (torchvision EfficientNet-B0 `features`, global average, Linear(1280, d)) | none for B0; the compact CNN has one | structure and size only, F8 |
| GAT over the T frames, fully connected, learned edge weights | Sec. V-B | `GraphAttentionLayer` x 4, each wrapped as LayerNorm(x + ELU(GAT(x))) | `gat_layer`, `image_branch` | yes; form of the score is a choice, F6 |
| one photo per pass, replicated over the pass; updated in later passes | Sec. VI-A | `collate_windows` + `image_indices`, `torch.gather` in `ImageBranch` | replication test | yes, F7; `image_lag` decides which pass |
| + positional and period embeddings on the image side | Sec. V-B | `ImageBranch.forward` | `image_branch` | yes |
| d = 32, L = 4, s = 3, r = 8, 4 GAT layers and heads | Sec. VI-B3 | `ModelConfig` | `test_b12_full_model_at_the_paper_dimensions...` | yes |

## 4. Findings

**F1 Eq. 1 is implemented as printed.** Two linear layers with a GELU between (the paper says "two stacked linear layers", the activation is not named), the standard sine / cosine positional encoding (the paper gives "predefined frequencies", no formula; the buffer is float32 and equals the float64 closed form to 2.3e-6, the rounding of the angle t f), three learnable tables of 128 / 8 / 16 rows, dropout. The positional-encoding buffer is not persistent, so checkpoints do not contain it and loading needs the same `input_len`.

**F2 What c, b, p are on Nonastreda.** The paper: spindle cycle, cutting batch, relative position within the cycle. Your `make_tool_timelines`: c = the run number, b = the block number (1 ... 4), p = the position inside the pass quantised to 16 bins. A missing code adds row 0 of the table (a learned constant, not "no embedding"); a code beyond the table is clipped to the last row. The tables have enough rows for the Nonastreda numbering (runs 1 ... 15 per tool in your `inspect` output, blocks 1 ... 4). Note that c is a monotone clock inside each tool: whatever the model learns from it is an allowed input of the paper's design, but it means accuracy can come from "which run is this" rather than from the force or the photo. The drop-modality test cannot tell; an ablation with the period tables zeroed would, and no flag for it exists yet (say so if you want one).

**F3 Eq. 3 cancels.** U = T + S = T + (H - T) = H, whatever P and Q are. Tested on your real block in literal mode: replacing P and Q by random matrices changes the MRTE output by less than 1e-12; in the default branch mode it changes by more than 1e-3. In literal mode the low-rank factors (2 d r = 512 parameters per block) and the two feed-forward branches are dead weights with zero gradient. Your default `branch` follows Fig. 5, where the two components are drawn as separate paths into a feed-forward block, and makes the decomposition matter. I cannot tell which the authors ran; compare `--decomp-mode literal` with the default. Tested for both modes against the reference; in branch mode the trend has rank exactly 8, the seasonal part has rank above 8, and trend + seasonal equals the downsampled signal.

**F4 "An independent feed-forward module that restores the temporal length to T."** Your version is a Linear over the time axis (T_l -> T, with bias), GELU, then a Linear-GELU-Linear over channels. Reasonable, not the only reading. It is the largest term of the temporal encoder: (T_l + 1) x T per block, 5,088 weights in total at T = 96.

**F5 Cascade, not parallel.** Eq. 2 writes the input of every block as H^r and gives each block its own stride s_l; with s_l = 3 for all l (Sec. VI-B3) that would be L blocks of the same resolution. Your encoder feeds each block the downsampled output of the previous one, so the lengths are 96 -> 32 -> 11 -> 4 -> 2 (the last two blocks see only 4 and 2 time steps), and the restored outputs are summed. The paper's phrase "in a residual manner" fits either. Both readings give the same shapes, so only an experiment or the authors can settle it; there is no flag for the parallel reading.

**F6 The graph attention.** Fully connected over the T frames, one shared projection d -> d, per-head source and target vectors, LeakyReLU(0.2), softmax over the neighbours, concatenated heads (Velickovic et al.). Around it your code adds a residual connection, an ELU and a LayerNorm per layer (not in the paper) and a Linear(1280 -> d) in front, whereas the paper's GAT outputs R^d directly. Parity with the reference holds for the layer (output and attention matrix) and for the whole branch. A standard GAT property worth knowing: when all pre-activation scores of a row are positive, LeakyReLU is the identity and the source term cancels in the softmax, so its gradient vanishes for that row.

**F7 Photos.** Gathering the per-photo feature by index equals running the CNN on a replicated stack (tested, 1e-9), so the CNN runs once per distinct photo of a window (one or two at lag 0 / 1) instead of T times. `image_lag` only changes which photo an index points to; the codes c, b, p always belong to the pass that owns the sample (documented in `make_tool_timelines`).

**F8 EfficientNet.** The paper says "lightweight EfficientNet"; B0 is the assumption. Random initialisation by default (`--pretrained` loads ImageNet weights, `--freeze-bn` keeps BatchNorm in eval). It has no NumPy reference here (too large to reimplement faithfully); the evidence for it is the structure and the parameter count (F9). Photos are resized to 128 x 384 by default.

**F9 Where the parameters are.** d = 32, T = 96, n = 2 rounds, compact CNN / EfficientNet-B0:

| Part | Parameters |
|---|---|
| Temporal embedding (Eq. 1) | 6,048 |
| MRTE (4 RTD blocks) | 53,600 |
| Image encoder | 9,128 (compact) / 4,048,540 (B0 features + Linear) |
| Image GAT (4 layers) + norms + position / period tables | 9,472 |
| C2F, one layer (two rounds = 45,140, or 22,570 with `--share-c2f`) | 22,570 |
| Refinement maps P^r, P^i | 2,048 |
| Heads at P = 24: `lead_conditioned` / `direct` / `paper_literal` | 3,046 / 4,752 / 198 |
| **Total with B0, `lead_conditioned`, P = 24** | **4,167,894** |
| **Total with B0, `direct`, P = 96** | **4,183,856** |

About 97 % of the model is the CNN backbone, the paper's modules together have about 120 thousand weights. A small data set (about 380 passes, ten tools) trains the backbone from random weights unless you use `--pretrained`.

**F10 The whole forward pass agrees with the reference** for every head, one and two rounds, shared or separate C2F, the layer options, `drop_modality`, with and without period codes, with one photo or several, at T = 24 and at T = 96 with the paper's dimensions (section 2). A sample's output does not depend on the other samples of the batch in eval mode, and a photo that no index points at has no effect.

## 5. Architecture flags (the one code change)

To make the paper's sensitivity study (Figs. 7-8: GAT heads 4 / 8 / 16, rank r 4 / 8 / 16 / 32, GAT layers 2-4, stride s 1-3, L 1-4, n 1-3) and the F3 comparison runnable from the command line and from Lightning without editing code, `omnifuser train`, `scripts/train_local.py` and `scripts/train_lightning.py` accept:

`--decomp-mode {branch,literal}  --rtd-layers L  --rtd-stride s  --rtd-rank r  --gat-layers N  --gat-heads H  --fusion-layers n  --landmarks k  --fusion-rank R  --dropout p`

Unset flags are not passed on, so the stored `ModelConfig` of existing runs does not change and no fold is rerun. `--gat-heads` sets the heads of every attention in the model (graph attention and the multi-head attentions inside C2F), and must divide `--d-model`.

## 6. Experiments for the front end (same fold, seed, pool, lag, stride within each comparison)

```
python -m omnifuser.cli train --data-dir Nonastreda_Data --test-tool T10 --output-dir runs/f_branch --pool 1000 --image-lag 1 ^
  --horizon 96 --stride 24 --prediction-head direct --epochs 30 --device cuda
python -m omnifuser.cli train ...same... --output-dir runs/f_literal --decomp-mode literal
python -m omnifuser.cli train ...same... --output-dir runs/f_pretrained --pretrained --freeze-bn
python -m omnifuser.cli train ...same... --output-dir runs/f_L2 --rtd-layers 2        (and 1, 3)
python -m omnifuser.cli train ...same... --output-dir runs/f_s2 --rtd-stride 2        (and 1)
python -m omnifuser.cli train ...same... --output-dir runs/f_r16 --rtd-rank 16        (and 4, 32)
```
Read every number next to the `last-value` and `persistence` baselines of `scripts/compare_with_paper.py`. A change of the front end that moves a score by less than the spread between seeds (run at least two seeds on the fold you tune on) is not a finding.

## 7. Side effects

`omnifuser/cli.py`: `add_protocol_arguments` gained the ten flags above, `protocol_kwargs` forwards only those that were given, and a module-level table `ARCHITECTURE_OPTIONS` maps flag to `ModelConfig` field. Nothing else in the package changed. The new files are test and diagnostic code only.
