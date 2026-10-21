from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Model dimensions and explicit choices where the paper is underspecified."""

    d_model: int = 32
    input_len: int = 96
    horizon: int = 96
    force_channels: int = 3
    num_classes: int = 3
    num_rtd_layers: int = 4
    rtd_stride: int = 3
    rtd_rank: int = 8
    num_gat_layers: int = 4
    num_gat_heads: int = 4
    num_fusion_layers: int = 2
    num_landmarks: int = 4
    fusion_rank: int = 8
    # Fig. 5: separate trend/seasonal feed-forwards (printed Eq. 3 cancels to H).
    decomp_mode: str = "branch"
    # Eq. 21-22 read as lead-indexed W_r, b_r applied to z_T only.
    prediction_head: str = "direct"
    image_encoder: str = "efficientnet_b0"
    image_height: int = 128
    image_width: int = 384
    dropout: float = 0.1
    pre_attn: bool = False
    # Eq. 10 per-token softmax over k, then each landmark normalized to a weighted mean.
    als_mode: str = "k_colnorm"
    # Sec. V-C step 2: P_hat = MHA(P), no residual or normalization.
    private_attn: str = "plain"
    # Eq. 5 keeps S = Z W'; True uses the projection c*S so that Z = P + S exactly (Eq. 7).
    exact_split: bool = False
    share_c2f: bool = False
    modality_dropout: float = 0.0
    pretrained: bool = False
    freeze_bn: bool = False
    # Keep the pretrained backbone's weights (and BatchNorm statistics) fixed; only the
    # projection to d_model and everything after it train.
    freeze_image_encoder: bool = False
    ablation: str = "none"

    def __post_init__(self) -> None:
        if self.input_len < 1 or self.horizon < 1:
            raise ValueError("input_len and horizon must be positive")
        if self.d_model < 1 or self.force_channels < 1:
            raise ValueError("model width and force channel count must be positive")
        if self.num_rtd_layers < 1 or self.rtd_stride < 1 or self.rtd_rank < 1:
            raise ValueError("RTD layer count, stride, and rank must be positive")
        if self.num_gat_layers < 1 or self.num_gat_heads < 1:
            raise ValueError("GAT layer count and head count must be positive")
        if self.d_model % self.num_gat_heads:
            raise ValueError("d_model must be divisible by num_gat_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.decomp_mode not in {"branch", "literal"}:
            raise ValueError("decomp_mode must be 'branch' or 'literal'")
        if self.prediction_head not in {
            "paper_literal",
            "lead_conditioned",
            "direct",
        }:
            raise ValueError(
                "prediction_head must be 'paper_literal', 'lead_conditioned' or 'direct'"
            )
        if self.als_mode not in {"time", "k_sum", "k_colnorm"}:
            raise ValueError("als_mode must be 'time', 'k_sum' or 'k_colnorm'")
        if self.private_attn not in {"residual_ln", "plain"}:
            raise ValueError("private_attn must be 'residual_ln' or 'plain'")
        if not 0.0 <= self.modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")
        if self.pretrained and self.image_encoder != "efficientnet_b0":
            raise ValueError("pretrained weights are only supported for efficientnet_b0")
        if self.freeze_image_encoder and not self.pretrained:
            raise ValueError("freeze_image_encoder needs pretrained weights; frozen random weights learn nothing")
        if self.ablation not in {
            "none",
            "without_rr",
            "without_mrte",
            "without_rr_mrte",
            "concat_c2f",
            "without_c2f",
            "without_als",
            "without_hybrid_gate",
            "moving_average",
        }:
            raise ValueError(f"unsupported ablation: {self.ablation}")
        if self.fusion_rank < 1 or self.num_landmarks < 1:
            raise ValueError("fusion_rank and num_landmarks must be positive")
        if self.num_fusion_layers < 1:
            raise ValueError("num_fusion_layers must be positive")
        if self.image_encoder not in {"efficientnet_b0", "compact"}:
            raise ValueError("image_encoder must be 'efficientnet_b0' or 'compact'")
