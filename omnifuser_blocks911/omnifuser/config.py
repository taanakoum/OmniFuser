from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Model dimensions and explicit choices where the paper is underspecified."""

    d_model: int = 32
    input_len: int = 96
    horizon: int = 24
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
    decomp_mode: str = "branch"
    prediction_head: str = "lead_conditioned"
    image_encoder: str = "efficientnet_b0"
    image_height: int = 128
    image_width: int = 384
    dropout: float = 0.1
    # --- alignment switches (see omnifuser_alignment_review.md) -------------------------------------------
    pre_attn: bool = False  # False = paper (Fig. 4/6: Information Separation is the first step of C2F)
    als_mode: str = "time"  # "time" softmax over time | "k_sum" printed Eq. 10 | "k_colnorm" Eq. 10 / column mass
    modality_dropout: float = 0.0  # paper Sec. VI-B4 adopts modality dropout; rate not given
    pretrained: bool = False  # paper does not say whether EfficientNet is pretrained
    freeze_bn: bool = False  # keep BatchNorm in eval mode (useful with small batches / pretrained weights)
    # "residual_ln": P_hat = LayerNorm(P + MHA(P)) (previous behaviour); "plain": P_hat = MHA(P) (printed text, Sec. V-C step 2)
    private_attn: str = "residual_ln"
    # Eq. 20 uses one symbol G for every refinement round; Fig. 4 stacks Layer 1..n.  False: one C2F layer per round
    # (previous behaviour); True: the same C2F layer (same weights) in every round.
    share_c2f: bool = False

    def __post_init__(self) -> None:
        if self.input_len < 1 or self.horizon < 1:
            raise ValueError("input_len and horizon must be positive")
        if self.d_model < 1 or self.force_channels < 1:
            raise ValueError("model width and force channel count must be positive")
        if self.decomp_mode not in {"branch", "literal"}:
            raise ValueError("decomp_mode must be 'branch' or 'literal'")
        if self.prediction_head not in {"paper_literal", "lead_conditioned", "direct"}:
            raise ValueError(
                "prediction_head must be 'paper_literal', 'lead_conditioned' or 'direct'"
            )
        if self.als_mode not in {"time", "k_sum", "k_colnorm"}:
            raise ValueError("als_mode must be 'time', 'k_sum' or 'k_colnorm'")
        if self.private_attn not in {"residual_ln", "plain"}:
            raise ValueError("private_attn must be 'residual_ln' or 'plain'")
        if not 0.0 <= self.modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")
        if self.fusion_rank < 1 or self.num_landmarks < 1:
            raise ValueError("fusion_rank and num_landmarks must be positive")
        if self.num_fusion_layers < 1:
            raise ValueError("num_fusion_layers must be positive")
        if self.image_encoder not in {"efficientnet_b0", "compact"}:
            raise ValueError("image_encoder must be 'efficientnet_b0' or 'compact'")
