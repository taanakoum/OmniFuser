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
    pre_attn: bool = False
    als_mode: str = "time"
    modality_dropout: float = 0.0
    pretrained: bool = False
    freeze_bn: bool = False

    def __post_init__(self) -> None:
        if self.input_len < 1 or self.horizon < 1:
            raise ValueError("input_len and horizon must be positive")
        if self.d_model < 1 or self.force_channels < 1:
            raise ValueError("model width and force channel count must be positive")
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
        if not 0.0 <= self.modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")
        if self.pretrained and self.image_encoder != "efficientnet_b0":
            raise ValueError("pretrained weights are only supported for efficientnet_b0")
        if self.fusion_rank < 1 or self.num_landmarks < 1:
            raise ValueError("fusion_rank and num_landmarks must be positive")
        if self.num_fusion_layers < 1:
            raise ValueError("num_fusion_layers must be positive")
        if self.image_encoder not in {"efficientnet_b0", "compact"}:
            raise ValueError("image_encoder must be 'efficientnet_b0' or 'compact'")
