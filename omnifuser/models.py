from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from omnifuser.config import ModelConfig


def sinusoidal_position_encoding(length: int, d_model: int) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    encoding = torch.zeros(length, d_model, dtype=torch.float32)
    angles = positions * frequencies
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding


def _metadata_code(
    table: nn.Embedding, values: Tensor | None, batch: int, steps: int, device: torch.device
) -> Tensor:
    if values is None:
        indices = torch.zeros((batch, steps), dtype=torch.long, device=device)
    else:
        indices = values.to(device=device, dtype=torch.long).clamp(0, table.num_embeddings - 1)
        if indices.ndim == 1:
            indices = indices[:, None].expand(-1, steps)
    return table(indices)


class TemporalEmbedding(nn.Module):
    position_encoding: Tensor  # registered buffer; declared for type checkers

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.value = nn.Sequential(
            nn.Linear(config.force_channels, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.register_buffer(
            "position_encoding",
            sinusoidal_position_encoding(config.input_len, d),
            persistent=False,
        )
        self.cycle = nn.Embedding(128, d)
        self.batch = nn.Embedding(8, d)
        self.relative = nn.Embedding(16, d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        force: Tensor,
        cycle: Tensor | None = None,
        batch_code: Tensor | None = None,
        relative: Tensor | None = None,
    ) -> Tensor:
        batch, steps, _ = force.shape
        if steps > self.position_encoding.shape[0]:
            raise ValueError(
                f"sequence length {steps} exceeds {self.position_encoding.shape[0]}"
            )
        return self.dropout(
            self.value(force)
            + self.position_encoding[:steps].to(dtype=force.dtype)[None]
            + _metadata_code(self.cycle, cycle, batch, steps, force.device)
            + _metadata_code(self.batch, batch_code, batch, steps, force.device)
            + _metadata_code(self.relative, relative, batch, steps, force.device)
        )


class RTDBlock(nn.Module):
    """One temporal resolution decomposition block."""

    def __init__(self, input_steps: int, config: ModelConfig) -> None:
        super().__init__()
        d, stride = config.d_model, config.rtd_stride
        output_steps = math.ceil(input_steps / stride)
        self.input_steps = input_steps
        self.stride = stride
        self.output_steps = output_steps
        self.depthwise = nn.Conv1d(
            d, d, kernel_size=stride, stride=stride, groups=d, bias=False
        )
        self.mix = nn.Linear(d, d, bias=False)
        self.use_moving_average = config.ablation == "moving_average"
        if self.use_moving_average:
            self.register_parameter("low_rank_left", None)
            self.register_parameter("low_rank_right", None)
        else:
            self.low_rank_left = nn.Parameter(torch.empty(d, config.rtd_rank))
            self.low_rank_right = nn.Parameter(torch.empty(config.rtd_rank, d))
            nn.init.xavier_uniform_(self.low_rank_left)
            nn.init.xavier_uniform_(self.low_rank_right)
        self.trend_ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.seasonal_ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.time_up = nn.Linear(output_steps, config.input_len)
        self.channel_ffn = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
        )
        self.decomp_mode = config.decomp_mode

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.shape[1] != self.input_steps:
            raise ValueError(f"expected {self.input_steps} steps, received {x.shape[1]}")
        pad = (-x.shape[1]) % self.stride
        down = x.transpose(1, 2)
        if pad:
            down = F.pad(down, (0, pad))
        down = self.mix(self.depthwise(down).transpose(1, 2))
        if self.use_moving_average:
            trend = F.avg_pool1d(
                down.transpose(1, 2),
                kernel_size=3,
                stride=1,
                padding=1,
                count_include_pad=False,
            ).transpose(1, 2)
        else:
            if self.low_rank_left is None or self.low_rank_right is None:
                raise RuntimeError("low-rank trend parameters are not initialized")
            trend = (down @ self.low_rank_left) @ self.low_rank_right
        seasonal = down - trend
        if self.decomp_mode == "literal":
            decomposed = trend + seasonal
        else:
            decomposed = self.trend_ffn(trend) + self.seasonal_ffn(seasonal)
        up = self.time_up(decomposed.transpose(1, 2)).transpose(1, 2)
        up = self.channel_ffn(F.gelu(up))
        return down, up


class MultiResolutionTemporalExtractor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        lengths = [config.input_len]
        for _ in range(config.num_rtd_layers):
            lengths.append(math.ceil(lengths[-1] / config.rtd_stride))
        self.layers = nn.ModuleList(
            RTDBlock(lengths[i], config) for i in range(config.num_rtd_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        total = torch.zeros_like(x)
        current = x
        for layer in self.layers:
            current, restored = layer(current)
            total = total + restored
        return total


class GraphAttentionLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by num_gat_heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.source_score = nn.Parameter(torch.empty(heads, self.head_dim))
        self.target_score = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.source_score)
        nn.init.xavier_uniform_(self.target_score)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch, steps, _ = x.shape
        values = self.projection(x).view(batch, steps, self.heads, self.head_dim)
        source = (values * self.source_score).sum(-1).permute(0, 2, 1)
        target = (values * self.target_score).sum(-1).permute(0, 2, 1)
        logits = F.leaky_relu(source.unsqueeze(-1) + target.unsqueeze(-2), 0.2)
        attention = torch.softmax(logits, dim=-1)
        output = torch.einsum("bhij,bjhd->bihd", self.dropout(attention), values)
        return output.reshape(batch, steps, -1), attention


class ImageFeatureEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        d_model: int,
        pretrained: bool = False,
        freeze_bn: bool = False,
        freeze_weights: bool = False,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        # Feature-map stacks with other than 3 channels pass a lightweight learnable
        # image generator first (VLT Eq. 8). It always trains, even with frozen weights.
        self.generator: nn.Module = (
            nn.Identity()
            if in_channels == 3
            else nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 3, 1),
            )
        )
        # Frozen weights only make sense with frozen BatchNorm statistics as well.
        self.freeze_bn = freeze_bn or freeze_weights
        # A frozen backbone is a fixed function: eval mode also turns off the stochastic
        # depth inside torchvision's EfficientNet blocks, so a photo's features never
        # change and can be computed once per fold (``train.FrozenFeatureBank``).
        self.freeze_weights = freeze_weights
        # Photos arrive already ImageNet-normalized (``data.IMAGE_MEAN``/``IMAGE_STD``),
        # which is what pretrained weights expect; no further scaling happens here.
        if name not in {"efficientnet_b0", "compact"}:
            raise ValueError(f"unknown image encoder: {name!r}")
        if name == "efficientnet_b0":
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

            backbone = efficientnet_b0(
                weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            input_width = 1280
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(24),
                nn.SiLU(),
                nn.Conv2d(24, 48, 3, stride=2, padding=1, groups=24, bias=False),
                nn.BatchNorm2d(48),
                nn.SiLU(),
                nn.Conv2d(48, 96, 1, bias=False),
                nn.BatchNorm2d(96),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.pool = nn.Identity()
            input_width = 96
        self.feature_width = input_width
        self.projection = nn.Linear(input_width, d_model)
        if freeze_weights:
            self.features.requires_grad_(False)

    def train(self, mode: bool = True) -> ImageFeatureEncoder:
        super().train(mode)
        if mode and self.freeze_weights:
            self.features.eval()
        elif mode and self.freeze_bn:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def backbone_features(self, images: Tensor) -> Tensor:
        """(N, C, H, W) images -> (N, feature_width) pooled backbone features."""
        return self.pool(self.features(self.generator(images))).flatten(1)

    def forward(self, images: Tensor) -> Tensor:
        """(B, N, C, H, W) images, or (B, N, feature_width) precomputed features."""
        if images.shape[1] == 0:
            return images.new_zeros((images.shape[0], 0, self.projection.out_features))
        if images.ndim == 3:
            if images.shape[-1] != self.feature_width:
                raise ValueError(
                    f"precomputed image features must have width {self.feature_width}"
                )
            return self.projection(images)
        features = self.backbone_features(images.flatten(0, 1))
        return self.projection(features).reshape(images.shape[0], images.shape[1], -1)


class ImageBranch(nn.Module):
    position_encoding: Tensor  # registered buffer; declared for type checkers

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = ImageFeatureEncoder(
            config.image_encoder,
            config.d_model,
            pretrained=config.pretrained,
            freeze_bn=config.freeze_bn,
            freeze_weights=config.freeze_image_encoder,
            in_channels=config.image_channels,
        )
        self.register_buffer(
            "position_encoding",
            sinusoidal_position_encoding(config.input_len, config.d_model),
            persistent=False,
        )
        self.cycle = nn.Embedding(128, config.d_model)
        self.batch = nn.Embedding(8, config.d_model)
        self.relative = nn.Embedding(16, config.d_model)
        # With image_gat=False (an experiment) there are no GAT layers: the projected
        # encoder features go to fusion with only the position and metadata codes added.
        self.layers = nn.ModuleList(
            GraphAttentionLayer(config.d_model, config.num_gat_heads, config.dropout)
            for _ in range(config.num_gat_layers if config.image_gat else 0)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(config.d_model) for _ in self.layers)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        images: Tensor,
        image_indices: Tensor,
        cycle: Tensor | None = None,
        batch_code: Tensor | None = None,
        relative: Tensor | None = None,
    ) -> Tensor:
        batch, steps = image_indices.shape
        encoded = self.encoder(images)
        if encoded.shape[1] == 0:
            raise ValueError("each sequence must reference at least one image")
        if image_indices.min() < 0 or image_indices.max() >= encoded.shape[1]:
            raise ValueError("image_indices contains an invalid image index")
        x = torch.gather(
            encoded, 1, image_indices.unsqueeze(-1).expand(-1, -1, encoded.shape[-1])
        )
        for layer, norm in zip(self.layers, self.norms):
            output, _ = layer(x)
            x = norm(x + self.dropout(F.elu(output)))
        return (
            x
            + self.position_encoding[:steps].to(dtype=x.dtype)[None]
            + _metadata_code(self.cycle, cycle, batch, steps, x.device)
            + _metadata_code(self.batch, batch_code, batch, steps, x.device)
            + _metadata_code(self.relative, relative, batch, steps, x.device)
        )


class CrossModalFusionLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.shared_r = nn.Linear(d, d, bias=False)
        self.shared_i = nn.Linear(d, d, bias=False)
        self.score_r = nn.Linear(d, config.num_landmarks, bias=False)
        self.score_i = nn.Linear(d, config.num_landmarks, bias=False)
        self.pre_attn = config.pre_attn
        if self.pre_attn:
            self.self_r = nn.MultiheadAttention(
                d, config.num_gat_heads, dropout=config.dropout, batch_first=True
            )
            self.self_i = nn.MultiheadAttention(
                d, config.num_gat_heads, dropout=config.dropout, batch_first=True
            )
            self.self_norm_r = nn.LayerNorm(d)
            self.self_norm_i = nn.LayerNorm(d)
        self.als_mode = config.als_mode
        self.value_r = nn.Linear(d, d, bias=False)
        self.value_i = nn.Linear(d, d, bias=False)
        self.private_self_r = nn.MultiheadAttention(d, config.num_gat_heads, dropout=config.dropout, batch_first=True)
        self.private_self_i = nn.MultiheadAttention(d, config.num_gat_heads, dropout=config.dropout, batch_first=True)
        self.private_attn = config.private_attn
        if self.private_attn == "residual_ln":
            self.private_norm_r = nn.LayerNorm(d)
            self.private_norm_i = nn.LayerNorm(d)
        self.global_gate_r = nn.Linear(2 * d, d)
        self.global_gate_i = nn.Linear(2 * d, d)
        self.local_depthwise_r = nn.Conv1d(2 * d, 2 * d, 3, padding=1, groups=2 * d)
        self.local_depthwise_i = nn.Conv1d(2 * d, 2 * d, 3, padding=1, groups=2 * d)
        self.local_gate_r = nn.Linear(2 * d, d)
        self.local_gate_i = nn.Linear(2 * d, d)
        self.gate_mix_r = nn.Parameter(torch.zeros(()))
        self.gate_mix_i = nn.Parameter(torch.zeros(()))
        self.project = nn.Sequential(
            nn.Linear(2 * d, config.fusion_rank),
            nn.GELU(),
            nn.Linear(config.fusion_rank, d),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.eps = 1e-8

    def _landmarks(
        self, x: Tensor, scorer: nn.Linear, value_projection: nn.Module
    ) -> tuple[Tensor, Tensor]:
        if self.config.ablation == "without_als":
            # Fixed, non-adaptive landmarks: means of k contiguous time segments.
            k = scorer.out_features
            landmarks = F.adaptive_avg_pool1d(x.transpose(1, 2), k).transpose(1, 2)
            landmark_values = F.adaptive_avg_pool1d(
                value_projection(x).transpose(1, 2), k
            ).transpose(1, 2)
            return landmarks, landmark_values
        logits = scorer(x)
        if self.als_mode == "time":
            weights = torch.softmax(logits, dim=1)
        else:
            weights = torch.softmax(logits, dim=-1)
            if self.als_mode == "k_colnorm":
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        landmarks = weights.transpose(1, 2) @ x
        landmark_values = weights.transpose(1, 2) @ value_projection(x)
        return landmarks, landmark_values

    def _decompose(
        self, x: Tensor, shared_projection: nn.Linear
    ) -> tuple[Tensor, Tensor]:
        shared = shared_projection(x)
        numerator = (x * shared).sum(dim=(1, 2), keepdim=True)
        denominator = shared.square().sum(dim=(1, 2), keepdim=True).clamp_min(self.eps)
        projection = numerator / denominator * shared
        private = x - projection
        if self.config.exact_split:
            return projection, private
        return shared, private

    @staticmethod
    def _hybrid_gate(
        features: Tensor,
        global_projection: nn.Linear,
        local_depthwise: nn.Conv1d,
        local_projection: nn.Linear,
        mix_logit: Tensor,
    ) -> Tensor:
        global_gate = torch.sigmoid(global_projection(features))
        local = local_depthwise(features.transpose(1, 2)).transpose(1, 2)
        local_gate = torch.sigmoid(local_projection(local))
        mix = torch.sigmoid(mix_logit)
        return mix * global_gate + (1.0 - mix) * local_gate

    def forward(self, r: Tensor, i: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.pre_attn:
            self_r, _ = self.self_r(r, r, r, need_weights=False)
            self_i, _ = self.self_i(i, i, i, need_weights=False)
            r = self.self_norm_r(r + self.dropout(self_r))
            i = self.self_norm_i(i + self.dropout(self_i))
        shared_r, private_r = self._decompose(r, self.shared_r)
        shared_i, private_i = self._decompose(i, self.shared_i)
        orthogonality_r = (shared_r * private_r).sum(dim=(1, 2))
        orthogonality_i = (shared_i * private_i).sum(dim=(1, 2))
        private_att_r, _ = self.private_self_r(
            private_r, private_r, private_r, need_weights=False
        )
        private_att_i, _ = self.private_self_i(
            private_i, private_i, private_i, need_weights=False
        )
        if self.private_attn == "residual_ln":
            private_r = self.private_norm_r(private_r + self.dropout(private_att_r))
            private_i = self.private_norm_i(private_i + self.dropout(private_att_i))
        else:
            private_r, private_i = private_att_r, private_att_i

        land_r, value_land_r = self._landmarks(shared_r, self.score_r, self.value_r)
        land_i, value_land_i = self._landmarks(shared_i, self.score_i, self.value_i)
        cross_r = torch.softmax(
            shared_r @ land_i.transpose(1, 2) / math.sqrt(shared_r.shape[-1]), dim=-1
        ) @ value_land_i
        cross_i = torch.softmax(
            shared_i @ land_r.transpose(1, 2) / math.sqrt(shared_i.shape[-1]), dim=-1
        ) @ value_land_r

        if self.config.ablation == "without_hybrid_gate":
            fused_r = 0.5 * (cross_r + private_r)
            fused_i = 0.5 * (cross_i + private_i)
        else:
            gate_r = self._hybrid_gate(
                torch.cat((cross_r, private_r), dim=-1),
                self.global_gate_r,
                self.local_depthwise_r,
                self.local_gate_r,
                self.gate_mix_r,
            )
            gate_i = self._hybrid_gate(
                torch.cat((cross_i, private_i), dim=-1),
                self.global_gate_i,
                self.local_depthwise_i,
                self.local_gate_i,
                self.gate_mix_i,
            )
            fused_r = gate_r * cross_r + (1 - gate_r) * private_r
            fused_i = gate_i * cross_i + (1 - gate_i) * private_i
        fused = self.project(torch.cat((fused_r, fused_i), dim=-1))

        norm_r = F.normalize(shared_r, dim=-1)
        norm_i = F.normalize(shared_i, dim=-1)
        shared_alignment = (norm_r - norm_i).square().mean()
        orthogonality = (
            orthogonality_r.square().mean() + orthogonality_i.square().mean()
        )
        return self.dropout(fused), shared_alignment, orthogonality


class ConcatenationFusionLayer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.project = nn.Linear(2 * d_model, d_model)

    def forward(self, sensor: Tensor, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        zero = sensor.new_zeros(())
        return self.project(torch.cat((sensor, image), dim=-1)), zero, zero


class OmniFuser(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.temporal_embedding = TemporalEmbedding(self.config)
        if self.config.ablation in {"without_mrte", "without_rr_mrte"}:
            self.temporal_encoder = nn.Identity()
        else:
            self.temporal_encoder = MultiResolutionTemporalExtractor(self.config)
        self.image_branch = ImageBranch(self.config)
        if self.config.ablation == "without_c2f":
            self.fusion_layers = nn.ModuleList()
        elif self.config.ablation in {"without_rr", "without_rr_mrte"}:
            self.fusion_layers = nn.ModuleList(
                [CrossModalFusionLayer(self.config)]
            )
        elif self.config.ablation == "concat_c2f":
            self.fusion_layers = nn.ModuleList(
                ConcatenationFusionLayer(self.config.d_model)
                for _ in range(self.config.num_fusion_layers)
            )
        elif self.config.share_c2f:
            shared_layer = CrossModalFusionLayer(self.config)
            self.fusion_layers = nn.ModuleList(
                shared_layer for _ in range(self.config.num_fusion_layers)
            )
        else:
            self.fusion_layers = nn.ModuleList(
                CrossModalFusionLayer(self.config) for _ in range(self.config.num_fusion_layers)
            )
        self.refine_sensor = nn.ModuleList(
            nn.Linear(self.config.d_model, self.config.d_model, bias=False)
            for _ in range(max(len(self.fusion_layers) - 1, 0))
        )
        self.refine_image = nn.ModuleList(
            nn.Linear(self.config.d_model, self.config.d_model, bias=False)
            for _ in range(max(len(self.fusion_layers) - 1, 0))
        )
        d = self.config.d_model
        if self.config.prediction_head == "lead_conditioned":
            self.forecast_step_embedding = nn.Embedding(self.config.horizon, d)
            self.forecast_projection = nn.Sequential(
                nn.Linear(2 * d, d),
                nn.GELU(),
            )
        else:
            self.forecast_step_embedding = None
            self.forecast_projection = None
        if self.config.prediction_head == "direct":
            self.direct_force = nn.Linear(
                d, self.config.horizon * self.config.force_channels
            )
            self.direct_class = nn.Linear(
                d, self.config.horizon * self.config.num_classes
            )
        else:
            self.force_head = nn.Linear(d, self.config.force_channels)
            self.classification_head = nn.Linear(d, self.config.num_classes)

    def _mask_modalities(
        self, sensor: Tensor, image: Tensor, drop_modality: str | None
    ) -> tuple[Tensor, Tensor]:
        if drop_modality == "sensor":
            return torch.zeros_like(sensor), image
        if drop_modality == "image":
            return sensor, torch.zeros_like(image)
        if drop_modality is not None:
            raise ValueError("drop_modality must be None, 'sensor' or 'image'")
        probability = self.config.modality_dropout
        if self.training and probability > 0.0:
            draw = torch.rand(sensor.shape[0], device=sensor.device)
            keep_sensor = (draw >= probability / 2).to(sensor.dtype)
            keep_image = ((draw < probability / 2) | (draw >= probability)).to(
                image.dtype
            )
            return sensor * keep_sensor[:, None, None], image * keep_image[:, None, None]
        return sensor, image

    def forward(
        self,
        force: Tensor,
        images: Tensor,
        image_indices: Tensor,
        cycle: Tensor | None = None,
        batch_code: Tensor | None = None,
        relative: Tensor | None = None,
        drop_modality: str | None = None,
    ) -> dict[str, Any]:
        if force.ndim != 3 or force.shape[1] != self.config.input_len:
            raise ValueError(f"force must have shape (batch, {self.config.input_len}, channels)")
        if images.ndim not in (3, 5) or images.shape[0] != force.shape[0]:
            raise ValueError(
                "images must have shape (batch, images, channels, height, width), or "
                "(batch, images, features) for precomputed frozen-backbone features"
            )
        sensor = self.temporal_encoder(
            self.temporal_embedding(force, cycle, batch_code, relative)
        )
        image = self.image_branch(images, image_indices, cycle, batch_code, relative)
        sensor, image = self._mask_modalities(sensor, image, drop_modality)
        original_sensor, original_image = sensor, image
        fused: Tensor | None = None
        alignment_loss = force.new_zeros(())
        orthogonality_loss = force.new_zeros(())
        if self.config.ablation == "without_c2f":
            fused = 0.5 * (original_sensor + original_image)
        for index, layer in enumerate(self.fusion_layers):
            if index:
                if fused is None:
                    raise RuntimeError("fusion refinement requires the previous fused state")
                sensor_input = original_sensor + self.refine_sensor[index - 1](fused)
                image_input = original_image + self.refine_image[index - 1](fused)
            else:
                sensor_input, image_input = original_sensor, original_image
            fused, alignment, orthogonality = layer(sensor_input, image_input)
            alignment_loss = alignment_loss + alignment
            orthogonality_loss = orthogonality_loss + orthogonality
        if fused is None:
            raise RuntimeError("OmniFuser requires at least one fusion layer")
        final = fused[:, -1]
        if self.config.prediction_head == "direct":
            force_output = self.direct_force(final).view(
                -1, self.config.horizon, self.config.force_channels
            )
            class_output = self.direct_class(final).view(
                -1, self.config.horizon, self.config.num_classes
            )
        elif self.config.prediction_head == "paper_literal":
            force_step = self.force_head(final)
            class_step = self.classification_head(final)
            force_output = force_step[:, None].expand(
                -1, self.config.horizon, -1
            )
            class_output = class_step[:, None].expand(
                -1, self.config.horizon, -1
            )
        else:
            if self.forecast_step_embedding is None or self.forecast_projection is None:
                raise RuntimeError("lead-conditioned prediction layers are not initialized")
            lead_codes = self.forecast_step_embedding.weight.unsqueeze(0).expand(
                final.shape[0], -1, -1
            )
            context = final.unsqueeze(1).expand(-1, self.config.horizon, -1)
            future = self.forecast_projection(torch.cat((context, lead_codes), dim=-1))
            force_output = self.force_head(future)
            class_output = self.classification_head(future)
        return {
            "force": force_output,
            "class_logits": class_output,
            "alignment_loss": (
                alignment_loss / len(self.fusion_layers)
                if self.fusion_layers
                else alignment_loss
            ),
            "orthogonality_loss": (
                orthogonality_loss / len(self.fusion_layers)
                if self.fusion_layers
                else orthogonality_loss
            ),
        }


def multitask_loss(
    outputs: dict[str, Any],
    force_target: Tensor,
    class_target: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    force_loss = F.mse_loss(outputs["force"], force_target)
    class_loss = F.cross_entropy(
        outputs["class_logits"].reshape(-1, outputs["class_logits"].shape[-1]),
        class_target.reshape(-1).long(),
    )
    alignment_loss = outputs["alignment_loss"]
    orthogonality_loss = outputs["orthogonality_loss"]
    total = force_loss + class_loss
    return total, {
        "loss": total.detach(),
        "force_loss": force_loss.detach(),
        "class_loss": class_loss.detach(),
        "alignment_loss": alignment_loss.detach(),
        "orthogonality_loss": orthogonality_loss.detach(),
    }


def classification_metrics(logits: Tensor, targets: Tensor) -> dict[str, float]:
    predictions = logits.argmax(dim=-1).reshape(-1).long()
    targets = targets.reshape(-1).long()
    if predictions.shape != targets.shape:
        raise ValueError("logits and classification targets must have matching batch/time shapes")
    classes = logits.shape[-1]
    confusion = torch.bincount(
        targets * classes + predictions, minlength=classes * classes
    ).reshape(classes, classes)
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall = true_positive / confusion.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    support = confusion.sum(dim=1).float() / confusion.sum().clamp_min(1)
    return {
        "accuracy": float(true_positive.sum() / confusion.sum().clamp_min(1)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "precision_weighted": float((support * precision).sum()),
        "recall_weighted": float((support * recall).sum()),
        "f1_weighted": float((support * f1).sum()),
    }
