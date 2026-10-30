import pytest
import torch

from omnifuser.config import ModelConfig
from omnifuser.models import (
    CrossModalFusionLayer,
    GraphAttentionLayer,
    ImageBranch,
    MultiResolutionTemporalExtractor,
    OmniFuser,
    RTDBlock,
    TemporalEmbedding,
    classification_metrics,
    multitask_loss,
    sinusoidal_position_encoding,
)


def _config(prediction_head: str = "lead_conditioned") -> ModelConfig:
    return ModelConfig(
        d_model=32,
        input_len=12,
        horizon=3,
        num_rtd_layers=2,
        rtd_rank=8,
        num_gat_layers=1,
        num_gat_heads=4,
        num_fusion_layers=2,
        num_landmarks=2,
        image_encoder="compact",
        dropout=0.0,
        prediction_head=prediction_head,
    )


def test_graph_attention_rows_sum_to_one() -> None:
    layer = GraphAttentionLayer(8, 2, 0.0)
    output, attention = layer(torch.randn(2, 5, 8))
    assert output.shape == (2, 5, 8)
    assert attention.shape == (2, 2, 5, 5)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2, 2, 5))


def test_position_encoding_uses_fixed_sine_cosine_pairs() -> None:
    encoding = sinusoidal_position_encoding(3, 4)
    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [torch.sin(torch.tensor(1.0)), torch.cos(torch.tensor(1.0)),
             torch.sin(torch.tensor(0.01)), torch.cos(torch.tensor(0.01))],
            [torch.sin(torch.tensor(2.0)), torch.cos(torch.tensor(2.0)),
             torch.sin(torch.tensor(0.02)), torch.cos(torch.tensor(0.02))],
        ]
    )
    assert torch.allclose(encoding, expected)
    embedding = TemporalEmbedding(_config())
    assert "position_encoding" not in dict(embedding.named_parameters())
    first, second = embedding.value[0], embedding.value[2]
    assert isinstance(first, torch.nn.Linear) and isinstance(second, torch.nn.Linear)
    assert tuple(first.weight.shape) == (32, 3)
    assert tuple(second.weight.shape) == (32, 32)


def test_learned_period_codes_are_added_and_receive_gradients() -> None:
    embedding = TemporalEmbedding(_config()).eval()
    force = torch.zeros(2, 12, 3)
    cycle = torch.tensor([[1] * 12, [2] * 12])
    batch_code = torch.tensor([[1] * 12, [2] * 12])
    relative = torch.tensor([list(range(12)), list(range(4, 16))])

    output = embedding(force, cycle, batch_code, relative)
    # the default metadata mode ("all") creates every table
    assert embedding.cycle is not None and embedding.batch is not None and embedding.relative is not None
    expected = (
        embedding.value(force)
        + embedding.position_encoding[None]
        + embedding.cycle(cycle)
        + embedding.batch(batch_code)
        + embedding.relative(relative)
    )

    assert output.shape == (2, 12, 32)
    assert torch.allclose(output, expected)
    assert embedding.cycle.num_embeddings == 128
    assert embedding.batch.num_embeddings == 8
    assert embedding.relative.num_embeddings == 16
    output.sum().backward()
    assert embedding.cycle.weight.grad is not None
    assert embedding.batch.weight.grad is not None
    assert embedding.relative.weight.grad is not None
    assert "position_encoding" not in dict(embedding.named_parameters())


def test_image_branch_adds_positions_after_gat() -> None:
    config = _config()
    branch = ImageBranch(config).eval()
    images = torch.randn(1, 1, 3, 16, 24)
    indices = torch.zeros(1, config.input_len, dtype=torch.long)
    metadata = torch.zeros(1, config.input_len, dtype=torch.long)
    output = branch(images, indices, metadata, metadata, metadata)
    expected_delta = (
        branch.position_encoding[1] - branch.position_encoding[0]
    )
    assert torch.allclose(output[0, 1] - output[0, 0], expected_delta, atol=1e-5)


def test_c2f_uses_orthogonal_private_residual_and_low_rank_bottleneck() -> None:
    layer = CrossModalFusionLayer(_config())
    values = torch.randn(2, 12, 32)
    shared, private = layer._decompose(values, layer.shared_r)
    dot = (shared * private).sum(dim=(1, 2))
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-5)
    assert layer.project[0].in_features == 64
    assert layer.project[0].out_features == 8
    assert layer.project[2].out_features == 32


def test_adaptive_landmarks_are_temporal_weighted_averages() -> None:
    layer = CrossModalFusionLayer(_config())
    token = torch.randn(2, 1, 32)
    values = token.expand(-1, 12, -1)
    landmarks, landmark_values = layer._landmarks(values, layer.score_r, layer.value_r)

    expected_values = layer.value_r(token).expand(-1, layer.score_r.out_features, -1)
    assert torch.allclose(landmarks, token.expand_as(landmarks), atol=1e-6)
    assert torch.allclose(landmark_values, expected_values, atol=1e-6)


def test_rtd_and_multitask_model_forward_backward() -> None:
    config = _config()
    block = RTDBlock(config.input_len, config)
    # Eq. 2: depthwise temporal convolution with kernel = stride = s_l (3, as documented)
    assert config.rtd_stride == 3
    assert block.depthwise.kernel_size == (3,) and block.depthwise.stride == (3,)
    assert block.depthwise.groups == config.d_model
    down, restored = block(torch.randn(2, config.input_len, config.d_model))
    assert down.shape == (2, 4, config.d_model)
    assert restored.shape == (2, config.input_len, config.d_model)

    model = OmniFuser(config)
    force = torch.randn(2, config.input_len, 3)
    images = torch.randn(2, 3, 3, 16, 24)
    image_indices = torch.tensor([[0] * 12, [1] * 12])
    outputs = model(
        force,
        images,
        image_indices,
        cycle=torch.ones(2, 12, dtype=torch.long),
        batch_code=torch.ones(2, 12, dtype=torch.long),
        relative=torch.ones(2, 12, dtype=torch.long),
    )
    assert outputs["force"].shape == (2, config.horizon, 3)
    assert outputs["class_logits"].shape == (2, config.horizon, 3)
    assert not torch.allclose(outputs["force"][:, 0], outputs["force"][:, -1])
    assert not torch.allclose(
        outputs["class_logits"][:, 0], outputs["class_logits"][:, -1]
    )
    assert model.force_head.weight.shape == (3, config.d_model)
    assert model.forecast_step_embedding is not None
    assert model.forecast_step_embedding.weight.shape == (
        config.horizon,
        config.d_model,
    )
    loss, metrics = multitask_loss(
        outputs,
        torch.randn(2, config.horizon, 3),
        torch.randint(0, 3, (2, config.horizon)),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    refine_sensor, refine_image = model.refine_sensor[0], model.refine_image[0]
    assert isinstance(refine_sensor, torch.nn.Linear) and isinstance(refine_image, torch.nn.Linear)
    assert refine_sensor.weight.grad is not None
    assert refine_image.weight.grad is not None
    assert model.forecast_step_embedding is not None
    lead_gradient = model.forecast_step_embedding.weight.grad
    assert lead_gradient is not None and lead_gradient.abs().sum() > 0


def test_paper_literal_head_repeats_outputs_across_horizon() -> None:
    config = _config(prediction_head="paper_literal")
    model = OmniFuser(config)
    force = torch.randn(2, config.input_len, 3)
    images = torch.randn(2, 2, 3, 16, 24)
    image_indices = torch.zeros(2, config.input_len, dtype=torch.long)
    metadata = torch.ones(2, config.input_len, dtype=torch.long)

    outputs = model(force, images, image_indices, metadata, metadata, metadata)

    assert outputs["force"].shape == (2, config.horizon, 3)
    assert outputs["class_logits"].shape == (2, config.horizon, 3)
    assert torch.equal(
        outputs["force"],
        outputs["force"][:, :1].expand(-1, config.horizon, -1),
    )
    assert torch.equal(
        outputs["class_logits"],
        outputs["class_logits"][:, :1].expand(-1, config.horizon, -1),
    )
    assert model.forecast_step_embedding is None
    assert model.forecast_projection is None


def test_shared_c2f_weights_and_plain_private_attention_options() -> None:
    config = ModelConfig(
        **{
            **_config().__dict__,
            "share_c2f": True,
            "private_attn": "plain",
        }
    )
    model = OmniFuser(config)
    layer = model.fusion_layers[0]

    assert model.fusion_layers[1] is layer
    assert layer.private_attn == "plain"
    assert not hasattr(layer, "private_norm_r")
    output = layer(
        torch.randn(2, config.input_len, config.d_model),
        torch.randn(2, config.input_len, config.d_model),
    )
    assert output[0].shape == (2, config.input_len, config.d_model)


@pytest.mark.parametrize(
    "ablation",
    [
        "without_rr",
        "without_mrte",
        "without_rr_mrte",
        "concat_c2f",
        "without_c2f",
        "without_als",
        "without_hybrid_gate",
        "moving_average",
    ],
)
def test_paper_ablation_variants_run_forward_and_backward(ablation: str) -> None:
    config = ModelConfig(
        **{
            **_config().__dict__,
            "ablation": ablation,
        }
    )
    model = OmniFuser(config)
    force = torch.randn(2, config.input_len, 3)
    images = torch.randn(2, 2, 3, 16, 24)
    indices = torch.tensor([[0] * 6 + [1] * 6, [1] * config.input_len])
    outputs = model(force, images, indices)
    loss, _ = multitask_loss(
        outputs,
        torch.randn(2, config.horizon, 3),
        torch.randint(0, 3, (2, config.horizon)),
    )
    loss.backward()

    assert outputs["force"].shape == (2, config.horizon, 3)
    assert outputs["class_logits"].shape == (2, config.horizon, 3)
    assert torch.isfinite(loss)


def _model_inputs(config: ModelConfig, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    force = torch.randn(2, config.input_len, 3, generator=generator)
    images = torch.randn(2, 2, 3, 16, 24, generator=generator)
    indices = torch.tensor([[0] * 6 + [1] * 6, [1] * config.input_len])
    return force, images, indices


@pytest.mark.parametrize("ablation", ["none", "without_c2f", "concat_c2f"])
def test_masked_modality_input_cannot_influence_predictions(ablation: str) -> None:
    config = ModelConfig(**{**_config().__dict__, "ablation": ablation})
    model = OmniFuser(config).eval()
    force_a, images_a, indices = _model_inputs(config, 0)
    force_b, images_b, _ = _model_inputs(config, 1)
    with torch.no_grad():
        sensor_missing_a = model(force_a, images_a, indices, drop_modality="sensor")
        sensor_missing_b = model(force_b, images_a, indices, drop_modality="sensor")
        image_missing_a = model(force_a, images_a, indices, drop_modality="image")
        image_missing_b = model(force_a, images_b, indices, drop_modality="image")
        full_a = model(force_a, images_a, indices)
        full_b = model(force_b, images_a, indices)

    for key in ("force", "class_logits"):
        assert torch.allclose(sensor_missing_a[key], sensor_missing_b[key], atol=1e-6)
        assert torch.allclose(image_missing_a[key], image_missing_b[key], atol=1e-6)
        # with both streams present the force input does matter
        assert not torch.allclose(full_a[key], full_b[key])


def test_paper_ablation_variants_change_the_intended_component() -> None:
    def build(ablation: str) -> OmniFuser:
        return OmniFuser(ModelConfig(**{**_config().__dict__, "ablation": ablation}))

    full = build("none")
    assert isinstance(full.temporal_encoder, MultiResolutionTemporalExtractor)
    assert len(full.temporal_encoder.layers) == 2 and len(full.fusion_layers) == 2

    assert isinstance(build("without_mrte").temporal_encoder, torch.nn.Identity)
    assert len(build("without_mrte").fusion_layers) == 2
    assert len(build("without_rr").fusion_layers) == 1
    assert len(build("without_rr").refine_sensor) == 0
    both = build("without_rr_mrte")
    assert isinstance(both.temporal_encoder, torch.nn.Identity) and len(both.fusion_layers) == 1
    assert len(build("without_c2f").fusion_layers) == 0
    concat = build("concat_c2f")
    assert all(not isinstance(layer, CrossModalFusionLayer) for layer in concat.fusion_layers)
    assert len(concat.fusion_layers) == 2
    moving_average = build("moving_average")
    assert isinstance(moving_average.temporal_encoder, MultiResolutionTemporalExtractor)
    assert all(
        block.low_rank_left is None and block.use_moving_average
        for block in moving_average.temporal_encoder.layers
    )


def test_without_als_uses_fixed_segment_mean_landmarks() -> None:
    config = ModelConfig(**{**_config().__dict__, "ablation": "without_als"})
    layer = CrossModalFusionLayer(config)
    values = torch.randn(2, config.input_len, config.d_model)
    landmarks, landmark_values = layer._landmarks(values, layer.score_r, layer.value_r)

    segments = values.reshape(2, config.num_landmarks, -1, config.d_model).mean(dim=2)
    assert torch.allclose(landmarks, segments, atol=1e-6)
    assert torch.allclose(
        landmark_values,
        layer.value_r(values).reshape(2, config.num_landmarks, -1, config.d_model).mean(dim=2),
        atol=1e-6,
    )
    # landmarks stay distinct, so cross-attention is not collapsed to one global mean
    assert not torch.allclose(landmarks[:, 0], landmarks[:, 1])


def test_without_hybrid_gate_uses_fixed_equal_mixture() -> None:
    config = ModelConfig(**{**_config().__dict__, "ablation": "without_hybrid_gate"})
    model = OmniFuser(config)
    force, images, indices = _model_inputs(config, 0)
    outputs = model(force, images, indices)
    loss, _ = multitask_loss(
        outputs, torch.randn(2, config.horizon, 3), torch.randint(0, 3, (2, config.horizon))
    )
    loss.backward()
    layer = model.fusion_layers[0]
    assert isinstance(layer, CrossModalFusionLayer)
    assert layer.global_gate_r.weight.grad is None
    assert layer.gate_mix_r.grad is None
    assert layer.private_self_r.in_proj_weight.grad is not None


def test_model_config_rejects_unknown_prediction_head() -> None:
    with pytest.raises(ValueError, match="prediction_head"):
        _config(prediction_head="unknown")


def test_classification_metrics_report_macro_scores() -> None:
    logits = torch.tensor(
        [[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]]
    )
    targets = torch.tensor([[0, 1, 1]])
    metrics = classification_metrics(logits, targets)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["precision_macro"] == pytest.approx(2 / 3)
    assert metrics["recall_macro"] == pytest.approx(0.5)
    assert metrics["f1_macro"] == pytest.approx(5 / 9)
    assert metrics["precision_weighted"] == pytest.approx(1.0)
    assert metrics["recall_weighted"] == pytest.approx(2 / 3)
    assert metrics["f1_weighted"] == pytest.approx(7 / 9)


def test_defaults_select_the_printed_paper_readings() -> None:
    config = ModelConfig()
    assert config.private_attn == "plain"  # P_hat = MHA(P)
    assert config.als_mode == "k_colnorm"  # alpha_t = softmax over k per token
    assert config.pre_attn is False  # no attention before the shared/private split
    assert config.decomp_mode == "branch"  # Fig. 5 trend/seasonal feed-forwards
    assert config.prediction_head == "direct"  # Eq. 21-22 from z_T, one W_r per lead
    assert config.exact_split is False
    assert (config.d_model, config.num_rtd_layers, config.rtd_stride, config.rtd_rank) == (32, 4, 3, 8)
    assert (config.num_gat_layers, config.num_gat_heads, config.num_landmarks) == (4, 4, 4)
    assert (config.num_fusion_layers, config.dropout) == (2, 0.1)


def test_default_landmarks_keep_the_printed_per_token_softmax_but_bounded_scale() -> None:
    layer = CrossModalFusionLayer(_config())
    for steps in (12, 48):
        values = torch.randn(2, steps, 32)
        landmarks, _ = layer._landmarks(values, layer.score_r, torch.nn.Identity())
        alpha = torch.softmax(layer.score_r(values), dim=-1)  # Eq. 10: softmax over k
        printed = alpha.transpose(1, 2) @ values  # L = alpha^T S
        column_mass = alpha.sum(dim=1).unsqueeze(-1)
        assert torch.allclose(landmarks, printed / column_mass, atol=1e-5)
        # each landmark is a convex combination, so its norm cannot exceed the largest token norm
        assert landmarks.norm(dim=-1).max() <= values.norm(dim=-1).max() + 1e-5


def test_exact_split_makes_shared_plus_private_reconstruct_the_input() -> None:
    values = torch.randn(2, 12, 32)
    default = CrossModalFusionLayer(_config())
    exact = CrossModalFusionLayer(ModelConfig(**{**_config().__dict__, "exact_split": True}))
    exact.load_state_dict(default.state_dict())

    shared, private = default._decompose(values, default.shared_r)
    exact_shared, exact_private = exact._decompose(values, exact.shared_r)

    assert torch.allclose(private, exact_private)
    assert torch.allclose(exact_shared + exact_private, values, atol=1e-5)  # Z = P + S
    assert not torch.allclose(shared + private, values, atol=1e-3)  # Eq. 5 alone gives Z = P + cS
    dot = (exact_shared * exact_private).sum(dim=(1, 2))
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-3)
    OmniFuser(ModelConfig(**{**_config().__dict__, "exact_split": True}))(*_model_inputs(_config(), 0))


def test_residual_private_path_remains_available() -> None:
    config = ModelConfig(**{**_config().__dict__, "private_attn": "residual_ln"})
    layer = OmniFuser(config).fusion_layers[0]
    assert hasattr(layer, "private_norm_r")
    OmniFuser(config)(*_model_inputs(config, 0))


def test_image_encoder_rejects_unknown_names_instead_of_falling_back() -> None:
    from omnifuser.models import ImageFeatureEncoder

    with pytest.raises(ValueError, match="unknown image encoder"):
        ImageFeatureEncoder("efficientnet-b0", 32)
    with pytest.raises(ValueError, match="image_encoder"):
        ModelConfig(image_encoder="efficientnet-b0")


def test_photos_are_imagenet_normalized_exactly_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import numpy as np
    import torchvision.models as tvm
    from PIL import Image

    from omnifuser.data import IMAGE_MEAN, IMAGE_STD, ImageBank
    from omnifuser.models import ImageFeatureEncoder

    # the data pipeline applies the ImageNet statistics that pretrained weights expect
    pixels = np.random.default_rng(0).integers(0, 256, (32, 96, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "photo.png")
    bank = ImageBank([tmp_path / "photo.png"], height=32, width=96)
    expected = (pixels.astype(np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
    assert torch.allclose(bank.tensor(tmp_path / "photo.png"), torch.from_numpy(expected.transpose(2, 0, 1)))

    # an untrained backbone stands in for the download; the weights requested are recorded
    real = tvm.efficientnet_b0
    requested: list = []
    monkeypatch.setattr(
        tvm, "efficientnet_b0", lambda weights=None: requested.append(weights) or real(weights=None)
    )
    pretrained = ImageFeatureEncoder("efficientnet_b0", 8, pretrained=True).eval()
    scratch = ImageFeatureEncoder("efficientnet_b0", 8).eval()
    assert requested == [tvm.EfficientNet_B0_Weights.IMAGENET1K_V1, None]

    seen: list[torch.Tensor] = []
    for encoder in (pretrained, scratch):
        encoder.features.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    images = torch.rand(2, 1, 3, 32, 96)
    pretrained(images)
    scratch(images)

    # both encoders pass the already-normalized pixels through unchanged
    flat = images.flatten(0, 1)
    assert torch.equal(seen[0], flat)
    assert torch.equal(seen[1], flat)
    assert set(pretrained.state_dict()) == set(scratch.state_dict())


def _fit(model: OmniFuser, inputs: tuple, force_target: torch.Tensor, class_target: torch.Tensor, steps: int) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss, _ = multitask_loss(model(*inputs), force_target, class_target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def test_model_learns_end_to_end_by_memorizing_a_small_batch() -> None:
    """Competence check: the whole model, trained with its real loss, can fit data."""
    torch.manual_seed(0)
    config = _config("direct")
    model = OmniFuser(config)
    force = torch.randn(8, config.input_len, 3)
    images = torch.randn(8, 2, 3, 16, 24)
    indices = torch.tensor([[0] * 6 + [1] * 6] * 8)
    force_target = torch.randn(8, config.horizon, 3)
    class_target = torch.randint(0, 3, (8, config.horizon))
    losses = _fit(model, (force, images, indices), force_target, class_target, steps=150)
    assert losses[-1] < 0.2 * losses[0]
    model.eval()
    with torch.no_grad():
        predicted = model(force, images, indices)["class_logits"].argmax(-1)
    assert torch.equal(predicted, class_target)


def test_image_information_reaches_the_class_head_through_fusion() -> None:
    """The class depends only on the image; the force input is noise.

    The model must learn it through the image branch and C2F, and fall back to
    chance when the image is masked at inference.
    """
    torch.manual_seed(1)
    config = _config("direct")
    model = OmniFuser(config)
    classes = torch.arange(24) % 3
    patterns = torch.randn(3, 3, 16, 24)
    images = patterns[classes][:, None]  # one image per window, chosen by its class
    indices = torch.zeros(24, config.input_len, dtype=torch.long)
    force = torch.randn(24, config.input_len, 3)
    class_target = classes[:, None].expand(-1, config.horizon)
    force_target = torch.zeros(24, config.horizon, 3)
    _fit(model, (force, images, indices), force_target, class_target, steps=150)
    model.eval()
    with torch.no_grad():
        fresh_force = torch.randn_like(force)  # unseen force: only the image can help
        accuracy = (model(fresh_force, images, indices)["class_logits"].argmax(-1) == class_target).float().mean()
        masked = model(fresh_force, images, indices, drop_modality="image")["class_logits"].argmax(-1)
    assert accuracy == 1.0
    assert (masked == class_target).float().mean() < 0.7
