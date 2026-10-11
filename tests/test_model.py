import torch
import pytest

from omnifuser.config import ModelConfig
from omnifuser.models import (
    CrossModalFusionLayer,
    GraphAttentionLayer,
    ImageBranch,
    OmniFuser,
    RTDBlock,
    TemporalEmbedding,
    classification_metrics,
    multitask_loss,
    sinusoidal_position_encoding,
)


def _config() -> ModelConfig:
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
    assert tuple(embedding.value[0].weight.shape) == (32, 3)
    assert tuple(embedding.value[2].weight.shape) == (32, 32)


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
    assert torch.equal(outputs["force"][:, 0], outputs["force"][:, -1])
    assert torch.equal(outputs["class_logits"][:, 0], outputs["class_logits"][:, -1])
    assert model.force_head.weight.shape == (3, config.d_model)
    loss, metrics = multitask_loss(
        outputs,
        torch.randn(2, config.horizon, 3),
        torch.randint(0, 3, (2, config.horizon)),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert model.refine_sensor[0].weight.grad is not None
    assert model.refine_image[0].weight.grad is not None


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
