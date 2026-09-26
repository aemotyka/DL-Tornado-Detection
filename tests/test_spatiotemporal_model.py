"""Tests for the sweep-aware causal spatiotemporal model."""

import pytest

torch = pytest.importorskip("torch")

from tornado_detection.models import (  # noqa: E402
    SpatiotemporalTornadoDetector,
)


def _inputs(batch: int = 2):
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(
        batch,
        4,
        2,
        6,
        32,
        64,
        generator=generator,
    )
    finite_mask = torch.ones_like(values)
    finite_mask[:, :, :, :, 0, 0] = 0
    values[:, :, :, :, 0, 0] = float("nan")
    range_folded_mask = torch.zeros(
        batch,
        4,
        2,
        32,
        64,
    )
    coordinates = torch.randn(
        1,
        2,
        5,
        32,
        64,
        generator=generator,
    )
    return (
        values,
        finite_mask,
        range_folded_mask,
        coordinates,
    )


def _model():
    return SpatiotemporalTornadoDetector(
        encoder_widths=(8, 16),
        temporal_channels=16,
    )


def test_output_shapes_and_finite_gradients() -> None:
    model = _model()
    inputs = _inputs()

    logits, maps = model.forward_with_maps(*inputs)

    assert logits.shape == (2, 4)
    assert maps.shape == (2, 4, 8, 16)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(maps).all()

    logits.sum().backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert all(
        gradient is not None
        for gradient in gradients
    )
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_predictions_are_causal() -> None:
    torch.manual_seed(23)
    model = _model().eval()
    inputs = list(_inputs(batch=1))

    with torch.no_grad():
        original = model(*inputs)

    changed_values = inputs[0].clone()
    changed_values[:, 3] = torch.nan_to_num(
        changed_values[:, 3],
        nan=0.0,
    ) + 100.0
    inputs[0] = changed_values

    with torch.no_grad():
        changed = model(*inputs)

    torch.testing.assert_close(
        original[:, :3],
        changed[:, :3],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        original[:, 3],
        changed[:, 3],
    )


def test_rejects_flattened_frame_input() -> None:
    model = _model()
    values = torch.zeros(2, 4, 12, 32, 64)

    with pytest.raises(
        ValueError,
        match="values must have shape",
    ):
        model(
            values,
            values,
            torch.zeros(2, 4, 2, 32, 64),
            torch.zeros(1, 2, 5, 32, 64),
        )


def test_parameter_count_is_substantial_but_bounded() -> None:
    model = SpatiotemporalTornadoDetector()
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert 1_000_000 < parameter_count < 10_000_000


def test_sparse_pooling_uses_only_top_spatial_cells() -> None:
    model = SpatiotemporalTornadoDetector(
        encoder_widths=(8, 16),
        temporal_channels=16,
        pooling_topk_fraction=0.25,
    )
    maps = torch.arange(
        16,
        dtype=torch.float32,
    ).reshape(1, 1, 4, 4)

    actual = model._pool_likelihood(maps)

    selected = torch.tensor(
        [12.0, 13.0, 14.0, 15.0]
    )
    expected = (
        torch.logsumexp(
            selected,
            dim=0,
        )
        - torch.log(torch.tensor(4.0))
    )

    torch.testing.assert_close(
        actual,
        expected.reshape(1, 1),
    )


@pytest.mark.parametrize(
    "fraction",
    [0.0, -0.1, 1.1],
)
def test_rejects_invalid_sparse_pooling_fraction(
    fraction: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="pooling_topk_fraction",
    ):
        SpatiotemporalTornadoDetector(
            pooling_topk_fraction=fraction,
        )
