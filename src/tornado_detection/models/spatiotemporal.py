"""Sweep-aware causal spatiotemporal tornado detector."""

from __future__ import annotations

import math

import torch
from torch import nn


def _group_count(channels: int) -> int:
    """Return the largest useful GroupNorm divisor up to eight."""

    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    """Two-convolution residual block with optional downsampling."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(output_channels),
                output_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(output_channels),
                output_channels,
            ),
        )

        if (
            stride != 1
            or input_channels != output_channels
        ):
            self.skip = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(
                    _group_count(output_channels),
                    output_channels,
                ),
            )
        else:
            self.skip = nn.Identity()

        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(
            self.main(inputs) + self.skip(inputs)
        )


class SweepEncoder(nn.Module):
    """Shared encoder applied independently to each elevation sweep."""

    def __init__(
        self,
        input_channels: int,
        widths: tuple[int, ...],
    ) -> None:
        super().__init__()

        blocks: list[nn.Module] = []
        current_channels = input_channels

        for output_channels in widths:
            blocks.append(
                ResidualBlock(
                    current_channels,
                    output_channels,
                    stride=2,
                )
            )
            blocks.append(
                ResidualBlock(
                    output_channels,
                    output_channels,
                )
            )
            current_channels = output_channels

        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs)


class CausalConvGRUCell(nn.Module):
    """One causal recurrent update that preserves spatial structure."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        combined_channels = input_channels + hidden_channels

        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            combined_channels,
            2 * hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.candidate = nn.Conv2d(
            combined_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat((inputs, hidden), dim=1)
        reset, update = self.gates(combined).chunk(2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)

        candidate = torch.tanh(
            self.candidate(
                torch.cat(
                    (inputs, reset * hidden),
                    dim=1,
                )
            )
        )

        return (
            update * hidden
            + (1.0 - update) * candidate
        )


class CausalConvGRU(nn.Module):
    """Process a time-major feature sequence without future leakage."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.cell = CausalConvGRUCell(
            input_channels,
            hidden_channels,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(
                "ConvGRU inputs must have shape "
                "[batch, time, channels, height, width]"
            )

        batch, time, _, height, width = inputs.shape
        hidden = inputs.new_zeros(
            (
                batch,
                self.cell.hidden_channels,
                height,
                width,
            )
        )
        outputs = []

        for time_index in range(time):
            hidden = self.cell(
                inputs[:, time_index],
                hidden,
            )
            outputs.append(hidden)

        return torch.stack(outputs, dim=1)


class SpatiotemporalTornadoDetector(nn.Module):
    """Model explicit time, sweep, variable, and polar-coordinate axes.

    Expected input shapes:

    values: [batch, time, sweep, variable, azimuth, range]
    finite_mask: same as values
    range_folded_mask: [batch, time, sweep, azimuth, range]
    coordinates: [batch or 1, sweep, coordinate, azimuth, range]
    """

    def __init__(
        self,
        *,
        variable_count: int = 6,
        sweep_count: int = 2,
        coordinate_count: int = 5,
        encoder_widths: tuple[int, ...] = (
            32,
            64,
            128,
        ),
        temporal_channels: int = 128,
        pooling_temperature: float = 1.0,
        pooling_topk_fraction: float | None = None,
    ) -> None:
        super().__init__()

        if sweep_count != 2:
            raise ValueError(
                "The explicit sweep fusion currently requires "
                "exactly two elevation sweeps"
            )
        if variable_count < 1:
            raise ValueError(
                "variable_count must be positive"
            )
        if coordinate_count < 1:
            raise ValueError(
                "coordinate_count must be positive"
            )
        if pooling_temperature <= 0:
            raise ValueError(
                "pooling_temperature must be positive"
            )
        if (
            pooling_topk_fraction is not None
            and not 0.0 < pooling_topk_fraction <= 1.0
        ):
            raise ValueError(
                "pooling_topk_fraction must be in (0, 1]"
            )

        self.variable_count = variable_count
        self.sweep_count = sweep_count
        self.coordinate_count = coordinate_count
        self.pooling_temperature = pooling_temperature
        self.pooling_topk_fraction = pooling_topk_fraction

        sweep_input_channels = (
            variable_count
            + variable_count
            + 1
            + coordinate_count
        )
        encoded_channels = encoder_widths[-1]

        self.sweep_encoder = SweepEncoder(
            sweep_input_channels,
            encoder_widths,
        )
        self.sweep_fusion = nn.Sequential(
            nn.Conv2d(
                4 * encoded_channels,
                temporal_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(temporal_channels),
                temporal_channels,
            ),
            nn.SiLU(),
            ResidualBlock(
                temporal_channels,
                temporal_channels,
            ),
        )
        self.temporal = CausalConvGRU(
            temporal_channels,
            temporal_channels,
        )
        self.likelihood_head = nn.Sequential(
            nn.Conv2d(
                temporal_channels,
                temporal_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(
                temporal_channels // 2,
                1,
                kernel_size=1,
            ),
        )

    def _validate_inputs(
        self,
        values: torch.Tensor,
        finite_mask: torch.Tensor,
        range_folded_mask: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> None:
        if values.ndim != 6:
            raise ValueError(
                "values must have shape [batch, time, sweep, "
                "variable, azimuth, range]"
            )
        if finite_mask.shape != values.shape:
            raise ValueError(
                "finite_mask must have the same shape as values"
            )

        batch, time, sweeps, variables, height, width = values.shape

        if sweeps != self.sweep_count:
            raise ValueError(
                f"Expected {self.sweep_count} sweeps; got {sweeps}"
            )
        if variables != self.variable_count:
            raise ValueError(
                f"Expected {self.variable_count} variables; "
                f"got {variables}"
            )
        if range_folded_mask.shape != (
            batch,
            time,
            sweeps,
            height,
            width,
        ):
            raise ValueError(
                "range_folded_mask has an invalid shape"
            )
        if coordinates.ndim != 5:
            raise ValueError(
                "coordinates must have shape [batch or 1, sweep, "
                "coordinate, azimuth, range]"
            )
        if coordinates.shape[0] not in (1, batch):
            raise ValueError(
                "coordinate batch dimension must be 1 or match values"
            )
        if coordinates.shape[1:] != (
            sweeps,
            self.coordinate_count,
            height,
            width,
        ):
            raise ValueError(
                "coordinates have an invalid shape"
            )

    def _pool_likelihood(
        self,
        likelihood_maps: torch.Tensor,
    ) -> torch.Tensor:
        flattened = likelihood_maps.flatten(start_dim=2)
        temperature = self.pooling_temperature
        count = flattened.shape[-1]

        if self.pooling_topk_fraction is not None:
            count = max(
                1,
                math.ceil(
                    count
                    * self.pooling_topk_fraction
                ),
            )
            flattened = flattened.topk(
                count,
                dim=-1,
            ).values

        return temperature * (
            torch.logsumexp(
                flattened / temperature,
                dim=-1,
            )
            - math.log(count)
        )

    def forward_with_maps(
        self,
        values: torch.Tensor,
        finite_mask: torch.Tensor,
        range_folded_mask: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(
            values,
            finite_mask,
            range_folded_mask,
            coordinates,
        )

        batch, time, sweeps, _, height, width = values.shape

        if coordinates.shape[0] == 1 and batch != 1:
            coordinates = coordinates.expand(
                batch,
                -1,
                -1,
                -1,
                -1,
            )

        values = torch.nan_to_num(values, nan=0.0)
        finite_mask = finite_mask.to(dtype=values.dtype)
        range_folded_mask = range_folded_mask.to(dtype=values.dtype)
        coordinates = coordinates.to(dtype=values.dtype)

        coordinate_sequence = (
            coordinates[:, None]
            .expand(-1, time, -1, -1, -1, -1)
        )
        sweep_inputs = torch.cat(
            (
                values,
                finite_mask,
                range_folded_mask.unsqueeze(3),
                coordinate_sequence,
            ),
            dim=3,
        )
        sweep_inputs = sweep_inputs.reshape(
            batch * time * sweeps,
            -1,
            height,
            width,
        )

        encoded = self.sweep_encoder(sweep_inputs)
        _, channels, encoded_height, encoded_width = encoded.shape
        encoded = encoded.reshape(
            batch,
            time,
            sweeps,
            channels,
            encoded_height,
            encoded_width,
        )

        lower = encoded[:, :, 0]
        upper = encoded[:, :, 1]
        fused = torch.cat(
            (
                lower,
                upper,
                upper - lower,
                torch.abs(upper - lower),
            ),
            dim=2,
        )
        fused = self.sweep_fusion(
            fused.reshape(
                batch * time,
                4 * channels,
                encoded_height,
                encoded_width,
            )
        )
        fused = fused.reshape(
            batch,
            time,
            -1,
            encoded_height,
            encoded_width,
        )

        temporal_features = self.temporal(fused)
        likelihood_maps = self.likelihood_head(
            temporal_features.reshape(
                batch * time,
                temporal_features.shape[2],
                encoded_height,
                encoded_width,
            )
        ).reshape(
            batch,
            time,
            encoded_height,
            encoded_width,
        )
        logits = self._pool_likelihood(likelihood_maps)

        return logits, likelihood_maps

    def forward(
        self,
        values: torch.Tensor,
        finite_mask: torch.Tensor,
        range_folded_mask: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.forward_with_maps(
            values,
            finite_mask,
            range_folded_mask,
            coordinates,
        )
        return logits
