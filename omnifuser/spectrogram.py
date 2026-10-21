"""Optional image source: VLT-style images of the raw force inside each input window.

An experiment, not part of the paper protocol. Pooling to 1 s steps removes every
force component faster than 0.5 Hz; this image source is where that content can
re-enter the model. Following VLT (Wang et al., arXiv:2607.14510, Sec. III-C), the raw
(unpooled) force of a window is turned into a stack of frequency feature maps:

* channels 0-2, time-frequency: STFT log power per force axis (Fx, Fy, Fz), frequency
  on the rows (0 Hz at the top row) and time on the columns. VLT uses a wavelet
  scalogram here (its Eq. 7); the STFT is the fixed-window version of the same map and
  costs about a hundredth of a Morlet transform per window, which keeps it fast enough
  to compute in the data loader.
* channels 3-5, global spectrum: log power of the FFT of the whole segment per axis
  (VLT Eq. 6), averaged into the same frequency rows and repeated along the columns, so
  each pixel holds the local power and the segment's mean power at the same frequency.
* channel 6, recurrence: Theta(eps - ||v_i - v_j||) between the spectral columns v_i of
  channels 0-2 (VLT Eq. 5), with eps set per image so that ``recurrence_rate`` of the
  pairs recur. Columns are time, rows are time compressed to the image height.

A learnable convolution (``ModelConfig.image_channels``, VLT Eq. 8) maps these
``IMAGE_CHANNELS`` maps to the 3 channels EfficientNet-B0 takes.

Only this time-to-image construction (VLT Eqs. 5-8) is taken from VLT. Everything
after it is OmniFuser as implemented here: the EfficientNet-B0 and GAT image branch,
MRTE, C2F, heads and loss. VLT's Time-MoE, text branch, MAE encoder, attention fusion,
gradient alignment, consistency loss and two-stage training are not used.

As in the paper, each pass contributes one image, repeated over its steps: a window that
spans a pass boundary gets one image per pass segment, each built from that pass's
samples inside the window only.

Leakage: an image uses exactly the raw samples of the window's input steps, which the
sensor branch already sees as their 1 s means. No sample after the input end and no
label enters, so these runs use image lag 0 and every window has an image (there is no
"no previous image" case). Each map is standardized on its own, so no statistic is
fitted across windows or tools.
"""
from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from omnifuser.data import ForceNormalizer, Record, ToolTimeline, WindowDataset

IMAGE_CHANNELS = 7


@dataclass(frozen=True)
class SpectrogramSettings:
    """STFT window and hop in raw samples (1 ms each at 1 kHz), and the recurrence rate."""

    window: int = 256
    hop: int = 64
    recurrence_rate: float = 0.1

    def __post_init__(self) -> None:
        if self.window < 2 or self.hop < 1:
            raise ValueError("spectrogram window must be at least 2 and hop at least 1")
        if self.hop > self.window:
            raise ValueError("spectrogram hop must not exceed the window")
        if not 0.0 < self.recurrence_rate < 1.0:
            raise ValueError("recurrence_rate must be in (0, 1)")


def _standardize(maps: Tensor) -> Tensor:
    mean = maps.mean(dim=(1, 2), keepdim=True)
    std = maps.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return (maps - mean) / std


def force_window_image(
    force: np.ndarray, settings: SpectrogramSettings, height: int, width: int
) -> Tensor:
    """(samples, 3) raw force -> (IMAGE_CHANNELS, height, width) float32 feature maps."""
    if force.ndim != 2 or force.shape[1] != 3:
        raise ValueError(f"force must have shape (samples, 3), got {force.shape}")
    if len(force) < 2:
        raise ValueError("a force segment needs at least 2 raw samples")
    signal = torch.from_numpy(np.ascontiguousarray(force.T, dtype=np.float32))
    # The static cutting force is in the sensor branch already; the maps describe dynamics.
    signal = signal - signal.mean(dim=1, keepdim=True)
    if signal.shape[1] < settings.window:  # a segment shorter than one STFT frame
        signal = F.pad(signal, (0, settings.window - signal.shape[1]))
    transform = torch.stft(
        signal,
        n_fft=settings.window,
        hop_length=settings.hop,
        window=torch.hann_window(settings.window),
        center=False,
        return_complex=True,
    )
    log_power = torch.log10(transform.abs().square() + 1e-12)
    local = F.interpolate(
        log_power[None], size=(height, width), mode="bilinear", align_corners=False, antialias=True
    )[0]
    spectrum = torch.log10(torch.fft.rfft(signal).abs().square() + 1e-12)
    spectrum = F.adaptive_avg_pool1d(spectrum[None], height)[0]
    global_maps = spectrum[:, :, None].expand(-1, -1, width)
    columns = local.permute(2, 0, 1).reshape(width, -1)
    distances = torch.cdist(columns, columns)
    threshold = torch.quantile(distances.flatten(), settings.recurrence_rate)
    recurrence = (distances <= threshold).to(torch.float32)
    recurrence = F.adaptive_avg_pool2d(recurrence[None, None], (height, width))[0]
    return torch.cat(
        (_standardize(local), _standardize(global_maps), _standardize(recurrence))
    )


@dataclass(frozen=True)
class _ToolPasses:
    starts: tuple[int, ...]  # first pooled step of each pass in the tool's timeline
    raw: tuple[np.ndarray, ...]  # each pass's raw samples, steps * pool of them


def raw_passes_by_tool(raw_records: Sequence[Record], pool: int) -> dict[str, _ToolPasses]:
    """Each tool's raw passes in timeline order, cut to the samples its pooled steps cover."""
    by_tool: dict[str, _ToolPasses] = {}
    for tool in sorted({record.tool for record in raw_records}, key=lambda item: int(item[1:])):
        selected = sorted(
            (record for record in raw_records if record.tool == tool),
            key=lambda item: (item.run, item.block),
        )
        starts: list[int] = []
        raw: list[np.ndarray] = []
        position = 0
        for record in selected:
            steps = len(record.force) // pool
            starts.append(position)
            raw.append(record.force[: steps * pool])
            position += steps
        by_tool[tool] = _ToolPasses(tuple(starts), tuple(raw))
    return by_tool


class ForceWindowDataset(WindowDataset):
    """``WindowDataset`` whose images are built from the raw force of each input window.

    Items carry ``images`` (segments, IMAGE_CHANNELS, H, W) and ``image_index`` (one
    segment index per input step) instead of photo paths.
    """

    def __init__(
        self,
        timelines: Sequence[ToolTimeline],
        normalizer: ForceNormalizer,
        raw_passes: dict[str, _ToolPasses],
        pool: int,
        settings: SpectrogramSettings,
        height: int,
        width: int,
        input_len: int = 96,
        horizon: int = 96,
        stride: int = 24,
    ) -> None:
        super().__init__(timelines, normalizer, input_len, horizon, stride)
        self.pool = pool
        self.settings = settings
        self.height, self.width = height, width
        self.passes = [raw_passes[timeline.tool] for timeline in self.timelines]
        for timeline, passes in zip(self.timelines, self.passes):
            covered = sum(len(raw) for raw in passes.raw) // pool
            if covered != len(timeline.force):
                raise ValueError(
                    f"{timeline.tool}: raw passes cover {covered} steps, the timeline has "
                    f"{len(timeline.force)}; raw and pooled records must come from the same load"
                )

    def segments(self, timeline_index: int, window_start: int) -> list[tuple[int, int, int]]:
        """(pass, first step, end step) of each pass segment inside the input window."""
        passes = self.passes[timeline_index]
        input_end = window_start + self.input_len
        first = bisect.bisect_right(passes.starts, window_start) - 1
        result: list[tuple[int, int, int]] = []
        for pass_index in range(first, len(passes.starts)):
            start = passes.starts[pass_index]
            if start >= input_end:
                break
            end = start + len(passes.raw[pass_index]) // self.pool
            result.append((pass_index, max(start, window_start), min(end, input_end)))
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        del item["image_paths"]
        timeline_index, window_start = self.locate(index)
        passes = self.passes[timeline_index]
        images: list[Tensor] = []
        image_index = np.empty(self.input_len, dtype=np.int64)
        for position, (pass_index, start, end) in enumerate(
            self.segments(timeline_index, window_start)
        ):
            local = start - passes.starts[pass_index]
            samples = passes.raw[pass_index][
                local * self.pool : (local + end - start) * self.pool
            ]
            images.append(force_window_image(samples, self.settings, self.height, self.width))
            image_index[start - window_start : end - window_start] = position
        item["images"] = torch.stack(images)
        item["image_index"] = image_index
        return item
