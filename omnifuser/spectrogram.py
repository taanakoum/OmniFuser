"""Optional image source: per-pass force spectrograms instead of tool photos.

An experiment, not part of the paper protocol. Each pass's raw (unpooled) force is
turned into a 3-channel image, one channel per force axis (Fx, Fy, Fz): the log power
of a short-time Fourier transform, frequency on the rows and time on the columns,
resized to the photo size so the image branch is unchanged. Pooling to 1 s steps
removes everything faster than 1 Hz; the spectrogram is where that content can enter.

Leakage: a pass's spectrogram covers the whole pass, including force samples that fall
inside the forecast window of a window ending mid-pass. Spectrogram runs therefore
require the previous pass's image (image lag 1); ``train_fold`` enforces this.

Each image is standardized on its own (per channel), so no statistic is fitted across
tools and nothing flows from validation or test tools into training.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from omnifuser.data import NO_IMAGE, Record


@dataclass(frozen=True)
class SpectrogramSettings:
    """STFT window and hop, in raw samples (1 ms each at the recorded 1 kHz)."""

    window: int = 1024
    hop: int = 256

    def __post_init__(self) -> None:
        if self.window < 2 or self.hop < 1:
            raise ValueError("spectrogram window must be at least 2 and hop at least 1")
        if self.hop > self.window:
            raise ValueError("spectrogram hop must not exceed the window")


def pass_spectrogram(
    force: np.ndarray, settings: SpectrogramSettings, height: int, width: int
) -> np.ndarray:
    """(samples, 3) force -> (3, height, width) float32 standardized log-power image."""
    if force.ndim != 2 or force.shape[1] != 3:
        raise ValueError(f"force must have shape (samples, 3), got {force.shape}")
    if len(force) < settings.window:
        raise ValueError(
            f"a pass of {len(force)} samples is shorter than the spectrogram window "
            f"{settings.window}; spectrograms need the raw signal (they are computed before pooling)"
        )
    signal = torch.from_numpy(np.ascontiguousarray(force.T, dtype=np.float32))
    signal = signal - signal.mean(dim=1, keepdim=True)  # the static cutting force is not a spectral feature
    transform = torch.stft(
        signal,
        n_fft=settings.window,
        hop_length=settings.hop,
        window=torch.hann_window(settings.window),
        center=False,
        return_complex=True,
    )
    log_power = torch.log10(transform.abs().square() + 1e-12)
    image = F.interpolate(
        log_power[None], size=(height, width), mode="bilinear", align_corners=False, antialias=True
    )[0]
    mean = image.mean(dim=(1, 2), keepdim=True)
    std = image.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return ((image - mean) / std).numpy().astype(np.float32)


class SpectrogramBank:
    """Spectrogram images keyed by each pass's photo path, a drop-in for ``ImageBank``.

    Keying by the photo path reuses the timelines' image codes, so image lag and the
    per-step image indices work exactly as for photos.
    """

    def __init__(
        self,
        raw_records: Sequence[Record],
        settings: SpectrogramSettings,
        height: int = 128,
        width: int = 384,
    ) -> None:
        unique = {Path(record.image_path): record for record in raw_records}
        self.settings = settings
        self.height, self.width = height, width
        self.index = {path: position for position, path in enumerate(unique)}
        self.pixels = np.empty((len(unique), 3, height, width), dtype=np.float16)
        for position, record in enumerate(unique.values()):
            self.pixels[position] = pass_spectrogram(record.force, settings, height, width)

    def __len__(self) -> int:
        return len(self.index)

    def tensor(self, path: Path) -> Tensor:
        if Path(path) == NO_IMAGE:
            return torch.zeros(3, self.height, self.width)
        return torch.from_numpy(self.pixels[self.index[Path(path)]].astype(np.float32))
