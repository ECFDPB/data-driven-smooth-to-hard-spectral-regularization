"""Single-image Cartesian Fourier forward model after coil combination."""

from __future__ import annotations

from .fft import fft2c, ifft2c


def _expanded_mask(mask):
    if mask.ndim != 2:
        raise ValueError("mask must have layout [time, ky]")
    return mask[:, None, :]


def forward(image, mask, xp):
    """A(X) = P_Omega F_xy X for an image series [time,x,y]."""

    return fft2c(image, xp) * _expanded_mask(mask)


def adjoint(kspace, mask, xp):
    """A*(y) = F_xy^H P_Omega y."""

    return ifft2c(kspace * _expanded_mask(mask), xp)


def operator_norm_squared(mask, xp=None) -> float:
    # An orthonormal FFT followed by a non-empty binary projection has norm 1.
    return 1.0 if bool(mask.any()) else 0.0
