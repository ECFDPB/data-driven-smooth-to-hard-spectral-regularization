"""Centered orthonormal FFTs used by preprocessing and reconstruction."""

from __future__ import annotations


def fft2c(array, xp):
    axes = (-2, -1)
    return xp.fft.fftshift(
        xp.fft.fft2(xp.fft.ifftshift(array, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def ifft2c(array, xp):
    axes = (-2, -1)
    return xp.fft.fftshift(
        xp.fft.ifft2(xp.fft.ifftshift(array, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )
