"""Logistic smooth-to-hard singular-value proximal calculations."""

from __future__ import annotations

import numpy as np
from functools import lru_cache
from scipy.optimize import minimize_scalar
from scipy.special import lambertw


@lru_cache(maxsize=None)
def maximum_slope(a: float) -> float:
    if a <= 0:
        raise ValueError("a must be positive")

    def negative_derivative(x: float) -> float:
        probability = 1.0 / (1.0 + np.exp(-a * (x - 1.0)))
        return -(probability + a * x * probability * (1.0 - probability))

    upper = max(4.0, 1.0 + 20.0 / a)
    result = minimize_scalar(negative_derivative, bounds=(0.0, upper), method="bounded")
    if not result.success:
        raise RuntimeError("maximum-slope optimization failed")
    return float(-result.fun)


@lru_cache(maxsize=None)
def weak_convexity(a: float) -> float:
    slope = maximum_slope(a)
    return float(1.0 - 1.0 / slope)


def inverse_shrinker(u, a: float, lam: float):
    values = np.asarray(u, dtype=float)
    argument = (a * values / lam) * np.exp(a - a * values / lam)
    residual = (lam / a) * np.real(lambertw(argument, k=0))
    return values + residual


def scaled_prox_singular_values(
    singular_values,
    a: float,
    lam: float,
    beta: float,
    tolerance: float = 1e-10,
    maximum_iterations: int = 100,
):
    values = np.asarray(singular_values, dtype=float)
    rho = weak_convexity(a)
    if beta * rho >= 1.0:
        raise ValueError(f"scaled prox is not unique: beta*rho={beta * rho}")
    if beta == 0:
        return values.copy(), 0.0, rho
    lower = np.zeros_like(values)
    upper = values.copy()
    for _ in range(maximum_iterations):
        midpoint = 0.5 * (lower + upper)
        mapped = midpoint + beta * (inverse_shrinker(midpoint, a, lam) - midpoint)
        below = mapped < values
        lower = np.where(below, midpoint, lower)
        upper = np.where(below, upper, midpoint)
        scale = max(1.0, float(np.max(values, initial=0.0)))
        if np.max(upper - lower, initial=0.0) <= tolerance * scale:
            break
    result = 0.5 * (lower + upper)
    residual = result + beta * (inverse_shrinker(result, a, lam) - result) - values
    return result, float(np.max(np.abs(residual), initial=0.0)), rho


def spectral_scaled_prox(matrix, a: float, lam: float, beta: float, xp, tolerance: float):
    left, singular_values, right_h = xp.linalg.svd(matrix, full_matrices=False)
    singular_cpu = np.asarray(
        singular_values.get() if hasattr(singular_values, "get") else singular_values
    )
    updated_cpu, root_residual, rho = scaled_prox_singular_values(
        singular_cpu, a, lam, beta, tolerance=tolerance
    )
    updated = xp.asarray(updated_cpu, dtype=singular_values.dtype)
    return (left * updated[None, :]) @ right_h, root_residual, rho
