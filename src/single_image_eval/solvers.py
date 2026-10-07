"""Proposed, Soft, Hard, and L+S solvers for the single-image model."""

from __future__ import annotations

import time

from .operator import adjoint, forward, operator_norm_squared
from .spectral import spectral_scaled_prox, weak_convexity


def _to_matrix(image, xp):
    frames, nx, ny = image.shape
    return xp.transpose(image, (1, 2, 0)).reshape(nx * ny, frames)


def _from_matrix(matrix, shape, xp):
    frames, nx, ny = shape
    return xp.transpose(matrix.reshape(nx, ny, frames), (2, 0, 1))


def largest_singular_value(matrix, xp) -> float:
    gram = xp.conj(matrix).T @ matrix
    return float(xp.sqrt(xp.maximum(xp.real(xp.linalg.eigvalsh(gram)[-1]), 0)))


def singular_value_prox(matrix, threshold: float, kind: str, xp):
    if threshold < 0 or kind not in {"soft", "hard"}:
        raise ValueError("invalid spectral threshold request")
    gram = xp.conj(matrix).T @ matrix
    eigenvalues, right_vectors = xp.linalg.eigh(gram)
    order = xp.argsort(eigenvalues)[::-1]
    eigenvalues = xp.maximum(xp.real(eigenvalues[order]), 0)
    right_vectors = right_vectors[:, order]
    values = xp.sqrt(eigenvalues)
    updated = xp.maximum(values - threshold, 0) if kind == "soft" else values * (values > threshold)
    ratios = xp.where(values > 0, updated / values, 0)
    projected = matrix @ right_vectors
    return (projected * ratios[None, :]) @ xp.conj(right_vectors).T, values, updated


def complex_soft_threshold(array, threshold: float, xp):
    magnitude = xp.abs(array)
    epsilon = xp.finfo(magnitude.dtype).eps
    return array * xp.maximum(1.0 - threshold / xp.maximum(magnitude, epsilon), 0)


def solve_proposed(
    y,
    mask,
    a: float,
    c: float,
    alpha: float,
    xp,
    relative_tolerance: float = 1e-5,
    root_tolerance: float = 1e-9,
    maximum_iterations: int = 300,
    initial_image=None,
):
    started = time.perf_counter()
    lipschitz = operator_norm_squared(mask, xp)
    rho = weak_convexity(a)
    bounds = [1.0 / lipschitz]
    if rho > 0:
        bounds.append(1.0 / (alpha * rho))
    eta = 0.95 * min(bounds)
    beta = eta * alpha
    image = initial_image.copy() if initial_image is not None else adjoint(y, mask, xp)
    sigma_one = largest_singular_value(_to_matrix(image, xp), xp)
    lam = c * sigma_one
    if not lam > 0:
        raise ValueError("lambda is not positive")
    history = []
    root_residual = float("nan")
    for iteration in range(1, maximum_iterations + 1):
        residual = forward(image, mask, xp) - y
        trial = image - eta * adjoint(residual, mask, xp)
        updated_matrix, root_residual, _ = spectral_scaled_prox(
            _to_matrix(trial, xp), a, lam, beta, xp, root_tolerance
        )
        updated = _from_matrix(updated_matrix, image.shape, xp)
        relative_change = float(
            xp.linalg.norm((updated - image).ravel())
            / xp.maximum(1.0, xp.linalg.norm(image.ravel()))
        )
        if iteration == 1 or iteration % 10 == 0 or relative_change <= relative_tolerance:
            history.append(
                {
                    "iteration": iteration,
                    "relative_change": relative_change,
                    "data_loss": 0.5 * float(xp.linalg.norm(residual.ravel()) ** 2),
                    "root_residual": root_residual,
                }
            )
        image = updated
        if relative_change <= relative_tolerance:
            break
    return image, {
        "a": a,
        "c": c,
        "alpha": alpha,
        "lambda": lam,
        "sigma1_initial": sigma_one,
        "lipschitz": lipschitz,
        "rho": rho,
        "eta": eta,
        "beta": beta,
        "beta_rho": beta * rho,
        "iterations": iteration,
        "converged": relative_change <= relative_tolerance,
        "final_relative_change": relative_change,
        "final_root_residual": root_residual,
        "runtime_seconds": time.perf_counter() - started,
        "history": history,
    }


def solve_spectral_baseline(
    y,
    mask,
    normalized_threshold: float,
    kind: str,
    xp,
    relative_tolerance: float = 1e-5,
    maximum_iterations: int = 300,
    initial_image=None,
):
    if normalized_threshold < 0:
        raise ValueError("threshold must be nonnegative")
    started = time.perf_counter()
    lipschitz = operator_norm_squared(mask, xp)
    eta = 0.95 / lipschitz
    image = initial_image.copy() if initial_image is not None else adjoint(y, mask, xp)
    sigma_one = largest_singular_value(_to_matrix(image, xp), xp)
    threshold = normalized_threshold * sigma_one
    objective_weight = threshold / eta if kind == "soft" else threshold**2 / (2.0 * eta)
    history = []
    for iteration in range(1, maximum_iterations + 1):
        residual = forward(image, mask, xp) - y
        trial = image - eta * adjoint(residual, mask, xp)
        updated_matrix, _, updated_values = singular_value_prox(
            _to_matrix(trial, xp), threshold, kind, xp
        )
        updated = _from_matrix(updated_matrix, image.shape, xp)
        relative_change = float(
            xp.linalg.norm((updated - image).ravel())
            / xp.maximum(1.0, xp.linalg.norm(image.ravel()))
        )
        if iteration == 1 or iteration % 10 == 0 or relative_change <= relative_tolerance:
            history.append({"iteration": iteration, "relative_change": relative_change})
        image = updated
        if relative_change <= relative_tolerance:
            break
    return image, {
        "kind": kind,
        "normalized_threshold": normalized_threshold,
        "threshold": threshold,
        "objective_weight": objective_weight,
        "sigma1_initial": sigma_one,
        "lipschitz": lipschitz,
        "eta": eta,
        "iterations": iteration,
        "converged": relative_change <= relative_tolerance,
        "final_relative_change": relative_change,
        "runtime_seconds": time.perf_counter() - started,
        "history": history,
    }


def solve_l_plus_s(
    y,
    mask,
    normalized_low_rank_threshold: float,
    normalized_sparse_threshold: float,
    xp,
    relative_tolerance: float = 1e-5,
    maximum_iterations: int = 300,
    initial_image=None,
):
    if normalized_low_rank_threshold < 0 or normalized_sparse_threshold < 0:
        raise ValueError("thresholds must be nonnegative")
    started = time.perf_counter()
    lipschitz = operator_norm_squared(mask, xp)
    eta = 0.95 / (2.0 * lipschitz)
    initial = initial_image.copy() if initial_image is not None else adjoint(y, mask, xp)
    low_rank = initial.copy()
    sparse = xp.zeros_like(initial)
    sigma_one = largest_singular_value(_to_matrix(initial, xp), xp)
    sparse_scale = float(xp.max(xp.abs(xp.fft.fft(initial, axis=0, norm="ortho"))))
    low_threshold = normalized_low_rank_threshold * sigma_one
    sparse_threshold = normalized_sparse_threshold * sparse_scale
    history = []
    for iteration in range(1, maximum_iterations + 1):
        image = low_rank + sparse
        residual = forward(image, mask, xp) - y
        gradient = adjoint(residual, mask, xp)
        updated_matrix, _, _ = singular_value_prox(
            _to_matrix(low_rank - eta * gradient, xp), low_threshold, "soft", xp
        )
        updated_low = _from_matrix(updated_matrix, low_rank.shape, xp)
        transformed = xp.fft.fft(sparse - eta * gradient, axis=0, norm="ortho")
        updated_sparse = xp.fft.ifft(
            complex_soft_threshold(transformed, sparse_threshold, xp), axis=0, norm="ortho"
        )
        numerator = xp.sqrt(
            xp.linalg.norm((updated_low - low_rank).ravel()) ** 2
            + xp.linalg.norm((updated_sparse - sparse).ravel()) ** 2
        )
        denominator = xp.maximum(
            1.0,
            xp.sqrt(xp.linalg.norm(low_rank.ravel()) ** 2 + xp.linalg.norm(sparse.ravel()) ** 2),
        )
        relative_change = float(numerator / denominator)
        if iteration == 1 or iteration % 10 == 0 or relative_change <= relative_tolerance:
            history.append({"iteration": iteration, "relative_change": relative_change})
        low_rank, sparse = updated_low, updated_sparse
        if relative_change <= relative_tolerance:
            break
    return low_rank + sparse, {
        "normalized_low_rank_threshold": normalized_low_rank_threshold,
        "normalized_sparse_threshold": normalized_sparse_threshold,
        "low_rank_threshold": low_threshold,
        "sparse_threshold": sparse_threshold,
        "sigma1_initial": sigma_one,
        "sparse_scale_initial": sparse_scale,
        "lipschitz": lipschitz,
        "eta": eta,
        "iterations": iteration,
        "converged": relative_change <= relative_tolerance,
        "final_relative_change": relative_change,
        "runtime_seconds": time.perf_counter() - started,
        "history": history,
    }
