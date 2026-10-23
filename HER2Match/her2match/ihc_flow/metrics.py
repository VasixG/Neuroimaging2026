from __future__ import annotations
import math
import numpy as np
from scipy import linalg
from skimage.metrics import structural_similarity

def pixel_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} != {target.shape}")
    mse = float(np.mean(((prediction.astype(np.float64) - target) / 255) ** 2))
    return {
        "psnr_db": -10 * math.log10(mse) if mse > 0 else None,
        "ssim": float(structural_similarity(prediction, target, data_range=255, channel_axis=-1)),
        "mae": float(np.abs(prediction.astype(np.float64) - target).mean() / 255),
        "perfect_match": mse == 0,
    }


def frechet_distance(real: np.ndarray, fake: np.ndarray) -> float:
    if len(real) < 2 or len(fake) < 2:
        raise ValueError("FID needs at least two images per distribution")
    real, fake = real.astype(np.float64), fake.astype(np.float64)
    mean_real, mean_fake = real.mean(0), fake.mean(0)
    if max(len(real), len(fake)) < real.shape[1]:
        # Equivalent covariance trace via singular values; efficient for N < 2048.
        a = (real - mean_real) / np.sqrt(len(real) - 1)
        b = (fake - mean_fake) / np.sqrt(len(fake) - 1)
        covariance_trace = np.linalg.svd(a @ b.T, compute_uv=False).sum()
        return float(
            np.square(mean_real - mean_fake).sum()
            + np.square(a).sum()
            + np.square(b).sum()
            - 2 * covariance_trace
        )
    cov_real, cov_fake = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    root = linalg.sqrtm(cov_real @ cov_fake)
    if not np.isfinite(root).all():
        offset = np.eye(cov_real.shape[0]) * 1e-6
        root = linalg.sqrtm((cov_real + offset) @ (cov_fake + offset))
    if np.iscomplexobj(root):
        if not np.allclose(np.diagonal(root).imag, 0, atol=1e-3):
            raise ValueError("FID covariance square root has a significant imaginary component")
        root = root.real
    difference = mean_real - mean_fake
    return float(
        difference @ difference + np.trace(cov_real) + np.trace(cov_fake) - 2 * np.trace(root)
    )


def kernel_distance(
    real: np.ndarray, fake: np.ndarray, subset_size: int = 100, subsets: int = 100, seed: int = 0
) -> dict:
    count = min(subset_size, len(real), len(fake))
    if count < 2 or subsets < 1:
        raise ValueError("KID needs at least two images and a positive number of subsets")
    generator = np.random.default_rng(seed)
    scores = []
    for _ in range(subsets):
        x = real[generator.choice(len(real), count, replace=False)].astype(np.float64)
        y = fake[generator.choice(len(fake), count, replace=False)].astype(np.float64)
        xx, yy, xy = (
            (x @ x.T / x.shape[1] + 1) ** 3,
            (y @ y.T / y.shape[1] + 1) ** 3,
            (x @ y.T / x.shape[1] + 1) ** 3,
        )
        value = (xx.sum() - np.trace(xx) + yy.sum() - np.trace(yy)) / (
            count * (count - 1)
        ) - 2 * xy.mean()
        scores.append(value)
    return {
        "kid": float(np.mean(scores)),
        "kid_std": float(np.std(scores)),
        "kid_x1000": float(np.mean(scores) * 1000),
        "kid_subset_size": count,
        "kid_subsets": subsets,
        "kid_seed": seed,
    }
