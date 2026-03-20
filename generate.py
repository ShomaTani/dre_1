"""Synthetic datasets for density-ratio estimation demos.

This file intentionally uses only the Python standard library so that the
example runs in a minimal environment.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence


Vector = List[float]


@dataclass
class CovariateShiftDataset:
    """Small container for a two-sample density-ratio experiment."""

    numerator_train: List[Vector]
    denominator_train: List[Vector]
    evaluation: List[Vector]
    true_ratios: List[float]
    numerator_mean: Vector
    denominator_mean: Vector


def squared_distance(x: Sequence[float], y: Sequence[float]) -> float:
    return sum((xi - yi) ** 2 for xi, yi in zip(x, y))


def gaussian_ratio_same_covariance(
    x: Sequence[float],
    numerator_mean: Sequence[float],
    denominator_mean: Sequence[float],
) -> float:
    """True p(x) / q(x) when both are Gaussians with identity covariance."""

    num = squared_distance(x, denominator_mean) - squared_distance(x, numerator_mean)
    return math.exp(0.5 * num)


def sample_gaussian(mean: Sequence[float], rng: random.Random) -> Vector:
    return [rng.gauss(mu, 1.0) for mu in mean]


def build_covariate_shift_dataset(
    seed: int = 7,
    dimension: int = 10,
    informative_dims: int = 2,
    shift: float = 0.8,
    numerator_size: int = 180,
    denominator_size: int = 180,
    evaluation_size: int = 120,
) -> CovariateShiftDataset:
    """Create a simple problem where only a low-dimensional subspace changes.

    The numerator and denominator are both standard-covariance Gaussians, but
    only the first `informative_dims` coordinates have different means. This is
    exactly the kind of setting where dimensionality reduction is helpful.
    """

    rng = random.Random(seed)
    numerator_mean = [shift if i < informative_dims else 0.0 for i in range(dimension)]
    denominator_mean = [0.0 for _ in range(dimension)]

    numerator_train = [sample_gaussian(numerator_mean, rng) for _ in range(numerator_size)]
    denominator_train = [
        sample_gaussian(denominator_mean, rng) for _ in range(denominator_size)
    ]

    evaluation = []
    true_ratios = []
    for _ in range(evaluation_size):
        if rng.random() < 0.5:
            x = sample_gaussian(numerator_mean, rng)
        else:
            x = sample_gaussian(denominator_mean, rng)
        evaluation.append(x)
        true_ratios.append(
            gaussian_ratio_same_covariance(x, numerator_mean, denominator_mean)
        )

    return CovariateShiftDataset(
        numerator_train=numerator_train,
        denominator_train=denominator_train,
        evaluation=evaluation,
        true_ratios=true_ratios,
        numerator_mean=numerator_mean,
        denominator_mean=denominator_mean,
    )
