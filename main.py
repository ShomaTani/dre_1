"""Minimal D3-style density-ratio estimation demo.

Paper inspiration:
Direct Density-Ratio Estimation with Dimensionality Reduction

This is a didactic implementation, not a reproduction of every optimization in
the original paper. The core flow is preserved:

1. Build a numerator sample set p(x) and denominator sample set q(x).
2. Find a low-dimensional subspace where the two distributions differ.
3. Estimate the density ratio r(x) = p(x) / q(x) directly in that subspace.

To keep the code easy to run, we use only the Python standard library.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from generate import CovariateShiftDataset, build_covariate_shift_dataset


Vector = List[float]
Matrix = List[List[float]]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def mean_vector(samples: Sequence[Sequence[float]]) -> Vector:
    dim = len(samples[0])
    acc = [0.0] * dim
    for x in samples:
        for i, value in enumerate(x):
            acc[i] += value
    return [value / len(samples) for value in acc]


def dot(x: Sequence[float], y: Sequence[float]) -> float:
    return sum(xi * yi for xi, yi in zip(x, y))


def squared_distance(x: Sequence[float], y: Sequence[float]) -> float:
    return sum((xi - yi) ** 2 for xi, yi in zip(x, y))


def vector_sub(x: Sequence[float], y: Sequence[float]) -> Vector:
    return [xi - yi for xi, yi in zip(x, y)]


def add_ridge(matrix: Matrix, ridge: float) -> Matrix:
    out = [row[:] for row in matrix]
    for i in range(len(out)):
        out[i][i] += ridge
    return out


def solve_linear_system(matrix: Matrix, rhs: Sequence[float]) -> Vector:
    """Gaussian elimination with partial pivoting."""

    n = len(matrix)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(a[r][col]))
        pivot_value = a[pivot_row][col]
        if abs(pivot_value) < 1e-12:
            raise ValueError("Linear system is singular or ill-conditioned.")
        if pivot_row != col:
            a[col], a[pivot_row] = a[pivot_row], a[col]

        scale = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= scale

        for row in range(col + 1, n):
            factor = a[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                a[row][j] -= factor * a[col][j]

    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        solution[row] = a[row][n] - sum(
            a[row][j] * solution[j] for j in range(row + 1, n)
        )
    return solution


def fisher_lda_direction(
    numerator_samples: Sequence[Sequence[float]],
    denominator_samples: Sequence[Sequence[float]],
    ridge: float = 1e-3,
) -> Vector:
    """Two-class LDA direction used as an easy-to-read D3-style subspace.

    The original paper learns a hetero-distributional subspace jointly with
    density-ratio estimation. For a lightweight runnable demo, we use the
    classic direction that emphasizes where the two sample sets differ.
    """

    mean_p = mean_vector(numerator_samples)
    mean_q = mean_vector(denominator_samples)
    dim = len(mean_p)
    scatter = zeros(dim, dim)

    for group, group_mean in (
        (numerator_samples, mean_p),
        (denominator_samples, mean_q),
    ):
        for x in group:
            diff = vector_sub(x, group_mean)
            for i in range(dim):
                for j in range(dim):
                    scatter[i][j] += diff[i] * diff[j]

    direction = solve_linear_system(add_ridge(scatter, ridge), vector_sub(mean_p, mean_q))
    norm = math.sqrt(dot(direction, direction))
    return [value / norm for value in direction]


def project_onto_direction(
    samples: Sequence[Sequence[float]], direction: Sequence[float]
) -> List[Vector]:
    return [[dot(x, direction)] for x in samples]


def median_heuristic_sigma(
    numerator_samples: Sequence[Sequence[float]],
    denominator_samples: Sequence[Sequence[float]],
) -> float:
    merged = list(numerator_samples) + list(denominator_samples)
    distances = []
    limit = min(50, len(merged))
    for i in range(limit):
        for j in range(i + 1, limit):
            distances.append(math.sqrt(squared_distance(merged[i], merged[j])))
    distances.sort()
    if not distances:
        return 1.0
    return max(distances[len(distances) // 2], 1e-3)


def rbf(sample: Sequence[float], center: Sequence[float], sigma: float) -> float:
    return math.exp(-squared_distance(sample, center) / (2.0 * sigma * sigma))


@dataclass
class ULSIFRatioEstimator:
    centers: List[Vector]
    sigma: float
    alpha: Vector

    def predict_one(self, sample: Sequence[float]) -> float:
        value = sum(
            weight * rbf(sample, center, self.sigma)
            for weight, center in zip(self.alpha, self.centers)
        )
        return max(value, 0.0)

    def predict(self, samples: Sequence[Sequence[float]]) -> List[float]:
        return [self.predict_one(x) for x in samples]


def average_prediction(
    estimator: ULSIFRatioEstimator, samples: Sequence[Sequence[float]]
) -> float:
    predictions = estimator.predict(samples)
    return sum(predictions) / len(predictions)


def empirical_ulsif_objective(
    estimator: ULSIFRatioEstimator,
    numerator_samples: Sequence[Sequence[float]],
    denominator_samples: Sequence[Sequence[float]],
) -> float:
    pred_num = estimator.predict(numerator_samples)
    pred_den = estimator.predict(denominator_samples)
    first = 0.5 * sum(value * value for value in pred_den) / len(pred_den)
    second = sum(pred_num) / len(pred_num)
    return first - second


def split_train_validation(
    samples: Sequence[Sequence[float]], validation_fraction: float = 0.3
) -> Tuple[List[Vector], List[Vector]]:
    cut = max(1, int(len(samples) * (1.0 - validation_fraction)))
    return list(samples[:cut]), list(samples[cut:])


def fit_ulsif(
    numerator_samples: Sequence[Sequence[float]],
    denominator_samples: Sequence[Sequence[float]],
    num_centers: int = 30,
    regularization: float = 1e-3,
    seed: int = 0,
    sigma: float | None = None,
) -> ULSIFRatioEstimator:
    """Fit a tiny uLSIF-style density-ratio estimator with RBF bases."""

    rng = random.Random(seed)
    sigma = sigma or median_heuristic_sigma(numerator_samples, denominator_samples)
    centers = list(numerator_samples)
    rng.shuffle(centers)
    centers = centers[: min(num_centers, len(centers))]

    b = len(centers)
    h = [0.0] * b
    H = zeros(b, b)

    for i, center_i in enumerate(centers):
        h[i] = sum(rbf(x, center_i, sigma) for x in numerator_samples) / len(
            numerator_samples
        )
        for j, center_j in enumerate(centers):
            H[i][j] = sum(
                rbf(x, center_i, sigma) * rbf(x, center_j, sigma)
                for x in denominator_samples
            ) / len(denominator_samples)

    alpha = solve_linear_system(add_ridge(H, regularization), h)
    alpha = [max(value, 0.0) for value in alpha]
    estimator = ULSIFRatioEstimator(centers=centers, sigma=sigma, alpha=alpha)

    # Importance ratios should average to 1 under the denominator distribution.
    den_mean = average_prediction(estimator, denominator_samples)
    if den_mean > 1e-12:
        estimator.alpha = [value / den_mean for value in estimator.alpha]
    return estimator


def fit_ulsif_with_model_selection(
    numerator_samples: Sequence[Sequence[float]],
    denominator_samples: Sequence[Sequence[float]],
    num_centers: int = 30,
    seed: int = 0,
) -> ULSIFRatioEstimator:
    train_num, val_num = split_train_validation(numerator_samples)
    train_den, val_den = split_train_validation(denominator_samples)
    base_sigma = median_heuristic_sigma(train_num, train_den)

    candidates = []
    for sigma_scale in (0.5, 1.0, 2.0):
        for regularization in (1e-3, 1e-2, 1e-1, 1.0):
            sigma = max(base_sigma * sigma_scale, 1e-3)
            estimator = fit_ulsif(
                train_num,
                train_den,
                num_centers=num_centers,
                regularization=regularization,
                seed=seed,
                sigma=sigma,
            )
            score = empirical_ulsif_objective(estimator, val_num, val_den)
            candidates.append((score, sigma, regularization))

    candidates.sort(key=lambda item: item[0])
    _, best_sigma, best_regularization = candidates[0]
    return fit_ulsif(
        numerator_samples,
        denominator_samples,
        num_centers=num_centers,
        regularization=best_regularization,
        seed=seed,
        sigma=best_sigma,
    )


def mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def summarize_top_coordinates(direction: Sequence[float], k: int = 5) -> List[Tuple[int, float]]:
    indexed = list(enumerate(direction))
    indexed.sort(key=lambda item: abs(item[1]), reverse=True)
    return indexed[:k]


def run_demo(dataset: CovariateShiftDataset) -> None:
    direction = fisher_lda_direction(
        dataset.numerator_train,
        dataset.denominator_train,
    )

    numerator_projected = project_onto_direction(dataset.numerator_train, direction)
    denominator_projected = project_onto_direction(dataset.denominator_train, direction)
    evaluation_projected = project_onto_direction(dataset.evaluation, direction)

    plain_estimator = fit_ulsif_with_model_selection(
        dataset.numerator_train, dataset.denominator_train
    )
    d3_estimator = fit_ulsif_with_model_selection(
        numerator_projected, denominator_projected
    )

    plain_predictions = plain_estimator.predict(dataset.evaluation)
    d3_predictions = d3_estimator.predict(evaluation_projected)

    plain_mse = mean_squared_error(dataset.true_ratios, plain_predictions)
    d3_mse = mean_squared_error(dataset.true_ratios, d3_predictions)

    print("=== Theme ===")
    print("Covariate shift in 10 dimensions, but only the first 2 dimensions differ.")
    print("Goal: estimate r(x) = p(x) / q(x) directly without estimating p and q themselves.")
    print()

    print("=== D3-style pipeline ===")
    print("1. Prepare numerator samples p(x) and denominator samples q(x).")
    print("2. Find a low-dimensional direction where the two sample sets differ.")
    print("3. Run direct density-ratio estimation only in that projected space.")
    print()

    print("=== Learned subspace direction (top coordinates) ===")
    for index, weight in summarize_top_coordinates(direction):
        print(f"dimension {index:2d}: weight = {weight:+.4f}")
    print()

    print("=== Evaluation against the true density ratio ===")
    print(f"Baseline uLSIF in full 10D space    MSE: {plain_mse:.4f}")
    print(f"D3-style projection + uLSIF in 1D   MSE: {d3_mse:.4f}")
    print()

    print("=== Sample predictions ===")
    for i in range(5):
        print(
            f"sample {i:2d} | true={dataset.true_ratios[i]:7.4f} "
            f"| full={plain_predictions[i]:7.4f} | d3={d3_predictions[i]:7.4f}"
        )


if __name__ == "__main__":
    data = build_covariate_shift_dataset()
    run_demo(data)
