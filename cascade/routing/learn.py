"""The two small models the predictive router needs, in plain numpy: deterministic and typed.

- ``LogisticModel``: L2-regularized logistic regression fitted by Newton's method. ``C`` is the
  inverse regularization strength (smaller is stronger): it minimizes
  ``sum(log-loss) + ||w||^2 / (2C)`` over standardized features, with an unpenalized intercept. The
  penalty also keeps the fit finite when a class is all-success or all-failure, which happens with a
  few dozen tasks.
- ``RidgeLogCost``: ridge regression of log(cost) with Duan's smearing factor, so the predicted cost
  is an estimate of the mean cost and not of the geometric mean.

No randomness anywhere: the same inputs give the same coefficients on every run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float64]
_COST_FLOOR = 1e-9


@dataclass(frozen=True)
class Standardizer:
    mean: Matrix
    scale: Matrix

    @classmethod
    def fit(cls, x: Matrix) -> Standardizer:
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)  # a constant column becomes all zeros
        return cls(mean, scale)

    def transform(self, x: Matrix) -> Matrix:
        return (x - self.mean) / self.scale


def _sigmoid(z: Matrix) -> Matrix:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


@dataclass(frozen=True)
class LogisticModel:
    scaler: Standardizer
    weights: Matrix
    intercept: float

    @classmethod
    def fit(
        cls, x: Matrix, y: NDArray[np.bool_], c: float, *, max_iter: int = 100, tol: float = 1e-10
    ) -> LogisticModel:
        if c <= 0:
            raise ValueError("C must be positive")
        if len(x) != len(y):
            raise ValueError("x and y differ in length")
        scaler = Standardizer.fit(x)
        z = scaler.transform(x)
        n, d = z.shape
        design = np.hstack([z, np.ones((n, 1))])
        target = y.astype(np.float64)
        penalty = np.full(d + 1, 1.0 / c)
        penalty[-1] = 1e-9  # the intercept is (almost) unpenalized; the epsilon keeps H invertible
        beta = np.zeros(d + 1)
        for _ in range(max_iter):
            p = _sigmoid(design @ beta)
            grad = design.T @ (p - target) + penalty * beta
            hess = design.T @ (design * (p * (1 - p))[:, None]) + np.diag(penalty)
            step = np.linalg.solve(hess, grad)
            beta = beta - step
            if float(np.max(np.abs(step))) < tol:
                break
        return cls(scaler, beta[:-1].copy(), float(beta[-1]))

    def predict_proba(self, x: Matrix) -> Matrix:
        return _sigmoid(self.scaler.transform(x) @ self.weights + self.intercept)


@dataclass(frozen=True)
class RidgeLogCost:
    scaler: Standardizer
    weights: Matrix
    intercept: float
    smear: float

    @classmethod
    def fit(cls, x: Matrix, cost: Matrix, lam: float) -> RidgeLogCost:
        if lam < 0:
            raise ValueError("lambda must not be negative")
        scaler = Standardizer.fit(x)
        z = scaler.transform(x)
        y = np.log(np.maximum(cost, _COST_FLOOR))
        y_mean = float(y.mean())
        d = z.shape[1]
        w: Matrix = np.asarray(
            np.linalg.solve(z.T @ z + lam * np.eye(d), z.T @ (y - y_mean)), dtype=np.float64
        )
        resid = y - (z @ w + y_mean)
        return cls(scaler, w, y_mean, float(np.mean(np.exp(resid))))

    def predict(self, x: Matrix) -> Matrix:
        return np.exp(self.scaler.transform(x) @ self.weights + self.intercept) * self.smear
