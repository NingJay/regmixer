from __future__ import annotations

"""Lightweight regression adapters used by the local fit workflow."""

import random
from typing import Iterator

import lightgbm as lgb
import numpy as np
import torch

from regmixer.eval.law import ScalingLaw

LGBM_HPS = {
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": ["l1", "l2"],
    "seed": 42,
    "num_iterations": 10000,
    "learning_rate": 1e-2,
    "verbosity": -1,
    "early_stopping_round": 3,
}

LOG_LINEAR_INIT_LOG_C_POINTS = 6
LOG_LINEAR_RESTARTS_PER_POINT = 8
LOG_LINEAR_MAX_STEP = 40


class LightGBMRegressor:
    def __init__(self):
        self.model = lgb.LGBMRegressor(**LGBM_HPS)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        target = np.asarray(y, dtype=np.float64).reshape(-1)
        self.model = self.model.fit(
            x,
            target,
            eval_set=[(x, target)],
            eval_metric="l2",
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(x), dtype=np.float64).reshape(-1)


def mixing_law(
    x: torch.Tensor,
    param: torch.Tensor,
    **_: object,
) -> torch.Tensor:
    log_c_i = param[0]
    t_i = param[1:]
    return torch.exp(log_c_i) + torch.exp(torch.matmul(x, t_i))


def init_params_log_linear_law(
    preferred_domain_idx: int = 0,
    *,
    num_domains: int,
) -> Iterator[list[float]]:
    for log_c_i in np.linspace(-2, 1.5, LOG_LINEAR_INIT_LOG_C_POINTS):
        for _ in range(LOG_LINEAR_RESTARTS_PER_POINT):
            ts = [
                -np.random.rand() if i == preferred_domain_idx else np.random.rand() * 0.1
                for i in range(num_domains)
            ]
            yield [log_c_i] + ts


class LogLinearRegressor:
    def __init__(self):
        np.random.seed(42)
        random.seed(42)
        self.model: list[float] | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        early_stopping: float = 0.0,
        max_step: int = LOG_LINEAR_MAX_STEP,
        delta: float = 0.02,
        preferred_domain_idx: int = 0,
    ) -> None:
        target = np.asarray(y, dtype=np.float64).reshape(-1)
        scaling_law = ScalingLaw(mixing_law)
        self.model = scaling_law.fit(
            x,
            target,
            init_params_log_linear_law(preferred_domain_idx, num_domains=x.shape[-1]),
            max_step=max_step,
            delta=delta,
            eps=early_stopping,
            workers=1,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LogLinearRegressor.predict() called before fit().")
        return mixing_law(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.model, dtype=torch.float32),
        ).numpy()


def build_local_regression(
    regression_type: str,
    x: np.ndarray,
    y: np.ndarray,
) -> LightGBMRegressor | LogLinearRegressor:
    if regression_type == "lightgbm":
        regressor = LightGBMRegressor()
        regressor.fit(x, y)
        return regressor
    if regression_type == "log_linear":
        regressor = LogLinearRegressor()
        regressor.fit(x, y)
        return regressor

    raise ValueError(f"Unsupported local regression type: {regression_type}")
