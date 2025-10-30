from __future__ import annotations

from typing import Dict, List, Tuple
import json
import numpy as np
import torch

from .adapt import _load_params
from .data import load_dataset
from .model import posterior


def rollout(A: np.ndarray, x0: np.ndarray, T: int, sigma: float) -> np.ndarray:
    """Roll out trajectory with Gaussian noise (for simulation)."""
    n = x0.shape[0]
    X = np.zeros((n, T))
    x = x0.copy()
    for t in range(T):
        X[:, t] = x
        noise = np.random.normal(scale=sigma, size=n)
        x = A @ x + noise
    return X


def _ridge_closed_form(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    # A_hat = Y X^T (X X^T + lam I)^{-1}
    n = X.shape[0]
    XXt = X @ X.T
    A = Y @ X.T @ np.linalg.inv(XXt + lam * np.eye(n))
    return A


def _pooled_ridge(tasks: List[Tuple[np.ndarray, np.ndarray]], lam: float) -> np.ndarray:
    n = tasks[0][0].shape[0]
    XXt = np.zeros((n, n))
    YXt = np.zeros((n, n))
    for X, Y in tasks:
        XXt += X @ X.T
        YXt += Y @ X.T
    A = YXt @ np.linalg.inv(XXt + lam * np.eye(n))
    return A


def _kstep_mse(A: np.ndarray, X: np.ndarray, Y: np.ndarray, start: int, k: int) -> float:
    # Start from x_start = X[:, start]
    n = X.shape[0]
    T = X.shape[1]
    steps = min(k, T - start)
    if steps <= 0:
        return np.nan
    x_pred = X[:, start].copy()
    err = 0.0
    for i in range(steps):
        y_pred = A @ x_pred
        y_true = Y[:, start + i]
        err += float(np.sum((y_pred - y_true) ** 2))
        x_pred = y_pred
    return err / (steps * n)


def evaluate_fewshot(ckpt: str, data_path: str, fewshot: int, k_steps: int) -> Dict[str, float]:
    data = load_dataset(data_path)
    params = _load_params(ckpt, device="cpu")
    n = int(params.W.shape[0])
    sigma = float(params.sigma2.sqrt().item())

    mses: List[float] = []
    baseline_zero: List[float] = []

    # Pooled and per-task ridge baselines
    tasks_np = [(t["X"], t["Y"]) for t in data["train"]]
    A_pooled = _pooled_ridge(tasks_np, lam=1e-3)

    for t in data["test"]:
        X = t["X"]
        Y = t["Y"]
        T = X.shape[1]
        T_new = min(fewshot, T)

        # Adaptation posterior
        Xt = torch.from_numpy(X[:, :T_new])
        Yt = torch.from_numpy(Y[:, :T_new])
        with torch.no_grad():
            M, Vm = posterior(Xt, Yt, params.V, params.W)
        A_hat = M.cpu().numpy()

        # Evaluate k-step starting after few-shot segment
        start = T_new - 1
        mse = _kstep_mse(A_hat, X, Y, start=start, k=k_steps)
        mses.append(mse)

        # Zero baseline (A=0)
        mse_zero = _kstep_mse(np.zeros_like(A_hat), X, Y, start=start, k=k_steps)
        baseline_zero.append(mse_zero)

    return {
        "mse_mean": float(np.nanmean(mses)),
        "mse_std": float(np.nanstd(mses)),
        "zero_mse_mean": float(np.nanmean(baseline_zero)),
        "pooled_mse": float(np.nanmean([
            _kstep_mse(A_pooled, t["X"], t["Y"], start=min(fewshot, t["X"].shape[1]) - 1, k=k_steps)
            for t in data["test"]
        ])),
    }


def save_report(path: str, report: Dict[str, float]) -> None:
    with open(path, "w") as f:
        json.dump(report, f)
