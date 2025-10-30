from __future__ import annotations

from typing import Dict, List
import numpy as np


def _spectral_radius_np(A: np.ndarray) -> float:
    vals = np.linalg.eigvals(A)
    return float(np.max(np.abs(vals)))


def _make_stable_matrix(n: int, rho0: float, rng: np.random.Generator) -> np.ndarray:
    G = rng.normal(0.0, 1.0 / n, size=(n, n))
    rhoG = _spectral_radius_np(G)
    if rhoG < 1e-12:
        scale = 1.0
    else:
        scale = 1.0 / max(rhoG, 1.0)
    alpha = 0.9 * rho0
    W = alpha * G * scale
    return W


def generate_dataset(
    n: int,
    M_train: int,
    M_val: int,
    M_test: int,
    T_min: int,
    T_max: int,
    sigma_true: float,
    v_true: float,
    rho0_gen: float,
    seed: int,
) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """Generate synthetic tasks and trajectories.

    Args:
        n: state dimension.
        M_train: number of training tasks.
        M_val: number of validation tasks.
        M_test: number of test tasks.
        T_min: minimum trajectory length.
        T_max: maximum trajectory length.
        sigma_true: observation/process noise std (Σ = σ^2 I).
        v_true: column covariance scalar for matrix-normal prior (V★ = v_true I).
        rho0_gen: maximum spectral radius enforced for sampled A_m.
        seed: RNG seed.

    Returns:
        Dict with keys 'train', 'val', 'test', each a list of dicts with X,Y,A_true.
    """
    assert T_min >= 2 and T_max >= T_min
    rng = np.random.default_rng(seed)

    # True meta-parameters
    W_star = _make_stable_matrix(n, rho0_gen, rng)
    V_star = (v_true) * np.eye(n)
    sigma2_star = float(sigma_true ** 2)

    def sample_task() -> Dict[str, np.ndarray]:
        # Sample A ~ MN(W*, Σ*, V*) with Σ* = σ^2 I
        L_row = np.sqrt(sigma2_star) * np.eye(n)
        L_col = np.linalg.cholesky(V_star)
        Z = rng.normal(size=(n, n))
        B = L_row @ Z @ L_col.T  # row-covariance then column-covariance
        A = W_star + B
        # Optional stability enforcement
        rhoA = _spectral_radius_np(A)
        if rhoA > rho0_gen:
            A = A * (rho0_gen / (rhoA + 1e-12))

        # Rollout trajectory
        T = int(rng.integers(T_min, T_max + 1))
        x = rng.normal(size=(n,))
        X = np.zeros((n, T))
        Y = np.zeros((n, T))
        for t in range(T):
            X[:, t] = x
            noise = rng.normal(scale=np.sqrt(sigma2_star), size=(n,))
            y = A @ x + noise
            Y[:, t] = y
            x = y
        return {"X": X, "Y": Y, "A_true": A}

    def make_split(M: int) -> List[Dict[str, np.ndarray]]:
        return [sample_task() for _ in range(M)]

    data = {
        "train": make_split(M_train),
        "val": make_split(M_val),
        "test": make_split(M_test),
    }
    return data


def save_dataset(path: str, data: Dict) -> None:
    """Save dataset dict-of-lists to npz file."""
    # Convert to a serializable form
    out: Dict[str, object] = {}
    for split, tasks in data.items():
        out[f"{split}_len"] = len(tasks)
        for i, t in enumerate(tasks):
            out[f"{split}_{i}_X"] = t["X"]
            out[f"{split}_{i}_Y"] = t["Y"]
            out[f"{split}_{i}_A_true"] = t["A_true"]
    np.savez(path, **out)


def load_dataset(path: str) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """Load dataset saved by save_dataset."""
    npz = np.load(path, allow_pickle=False)
    result: Dict[str, List[Dict[str, np.ndarray]]] = {"train": [], "val": [], "test": []}
    for split in ["train", "val", "test"]:
        key_len = f"{split}_len"
        if key_len not in npz:
            continue
        M = int(npz[key_len])
        for i in range(M):
            X = npz[f"{split}_{i}_X"]
            Y = npz[f"{split}_{i}_Y"]
            A = npz[f"{split}_{i}_A_true"]
            result[split].append({"X": X, "Y": Y, "A_true": A})
    return result
