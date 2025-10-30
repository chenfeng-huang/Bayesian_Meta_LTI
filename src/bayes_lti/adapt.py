from __future__ import annotations

from typing import Dict
import json
import numpy as np
import torch

from .model import MetaParams, posterior


def _load_params(ckpt: str, device: str = "cpu") -> MetaParams:
    data = torch.load(ckpt, map_location=device)
    n = int(data["n"]) if "n" in data else int(data["state_dict"]["W"].shape[-1])
    params = MetaParams(n).to(device)
    params.load_state_dict(data["state_dict"])  # type: ignore[arg-type]
    params.eval()
    return params


def adapt(ckpt: str, X_new: np.ndarray, Y_new: np.ndarray) -> Dict[str, np.ndarray]:
    """Few-shot adaptation on a new task using posterior closed forms.

    Args:
        ckpt: path to checkpoint saved by training.
        X_new: (n,T) numpy array for new task inputs/states.
        Y_new: (n,T) numpy array for new task next-states.
    Returns:
        Dict with A_hat (posterior mean) and covariance diagonal of vec(A).
    """
    device = "cpu"
    params = _load_params(ckpt, device=device)

    X = torch.from_numpy(X_new).to(device=device, dtype=torch.get_default_dtype())
    Y = torch.from_numpy(Y_new).to(device=device, dtype=torch.get_default_dtype())

    with torch.no_grad():
        M, Vm = posterior(X, Y, params.V, params.W)
        sigma2 = float(params.sigma2.item())
        n = X.shape[0]
        # diag(Vm ⊗ σ^2 I) = σ^2 * kron(diag(Vm), ones(n))
        diag_Vm = torch.diag(Vm)
        cov_diag = torch.kron(diag_Vm, torch.ones(n, dtype=diag_Vm.dtype, device=diag_Vm.device)) * sigma2
        return {
            "A_hat": M.cpu().numpy(),
            "cov_diag": cov_diag.cpu().numpy(),
        }


def save_adaptation(path: str, out: Dict[str, np.ndarray]) -> None:
    serial = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in out.items()}
    with open(path, "w") as f:
        json.dump(serial, f)
