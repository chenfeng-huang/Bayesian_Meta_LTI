import numpy as np
import torch

from bayes_lti.model import MetaParams, posterior, kl_matrix_normal


def test_posterior_shapes_and_spd():
    torch.manual_seed(0)
    n, T = 3, 6
    X = torch.randn(n, T, dtype=torch.get_default_dtype())
    Y = torch.randn(n, T, dtype=torch.get_default_dtype())

    params = MetaParams(n)
    V = params.V
    W = params.W

    M, Vm = posterior(X, Y, V, W)

    # Shape checks
    assert M.shape == (n, n)
    assert Vm.shape == (n, n)

    # Symmetry and SPD checks (via eigenvalues > 0)
    assert torch.allclose(Vm, Vm.mH, atol=1e-8)
    evals = torch.linalg.eigvalsh(Vm)
    assert torch.all(evals > 0)


def test_kl_nonnegative():
    torch.manual_seed(1)
    n, T = 3, 6
    X = torch.randn(n, T, dtype=torch.get_default_dtype())
    Y = torch.randn(n, T, dtype=torch.get_default_dtype())

    params = MetaParams(n)
    V = params.V
    W = params.W
    sigma2 = params.sigma2

    M, Vm = posterior(X, Y, V, W)
    kl = kl_matrix_normal(M, Vm, W, V, sigma2)
    assert kl.item() >= -1e-8
