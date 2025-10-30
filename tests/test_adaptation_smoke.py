import os
import tempfile

from bayes_lti.data import generate_dataset, save_dataset
from bayes_lti.train import TrainConfig, train
from bayes_lti.eval import evaluate_fewshot


def test_adaptation_fewshot_better_than_zero():
    tmpdir = tempfile.TemporaryDirectory()
    data_path = os.path.join(tmpdir.name, "dataset.npz")
    runs_dir = os.path.join(tmpdir.name, "runs")

    ds = generate_dataset(
        n=3,
        M_train=32,
        M_val=8,
        M_test=8,
        T_min=6,
        T_max=8,
        sigma_true=0.1,
        v_true=0.5,
        rho0_gen=0.95,
        seed=123,
    )
    save_dataset(data_path, ds)

    cfg = TrainConfig(
        data=data_path,
        steps=60,
        batch=8,
        lr=3e-3,
        tauW=5.0,
        lambdaV=1e-2,
        gamma=1.0,
        eta=1.0,
        rho0=0.98,
        device="cpu",
        seed=2,
        out_dir=runs_dir,
        patience=1000,
    )

    ckpt = train(cfg)

    rep = evaluate_fewshot(ckpt=str(ckpt), data_path=data_path, fewshot=5, k_steps=5)
    assert rep["mse_mean"] == rep["mse_mean"]  # finite
    assert rep["mse_mean"] < rep["zero_mse_mean"]

    tmpdir.cleanup()
