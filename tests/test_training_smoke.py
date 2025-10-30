import os
import csv
import tempfile

import numpy as np

from bayes_lti.data import generate_dataset, save_dataset
from bayes_lti.train import TrainConfig, train


def test_training_loss_decreases():
    tmpdir = tempfile.TemporaryDirectory()
    data_path = os.path.join(tmpdir.name, "dataset.npz")
    runs_dir = os.path.join(tmpdir.name, "runs")

    ds = generate_dataset(
        n=3,
        M_train=16,
        M_val=4,
        M_test=4,
        T_min=5,
        T_max=8,
        sigma_true=0.1,
        v_true=0.5,
        rho0_gen=0.95,
        seed=42,
    )
    save_dataset(data_path, ds)

    cfg = TrainConfig(
        data=data_path,
        steps=50,
        batch=8,
        lr=5e-3,
        tauW=5.0,
        lambdaV=1e-2,
        gamma=1.0,
        eta=1.0,
        rho0=0.98,
        device="cpu",
        seed=1,
        out_dir=runs_dir,
        patience=1000,
    )

    ckpt = train(cfg)

    # Read training log and check loss decreased
    log_csv = os.path.join(runs_dir, "train_log.csv")
    losses = []
    with open(log_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            losses.append(float(row["loss"]))
    assert len(losses) > 5
    assert losses[-1] <= 0.9 * losses[0]

    tmpdir.cleanup()
