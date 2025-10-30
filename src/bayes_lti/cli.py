from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import generate_dataset, save_dataset, load_dataset
from .train import TrainConfig, train
from .adapt import adapt, save_adaptation
from .eval import evaluate_fewshot, save_report


def main() -> None:
    parser = argparse.ArgumentParser(prog="bayes_lti")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # generate
    p_gen = sub.add_parser("generate", help="Generate synthetic dataset")
    p_gen.add_argument("--out", type=str, required=True)
    p_gen.add_argument("--n", type=int, required=True)
    p_gen.add_argument("--M-train", type=int, required=True, dest="M_train")
    p_gen.add_argument("--M-val", type=int, required=True, dest="M_val")
    p_gen.add_argument("--M-test", type=int, required=True, dest="M_test")
    p_gen.add_argument("--T-min", type=int, required=True, dest="T_min")
    p_gen.add_argument("--T-max", type=int, required=True, dest="T_max")
    p_gen.add_argument("--sigma-true", type=float, required=True, dest="sigma_true")
    p_gen.add_argument("--v-true", type=float, required=True, dest="v_true")
    p_gen.add_argument("--rho0-gen", type=float, required=True, dest="rho0_gen")
    p_gen.add_argument("--seed", type=int, default=0)

    # train
    p_train = sub.add_parser("train", help="Train meta-parameters")
    p_train.add_argument("--data", type=str, required=True)
    p_train.add_argument("--steps", type=int, default=2000)
    p_train.add_argument("--batch", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--tauW", type=float, default=5.0)
    p_train.add_argument("--lambdaV", type=float, default=1e-2)
    p_train.add_argument("--gamma", type=float, default=1.0)
    p_train.add_argument("--eta", type=float, default=1.0)
    p_train.add_argument("--rho0", type=float, default=0.98)
    p_train.add_argument("--device", type=str, default="cpu")
    p_train.add_argument("--seed", type=int, default=1)

    # adapt
    p_adapt = sub.add_parser("adapt", help="Adapt to a new task")
    p_adapt.add_argument("--ckpt", type=str, required=True)
    p_adapt.add_argument("--data-new", type=str, required=True, dest="data_new")
    p_adapt.add_argument("--task-index", type=int, required=True, dest="task_index")
    p_adapt.add_argument("--save", type=str, required=True)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate few-shot performance")
    p_eval.add_argument("--ckpt", type=str, required=True)
    p_eval.add_argument("--data", type=str, required=True)
    p_eval.add_argument("--k-steps", type=int, required=True, dest="k_steps")
    p_eval.add_argument("--fewshot", type=int, required=True)
    p_eval.add_argument("--report", type=str, required=True)

    args = parser.parse_args()

    if args.cmd == "generate":
        ds = generate_dataset(
            n=args.n,
            M_train=args.M_train,
            M_val=args.M_val,
            M_test=args.M_test,
            T_min=args.T_min,
            T_max=args.T_max,
            sigma_true=args.sigma_true,
            v_true=args.v_true,
            rho0_gen=args.rho0_gen,
            seed=args.seed,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_dataset(args.out, ds)
        print(f"Saved dataset to {args.out}")

    elif args.cmd == "train":
        cfg = TrainConfig(
            data=args.data,
            steps=args.steps,
            batch=args.batch,
            lr=args.lr,
            tauW=args.tauW,
            lambdaV=args.lambdaV,
            gamma=args.gamma,
            eta=args.eta,
            rho0=args.rho0,
            device=args.device,
            seed=args.seed,
        )
        ckpt = train(cfg)
        print(f"Saved checkpoint to {ckpt}")

    elif args.cmd == "adapt":
        data = load_dataset(args.data_new)
        t = data["test"][args.task_index] if args.task_index < len(data["test"]) else data["train"][0]
        out = adapt(args.ckpt, t["X"], t["Y"])
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        save_adaptation(args.save, out)
        print(f"Saved adaptation to {args.save}")

    elif args.cmd == "eval":
        report = evaluate_fewshot(args.ckpt, args.data, fewshot=args.fewshot, k_steps=args.k_steps)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        save_report(args.report, report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
