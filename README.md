# Training and Testing Pipeline

This document describes the full pipeline for Bayesian meta‑learning of LTI system identification implemented in this repository.

## Overview

- Goal: Learn a hierarchical prior over task dynamics matrices so that we can adapt quickly to new tasks with few observations.
- Model: Each task m is an LTI system with states $x_t \in \mathbb{R}^n$ and dynamics
  $$ x_{t+1} = A_m x_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \Sigma), \; \Sigma = \sigma^2 I_n. $$
- Likelihood and prior (matrix‑normal):
  - $Y \mid A \sim \mathcal{MN}(A X, \Sigma, I_T)$
  - $A \mid \phi \sim \mathcal{MN}(W, \Sigma, V)$ with meta‑parameters $\phi=(W,V,\sigma^2)$
- Conjugate posterior for a task (closed form):
  - $V_m = (V^{-1} + X X^\top)^{-1}$
  - $M_m = (W V^{-1} + Y X^\top) V_m$
  - $q_m(A) = \mathcal{MN}(M_m, \Sigma, V_m)$


## Environment Setup

```bash
python -m pip install -r requirements.txt

pip install -e .
```

## Data generation (synthetic)

- We generate a dataset with train/val/test splits. True meta‑parameters are:
  - $W_\star$: stable by scaling a Gaussian matrix to spectral radius $\alpha = 0.9\,\rho_0$.
  - $V_\star = v_{\text{true}} I_n$.
  - $\Sigma_\star = \sigma_{\text{true}}^2 I_n$.
- For each task:
  1) Sample $A_m \sim \mathcal{MN}(W_\star, \Sigma_\star, V_\star)$.
  2) Optionally rescale $A_m$ to enforce spectral radius $\le \rho_0^{\text{gen}}$.
  3) Sample $x_0 \sim \mathcal{N}(0, I_n)$ and roll a trajectory of random length $T \in [T_{\min}, T_{\max}]$.
  4) Build $X=[x_0,\dots,x_{T-1}]$, $Y=[x_1,\dots,x_T]$.
- Saved as `dataset.npz` with lists of tasks per split.

CLI:
```bash
python -m bayes_lti.cli generate --out data/dataset.npz --n 4 --M-train 200 --M-val 40 --M-test 40 \
  --T-min 8 --T-max 16 --sigma-true 0.1 --v-true 0.5 --rho0-gen 0.95 --seed 123
```

## Meta‑training

- Parameters to learn: $\phi=(W,V,\sigma^2)$.
- Parameterization:
  - $V = L L^\top + \alpha I$ with unconstrained $L$ (lower‑triangular used) and $\alpha=\mathrm{softplus}(a)+10^{-6}$.
  - $\sigma^2 = \mathrm{softplus}(s) + 10^{-5}$.
  - $W$ is free; after each optimizer step we project $W$ to spectral radius $\le \rho_0$ by uniform scaling if needed.
- For a task minibatch $\mathcal{B}$, we compute posteriors $\{(M_m, V_m)\}$ and minimize the PAC‑Bayes surrogate:
  $$ L(\phi) = \mathbb{E}_{m\in\mathcal{B}}\big[ E_{\text{fit},m} + \lambda_m \, \mathrm{KL}(q_m\Vert p_\phi) \big] + \gamma\,\mathrm{HyperReg}(\phi) + \eta\,R_{\text{stab}}(W), $$
  with $\lambda_m = 1/T_m$.

### Terms

- Expected fit (drop constants):
  $$ E_{\text{fit},m} = \tfrac{1}{2} \Big[ T\log|\Sigma| + \mathrm{tr}(\Sigma^{-1}(Y-M_m X)(Y-M_m X)^\top) + n\,\mathrm{tr}(X X^\top V_m) \Big]. $$
  With $\Sigma=\sigma^2 I$: $\log|\Sigma|=n\log\sigma^2$ and $\mathrm{tr}(\Sigma^{-1}\cdot) = \|Y-M_m X\|_F^2/\sigma^2$.

- KL between matrix‑normals sharing row covariance $\Sigma$:
  $$ \mathrm{KL}(q_m\Vert p_\phi) = \tfrac{1}{2} \Big[ n\log(|V|/|V_m|) - n^2 + n\mathrm{tr}(V^{-1}V_m)
       + \mathrm{vec}(M_m-W)^\top (V^{-1}\otimes \Sigma^{-1})\mathrm{vec}(M_m-W) \Big]. $$
  With $\Sigma=\sigma^2 I$, the last term simplifies to $\tfrac{1}{\sigma^2} \, \mathrm{tr}((M_m-W) V^{-1} (M_m-W)^\top)$ without materializing the Kronecker product.

- Hyper‑regularizer:
  $$ \mathrm{HyperReg}(\phi) = \tfrac{1}{2\tau_W^2}\|W\|_F^2 + \lambda_V\,\big( \tfrac{1}{2}\|V-I\|_F^2 - \log\det V \big). $$

- Stability penalty:
  $$ R_{\text{stab}}(W) = \max(0, \rho(W) - \rho_0)^2, $$
  where $\rho(\cdot)$ is spectral radius (we use spectral norm as a differentiable surrogate in the loss; hard projection uses true radius post‑step).

### Optimization

- Optimizer: Adam with cosine annealing LR schedule.
- Minibatching: uniform over training tasks.
- Determinism: seeds for NumPy and PyTorch; deterministic cuDNN when available.
- Early stopping: monitor validation bound on held‑out tasks; save best checkpoint to `runs/last.ckpt`.
- Logging: per‑step CSV with loss, fit, KL, $\|W\|_F$, $\log\det V$, $\sigma^2$, $\rho(W)$, and validation bound.

CLI:
```bash
python -m bayes_lti.cli train --data data/dataset.npz --steps 2000 --batch 32 --lr 1e-3 \
  --tauW 5.0 --lambdaV 1e-2 --gamma 1.0 --eta 1.0 --rho0 0.98 --device cpu --seed 1
```

## Adaptation (test‑time)

- Given a new task with a few observations $D_{\text{new}}=(X_{\text{new}}, Y_{\text{new}})$, we compute the posterior using the learned meta‑prior:
  - $V_{\text{new}} = (V^{-1} + X_{\text{new}} X_{\text{new}}^\top)^{-1}$
  - $M_{\text{new}} = (W V^{-1} + Y_{\text{new}} X_{\text{new}}^\top) V_{\text{new}}$
  - $q_{\text{new}}(A) = \mathcal{MN}(M_{\text{new}}, \sigma^2 I, V_{\text{new}})$
- Point estimate: $\hat{A} = M_{\text{new}}$ (posterior mean = MAP for Gaussian case).
- Uncertainty: $\mathrm{Cov}(\mathrm{vec}(A)) = V_{\text{new}} \otimes (\sigma^2 I)$; we report its diagonal as a summary.

CLI:
```bash
python -m bayes_lti.cli adapt --ckpt runs/last.ckpt --data-new data/dataset.npz --task-index 0 --save outputs/new_task.json
```

## Evaluation

- Few‑shot evaluation uses the first $T_{\text{new}}$ points of each test task to adapt, then computes k‑step ahead MSE starting from $t=T_{\text{new}}-1$.
- Baselines: pooled ridge over all training tasks and a trivial zero‑$A$ model.
- Report: mean and std of MSE across test tasks, plus baseline metrics.

CLI:
```bash
python -m bayes_lti.cli eval --ckpt runs/last.ckpt --data data/dataset.npz --fewshot 5 --k-steps 5 --report outputs/report.json
```


