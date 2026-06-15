# EJOR

This repository contains the Python code associated with the arXiv version of the paper:

**Shared Infrastructure Investment and Pricing: Stackelberg Equilibria in Risk-Aware Take-or-Pay Contracts**
Amal Sakr, Andrea Araldo, Tamer Başar, and Tijani Chahed.

**Paper:** [arXiv:2606.12167](https://arxiv.org/abs/2606.12167)

## Overview

The code implements the risk-aware Stackelberg game studied in the paper. The Infrastructure Provider (InP) acts as the leader and jointly optimizes infrastructure capacity dimensioning and access pricing. Firms act as followers that share the infrastructure and commit upfront to future resource usage under uncertain revenues. Followers’ heterogeneous risk aversion is modeled through Conditional Value-at-Risk (CVaR), with firm-side operational and congestion costs.

## Repository structure

├── main.py
├── problem_setup.py
├── follower_solver.py
├── leader_optimization.py
├── postprocessing.py
├── probability.py
├── figure_leader.py
├── figure_proba.py
└── requirements.txt

## Main files

* `problem_setup.py`: builds the model parameters, generates stochastic samples, and computes the CVaR-adjusted matrix.
* `follower_solver.py`: solves the lower-level follower equilibrium for fixed capacity and access price.
* `leader_optimization.py`: solves the upper-level InP capacity-dimensioning and access-pricing problem.
* `postprocessing.py`: computes equilibrium summaries and follower-level performance measures.
* `probability.py`: computes the probabilistic profitability guarantees.
* `figure_leader.py`: generates the leader-side figures.
* `figure_proba.py`: generates the probability-guarantee figures.
* `main.py`: runs the full numerical experiment.

## Installation



## Running the experiments

To run the full numerical experiment:

```bash
python main.py
```

The script generates the numerical results and the data used to produce the figures.

## Outputs

The code produces equilibrium results, including the optimal capacity, optimal access price, InP profit, follower resource commitments, utilization levels, CVaR values, and profitability probability guarantees.

It also generates PDF figures for the leader-side results and the probability-guarantee analysis.

## Citation

If you use this code, please cite the associated arXiv paper:

```bibtex
@misc{sakr2026shared,
  title        = {Shared Infrastructure Investment and Pricing: Stackelberg Equilibria in Risk-Aware Take-or-Pay Contracts},
  author       = {Sakr, Amal and Araldo, Andrea and Başar, Tamer and Chahed, Tijani},
  year         = {2026},
  eprint       = {2606.12167},
  archivePrefix = {arXiv}
}
```
