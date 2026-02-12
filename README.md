# Bootstrap Uncertainty Analysis (Python)

This project studies the **statistical stability** of solutions of random linear systems by combining:
- **Gaussian elimination with partial pivoting**
- large-scale simulation (**K = 10,000** systems per dimension)
- **Bootstrap BCa** confidence intervals for the mean

The goal is to understand how the solution distribution behaves as the system dimension increases, and why classical CLT-based intervals may be unreliable in heavy-tailed settings.

## Problem
Given random linear systems of the form **A x = b** (with random A and b),
we simulate many solutions and analyze:
- global mean trends across dimensions
- dispersion/instability via standard deviation
- uncertainty of the mean via **Bootstrap BCa (95% CI)**

## Method
- Generate random matrices **A** and vectors **b**
- Solve using **Gaussian elimination with partial pivoting**
- Aggregate solutions across **n = 2..20**
- Build **BCa bootstrap** confidence intervals for the mean of the first component

## Repository structure
- `src/` — Python implementation
- `results/` — generated plots

## How to run
Install dependencies:
```bash
python3 -m pip install numpy matplotlib scipy
