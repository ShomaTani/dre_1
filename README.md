# dre_1
Direct Density Ratio Estimation with Dimensionality Reduction paper deployed


-----------------------------------------------------------------------------------
Implementation of

Sugiyama, M. et al.
"Direct Density Ratio Estimation with Dimensionality Reduction"

This repository contains an independent implementation for educational purposes.

## What this code does

This project implements a small, runnable, educational version of the paper's
main idea:

1. compare two sample sets `p(x)` and `q(x)`,
2. find a low-dimensional subspace where they differ,
3. estimate the density ratio `r(x) = p(x) / q(x)` directly in that subspace.

To keep the example easy to read and runnable everywhere, the code uses only
the Python standard library.

## Files

- `generate.py`: builds a synthetic covariate-shift dataset.
- `main.py`: runs a D3-style pipeline and compares it with a plain full-space
  uLSIF baseline.

## Simplifications

This is not a line-by-line reproduction of the full paper.

- The hetero-distributional subspace search is simplified to a two-sample LDA
  direction.
- The density-ratio estimator is a compact uLSIF-style RBF model.
- The demo uses synthetic Gaussian data so that the true density ratio is known
  and we can measure the error directly.

## Run

```bash
python main.py
```

You should see:

- which coordinates were identified as important,
- the baseline full-dimensional error,
- the D3-style projected error,
- a few example predictions.
