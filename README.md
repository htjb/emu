# astroemu — Next Generation Emulators for Cosmology and Astrophysics

[![version](https://img.shields.io/badge/version-0.4.2-blue)](https://github.com/htjb/astroemu)
[![docs](https://readthedocs.org/projects/astroemu/badge/?version=latest)](https://astroemu.readthedocs.io/en/latest/)
[![license](https://img.shields.io/github/license/htjb/astroemu)](https://github.com/htjb/astroemu/blob/main/LICENSE)

> **Under active development** — interfaces may change between versions.

`astroemu` is a generalised framework for emulating spectral signals in
cosmology and astrophysics, inspired by the
[`globalemu`](https://github.com/htjb/globalemu) package. Neural network
emulators are implemented in JAX, with an optax-based training loop and
PyTorch-style dataloaders.

## Installation

```bash
pip install astroemu
```

## How it works

The core idea (shared with `globalemu`) is to input independent variables
alongside physical model parameters, predicting a single spectral value per
call. Full spectra are recovered via a vectorised call over all $x$ points.

Given a signal $y = f(x, \theta)$ with $N$ parameter samples and $m$
independent variable points, the training data is tiled so that parameters
and independent variables are concatenated as inputs:

| Input | Output |
|---|---|
| $[\theta_0,\ x_0]$ | $y_0$ |
| $[\theta_0,\ x_1]$ | $y_1$ |
| $[\theta_0,\ \ldots]$ | $\ldots$ |
| $[\theta_0,\ x_m]$ | $y_m$ |
| $[\ldots,\ \ldots]$ | $\ldots$ |
| $[\theta_N,\ x_m]$ | $y_m$ |

For more details see the `globalemu`
[paper](https://arxiv.org/abs/2104.04336). A paper demonstrating
applications of `astroemu` to a broad range of astrophysical signals is in
preparation.

## Documentation

```bash
git clone git@github.com:htjb/astroemu.git
pip install ".[docs]"
mkdocs serve
```

or browse the hosted docs at
[astroemu.readthedocs.io](https://astroemu.readthedocs.io/en/latest/).

## Contributions

Contributions are welcome! Please open an issue to discuss and read the
contribution guidelines before submitting a pull request.
