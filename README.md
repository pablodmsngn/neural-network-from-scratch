# Neural Network from Scratch in NumPy

**Manual backpropagation — no PyTorch, no TensorFlow, no autograd. Just NumPy and the chain rule.**

A complete training stack (layers, losses, optimizers, trainer, gradient checking) implemented from first principles and trained on MNIST. Every gradient is derived by hand and *proved correct* against finite differences before any training happens.

## Results

| Experiment | Result |
|---|---|
| Gradient check, 784-128-64-10 MLP on real MNIST data | max relative error **3.7e-07** |
| MNIST 5k-sample subset (4k train / 1k test), SGD+momentum, 20 epochs | **93.8%** test acc, ~2 s on CPU |
| MNIST 5k-sample subset, Adam(1e-3), 20 epochs | **92.1%** test acc |
| Full MNIST (60k/10k), same architecture | typically **97–98%** — run the notebook to reproduce |

All numbers above were produced by the code in this repo with `SEED = 42`. The subset results are from the verification environment; the notebook downloads full MNIST when run in Colab.

## How to run

**Google Colab (recommended):** upload `neural_network_from_scratch.ipynb` (`File ▸ Upload notebook`), then `Runtime ▸ Run all`. Runs in a few minutes on the free CPU runtime. No setup: the only dependencies are `numpy` and `matplotlib` (preinstalled), with `scikit-learn` as a data-download fallback.

**Locally:** `pip install numpy matplotlib scikit-learn`, then open the notebook in Jupyter.

## Reproducibility

- A single `numpy.random.Generator` seeded with `SEED = 42` drives **all** randomness: weight initialization, minibatch shuffling, and gradient-check sampling. Two runs with the same seed produce bit-identical weights (asserted in tests).
- Python/NumPy versions are printed in the first cell.
- MNIST is fetched from two public mirrors with an OpenML fallback, and parsed from the raw IDX format with `np.frombuffer` — no framework involved.

## Design

The code follows SOLID principles because they map naturally onto how a network is composed:

| Principle | Where |
|---|---|
| Single Responsibility | `Layer` computes, `Loss` scores, `Optimizer` updates, `Trainer` orchestrates |
| Open/Closed | New layer/loss/optimizer = one new subclass; nothing else changes |
| Liskov Substitution | Any `Layer` works inside `Sequential`; any `Optimizer` inside `Trainer` |
| Interface Segregation | Minimal contracts: `forward`/`backward`, `step` |
| Dependency Inversion | `Trainer` depends on abstractions, never on `Dense` or `Adam` |

Patterns: **Strategy** (swappable losses/optimizers), **Composite** (`Sequential`), **Template Method** (ABCs fix the contract, subclasses fill in the math).

## What's implemented

- Layers: `Dense` (He/Xavier init), `ReLU`, `Tanh`, `Sigmoid`
- Losses: fused `SoftmaxCrossEntropy` (log-sum-exp stable), `MSE`
- Optimizers: `SGD` with momentum, `Adam` with bias correction
- `Sequential` model, `Trainer` with history, batched evaluation
- `gradient_check`: central finite differences on randomly sampled parameters

## Verification

A network with subtly wrong gradients still trains — just worse. That is why correctness here is *proved*, not assumed: central-difference gradient checking on every layer type and on the actual MNIST architecture yields relative errors near 1e-7 (float64 noise floor for ε = 1e-5). See `REPORT.md` for the full write-up.

## Companion project

[`../mini-autodiff-engine`](../mini-autodiff-engine) — the same MLP, but gradients come from a hand-built computation graph (reverse-mode autodiff) instead of hand-derived formulas. The two implementations agree to 1e-16.

## Roadmap

Dropout and L2 regularization, BatchNorm (a classic backward-derivation exercise), Conv2D via im2col, learning-rate schedules.

## License

MIT
