# Backpropagation by Hand: What I Learned Building a Neural Network in Pure NumPy

*Project report / blog post — companion to `neural_network_from_scratch.ipynb`*

## Why do this in 2026?

Frameworks make backpropagation invisible: you write `loss.backward()` and gradients appear. That is exactly why implementing it by hand is still the best exercise in deep learning — it forces you to derive every gradient, confront numerical stability yourself, and design a verification strategy, because nothing will fail loudly if your math is subtly wrong. This project implements a full training stack in NumPy — layers, losses, optimizers, a trainer, and gradient checking — and trains a multilayer perceptron on MNIST.

## Design decisions

**Backward passes return input gradients and store parameter gradients.** Each layer's `backward(dL/dy)` returns `dL/dx` for the layer below and caches `dW`, `db` internally. This makes backpropagation through the whole model a three-line loop over `reversed(layers)`, and keeps the optimizer decoupled from the network topology: it only ever sees flat, aligned lists of parameters and gradients.

**SOLID as an engineering constraint, not decoration.** The training loop (`Trainer`) depends only on three abstract interfaces: `Layer` (forward/backward), `Loss` (forward/backward), and `Optimizer` (step). The payoff was concrete during development: I compared Adam against SGD+momentum, and later swapped architectures, without touching a line of the trainer. Adding Dropout or BatchNorm later means writing one class with a `forward` and a `backward` — nothing else changes. Strategy, Composite, and Template Method patterns fall out of this naturally rather than being imposed.

**Fused softmax + cross-entropy.** I fused them into one `Loss` class for two reasons. Numerically, the log-sum-exp trick (subtracting the row max before exponentiating) makes the loss exact for arbitrarily large logits; my first unfused version produced NaNs within a few epochs at higher learning rates. Analytically, the fused gradient collapses to `(softmax − onehot)/N`, which is both simpler and better conditioned than chaining two Jacobians.

**One seeded generator for everything.** All randomness — weight init, minibatch shuffling, gradient-check sampling — flows from a single `numpy.random.default_rng(42)`. The test suite asserts that two runs with the same seed produce bit-identical weights. Reproducibility was a design requirement from day one, not an afterthought.

## Verification: the part that actually matters

A network with wrong gradients often still trains, just worse — silent failure is the default failure mode of hand-written backprop. So before any experiment, every gradient is compared against central finite differences, `[L(θ+ε) − L(θ−ε)]/2ε`, which has O(ε²) error. With ε = 1e-5 in float64, a correct implementation should show relative errors near 1e-7 or below.

Measured results (seed 42):

| Check | Max relative error |
|---|---|
| 3-layer MLP, ReLU+Tanh, softmax cross-entropy | 1.7e-09 |
| MLP with Sigmoid + MSE | 6.4e-09 |
| The real 784-128-64-10 architecture on real MNIST pixels | 3.7e-07 |

Checking the *actual* architecture on *actual* data matters: bugs love to hide in shapes and scales that toy checks never exercise.

## Experiments and results

Architecture: 784 → 128 → 64 → 10 with ReLU (~109k parameters), batch size 128, 20 epochs, identical He initialization for both optimizers (same seed — the optimizer is the only variable).

On the 5,000-sample balanced MNIST subset used in the verification environment (4,000 train / 1,000 test):

| Optimizer | Test accuracy | Wall time (CPU) |
|---|---|---|
| SGD(lr=0.1, momentum=0.9) | **93.8%** | 1.8 s |
| Adam(lr=1e-3) | 92.1% | 2.9 s |

Two observations worth reporting. First, Adam converged faster early (85.3% after one epoch vs 80.6% for SGD) but SGD+momentum overtook it by epoch 3 and kept a consistent ~1.5-point lead — consistent with the folklore that Adam's per-parameter step sizes help early progress while well-tuned SGD generalizes slightly better on simple architectures. Second, SGD was also *faster per epoch* (no moment bookkeeping). On full MNIST (60k/10k), which the notebook downloads and trains when run in Colab, this architecture typically lands at 97–98% test accuracy.

The error analysis in the notebook shows the expected confusion structure: most residual errors are 4↔9, 3↔5, and 7↔2 — pairs that are genuinely ambiguous at 28×28.

## Honest limitations

The Trainer holds the whole dataset in memory (fine for MNIST, wrong for anything big — a `DataLoader` abstraction is the fix). There is no regularization yet, so the model overfits the small subset (final train loss 0.001 vs test loss ~0.25). Dense layers only — convolutions via im2col are the natural next step, and BatchNorm's backward pass is the best derivation exercise I know of.

## What I actually learned

That numerical stability is part of the math, not an implementation detail; that broadcasting in the forward pass always means summation in the backward pass (the bias gradient is a row-sum *because* the bias was broadcast); and that gradient checking transforms backprop from an act of faith into an engineering discipline. The companion project — a [mini autodiff engine](../mini-autodiff-engine) that computes these same gradients from a computation graph — agrees with this hand-derived implementation to 1e-16, which is the most satisfying assertion in either test suite.

---
*All numbers reproducible with `SEED = 42`; environment versions are printed in the notebook's first cell.*
