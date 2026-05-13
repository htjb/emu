# Tutorial: Training Your First Emulator

This tutorial walks through training a neural network emulator on 21-cm power
spectra from the [Zeus21](https://github.com/JulianBMunoz/Zeus21) code by
Julian Muñoz. We will use the 100 example spectra shipped with `astroemu` and
cover the full workflow: data loading, normalisation, training, inference, and
saving/loading.

## The data

Each `.npz` file contains one 21-cm power spectrum evaluated at a fixed
redshift ($z = 15$) over 54 wavenumber values. The files contain the following
keys:

| Key | Description |
|---|---|
| `k` | Wavenumber array (independent variable $x$), shape `(54,)` |
| `power` | Power spectrum (dependent variable $y$), shape `(54,)` |
| `astro_params` | Dict of astrophysical parameters: `L40_xray`, `fesc10`, `epsstar` |
| `cosmo_params` | Dict of cosmological parameters: `h_fid` |

The emulator will learn the mapping
$[\theta, x] \to y$,
where $\theta$ concatenates the four parameters and $x$ is a single
wavenumber value.

## Setup

```python
import glob

import jax.numpy as jnp
import matplotlib.pyplot as plt

from astroemu.dataloaders import SpectrumDataset
from astroemu.network import mlp
from astroemu.normalisation import log_base_10, standardise
from astroemu.serialisation import load, save
from astroemu.train import train
from astroemu.utils import compute_mean_std
```

## Step 1: Load and split the data

```python
files = sorted(glob.glob("tests/example_data/sample_*.npz"))

train_files = files[:70]
val_files   = files[70:85]
test_files  = files[85:]

print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
```

## Step 2: Build a normalisation pipeline

Raw 21-cm power spectra span many orders of magnitude, so we apply a
$\log_{10}$ transformation to $x$, $y$, and the input parameters before
training. This is handled by the `log_base_10` pipeline.

```python
log = log_base_10(log_all_x=True, log_all_y=True, log_all_params=True)
```

We then standardise (zero mean, unit variance) using statistics computed
from the training set. To avoid loading all spectra at once, we use
`compute_mean_std`, which streams through the data in batches.

Note that we create this dataset with `tiling=False` so that
`compute_mean_std` receives `(specs, x, params)` tuples rather than the
tiled inputs used for training.

```python
train_ds_stats = SpectrumDataset(
    files=train_files,
    x="k",
    y="power",
    variable_input=["astro_params", "cosmo_params"],
    forward_pipeline=log,
    tiling=False,
    allow_pickle=True,
)

mean_spec, std_spec, mean_x, std_x, mean_params, std_params = compute_mean_std(
    train_ds_stats.get_batch_iterator(batch_size=32, shuffle=False)
)

standardiser = standardise(
    y_mean=mean_spec,
    y_std=std_spec,
    x_mean=mean_x,
    x_std=std_x,
    params_mean=mean_params,
    params_std=std_params,
    standardise_x=True,
    standardise_params=True,
)
```

## Step 3: Create training, validation, and test datasets

With the normalisation pipeline in hand, we create three `SpectrumDataset`
instances with `tiling=True` (the default). In tiling mode the dataset
restructures each spectrum into one training example per wavenumber point,
so a batch of $n$ spectra with $m = 54$ wavenumber values yields
$n \times m$ input–output pairs.

```python
pipeline = [log, standardiser]

train_dataset = SpectrumDataset(
    files=train_files,
    x="k",
    y="power",
    variable_input=["astro_params", "cosmo_params"],
    forward_pipeline=pipeline,
    tiling=True,
    allow_pickle=True,
)
val_dataset = SpectrumDataset(
    files=val_files,
    x="k",
    y="power",
    variable_input=["astro_params", "cosmo_params"],
    forward_pipeline=pipeline,
    tiling=True,
    allow_pickle=True,
)
test_dataset = SpectrumDataset(
    files=test_files,
    x="k",
    y="power",
    variable_input=["astro_params", "cosmo_params"],
    forward_pipeline=pipeline,
    tiling=True,
    allow_pickle=True,
)
```

## Step 4: Train the emulator

We call `train()` with a small network suitable for 70 training spectra.
The training loop uses AdamW and stops early if the validation loss does
not improve for `patience` consecutive epochs.

```python
best_params, train_losses, val_losses = train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    hidden_size=32,
    nlayers=2,
    act="relu",
    epochs=500,
    patience=20,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=32,
)
```

A `tqdm` progress bar will display the training and validation loss at each
epoch. Once training finishes, plot the loss curves to check for healthy
convergence:

```python
plt.plot(train_losses, label="Train")
plt.plot(val_losses,   label="Val")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.legend()
plt.show()
```

## Step 5: Run inference on the test set

To evaluate the emulator on unseen spectra we iterate over the test dataset
and apply `mlp` directly, then undo the normalisation via each pipeline's
`backward()` method.

The `x` values are identical for every spectrum, so we read them from the
first file for use in the backward pass.

```python
_, x, _ = test_dataset[0]   # x shape: (54,)

predictions = []
true_values = []

for y_flat, inputs in test_dataset.get_batch_iterator(
    batch_size=32, shuffle=False
):
    preds = mlp(best_params, inputs, act="relu")  # (batch*54, 1)
    preds = preds.squeeze(-1)                      # (batch*54,)

    # In tiled mode, inputs has shape (batch * n_k, n_params + 1)
    # First column is x, remaining columns are params
    x_tiled = inputs[:, 0]
    params_tiled = inputs[:, 1:]

    # Undo normalisation (reverse pipeline order)
    for pipe in reversed(pipeline):
        preds, x_tiled, params_tiled = pipe.backward(
            preds, x_tiled, params_tiled
        )
        y_flat, _, _ = pipe.backward(y_flat, x_tiled, params_tiled)
 

    # Undo normalisation (reverse pipeline order)
    for pipe in reversed(pipeline):
        preds, _, _ = pipe.backward(preds, x, inputs)
        y_flat, _, _ = pipe.backward(y_flat, x, inputs)

    predictions.append(preds.reshape(-1, len(x)))
    true_values.append(y_flat.reshape(-1, len(x)))

predictions = jnp.vstack(predictions)
true_values = jnp.vstack(true_values)
```

Plot a handful of test spectra alongside their emulated counterparts:

```python
fig, ax = plt.subplots()
for i in range(5):
    ax.loglog(x, true_values[i],  color="k", lw=1, label="True" if i == 0 else None)
    ax.loglog(x, predictions[i],  color="r", lw=1, ls="--", label="Emulated" if i == 0 else None)
ax.set_xlabel(r"$k\ [\mathrm{Mpc}^{-1}]$")
ax.set_ylabel(r"$\Delta^2_{21}\ [\mathrm{mK}^2]$")
ax.legend()
plt.show()
```

## Step 6: Save and reload the emulator

`save()` writes a self-contained `.astroemu` file (a zip archive) containing
the network weights, hyperparameters, training history, and dataset
configurations. The `.astroemu` extension is appended automatically if
omitted.

```python
save(
    "my_emulator",           # saved as my_emulator.astroemu
    best_params,
    train_losses,
    val_losses,
    hidden_size=32,
    nlayers=2,
    act="relu",
    epochs=500,
    patience=20,
    learning_rate=1e-3,
    weight_decay=1e-4,
    loss="mse",
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
)
```

To reload the emulator later:

```python
loaded = load("my_emulator.astroemu")

params   = loaded["params"]
act      = loaded["hyperparams"]["act"]
pipeline = loaded["train_pipeline"]
```

The returned dictionary also contains `train_losses`, `val_losses`,
`hyperparams`, `version`, and reconstructed `SpectrumDataset` instances for
each split (provided the original data files are still accessible).
