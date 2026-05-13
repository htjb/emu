"""Serialisation utilities for saving and loading trained emulators."""

import io
import json
import pickle
import zipfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from astroemu._version import __version__
from astroemu.dataloaders import SpectrumDataset


def save(
    path: str,
    params: dict,
    train_losses: list[float],
    val_losses: list[float],
    hidden_size: int,
    nlayers: int,
    loss: str,
    train_dataset: SpectrumDataset,
    val_dataset: SpectrumDataset,
    test_dataset: SpectrumDataset,
    act: str = "relu",
    epochs: int = 1000,
    patience: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> None:
    """Save a trained emulator to a .astroemu file.

    The .astroemu extension is appended automatically if not present.
    The file is a zip archive containing:
      - config.json  : hyperparameters, training history, loss criterion,
                       code version, and dataset configurations.
      - params.npz   : network weight arrays.
      - pipeline.pkl : pickled normalisation pipeline instances for the
                       training, validation, and test datasets.

    Args:
        path (str): Destination path. The .astroemu extension is appended
            if not already present, e.g. "emulator" → "emulator.astroemu".
        params (dict): Trained network parameters returned by train().
        train_losses (list[float]): Per-epoch training losses.
        val_losses (list[float]): Per-epoch validation losses.
        hidden_size (int): Number of nodes in each hidden layer.
        nlayers (int): Number of hidden layers.
        act (str): Activation function name used during training.
        epochs (int): Max epochs used during training.
        patience (int): Early stopping patience used during training.
        learning_rate (float): AdamW learning rate used during training.
        weight_decay (float): AdamW weight decay used during training.
        loss (str): Name of the loss criterion used.
        train_dataset (SpectrumDataset): Training dataset whose file paths
            and pipeline are saved.
        val_dataset (SpectrumDataset): Validation dataset whose file paths
            and pipeline are saved.
        test_dataset (SpectrumDataset): Test dataset whose file paths
            and pipeline are saved.
    """
    if not path.endswith(".astroemu"):
        path = path + ".astroemu"

    config: dict = {
        "version": __version__,
        "hyperparams": {
            "hidden_size": hidden_size,
            "nlayers": nlayers,
            "act": act,
            "epochs": epochs,
            "patience": patience,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
        "loss": loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    def _dataset_config(ds: SpectrumDataset) -> dict:
        return {
            "files": ds.files,
            "x": ds.x,
            "y": ds.y,
            "variable_input": ds.varied_input,
            "tiling": ds.tiling,
            "allow_pickle": ds.allow_pickle,
        }

    config["train_dataset"] = _dataset_config(train_dataset)
    config["val_dataset"] = _dataset_config(val_dataset)
    config["test_dataset"] = _dataset_config(test_dataset)

    pipelines = {
        "train": train_dataset.forward_pipeline,
        "val": val_dataset.forward_pipeline,
        "test": test_dataset.forward_pipeline,
    }

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.json", json.dumps(config, indent=2))

        buf = io.BytesIO()
        np.savez(buf, **{k: np.array(v) for k, v in params.items()})
        zf.writestr("params.npz", buf.getvalue())

        zf.writestr("pipeline.pkl", pickle.dumps(pipelines))


def load(path: str) -> dict:
    """Load a trained emulator from a .astroemu file.

    For each saved dataset, this function attempts to reconstruct a
    SpectrumDataset using the saved file paths and pipeline. If any files
    are missing the raw config dict is returned
    under the dataset key instead.

    Args:
        path (str): Path to a .astroemu file.

    Returns:
        dict: Dictionary with the following keys:

            - **params** (dict): Network weight arrays as jnp.ndarrays.
            - **hyperparams** (dict): Architecture and training
              hyperparameters.
            - **train_losses** (list[float]): Per-epoch training losses.
            - **val_losses** (list[float]): Per-epoch validation losses.
            - **loss** (str): Name of the loss criterion used.
            - **version** (str): astroemu version the emulator was trained
              with.
            - **train_pipeline** (list): Normalisation pipeline instances
              used for the training dataset.
            - **val_pipeline** (list): Normalisation pipeline instances
              used for the validation dataset.
            - **test_pipeline** (list): Normalisation pipeline instances
              used for the test dataset.
            - **train_dataset** (SpectrumDataset | dict): Reconstructed
              training dataset if all files are found, otherwise the raw
              config dict.
            - **val_dataset** (SpectrumDataset | dict): Reconstructed
              validation dataset if all files are found, otherwise the raw
              config dict.
            - **test_dataset** (SpectrumDataset | dict): Reconstructed
              test dataset if all files are found, otherwise the raw config
              dict.
    """
    with zipfile.ZipFile(path, "r") as zf:
        config = json.loads(zf.read("config.json"))

        params_buf = io.BytesIO(zf.read("params.npz"))
        params_np = np.load(params_buf)
        params = {k: jnp.array(params_np[k]) for k in params_np.files}

        pipelines = pickle.loads(zf.read("pipeline.pkl"))

    result: dict = {
        "params": params,
        "hyperparams": config["hyperparams"],
        "train_losses": config["train_losses"],
        "val_losses": config["val_losses"],
        "loss": config["loss"],
        "version": config["version"],
        "train_pipeline": pipelines["train"],
        "val_pipeline": pipelines["val"],
        "test_pipeline": pipelines["test"],
    }

    for split in ("train", "val", "test"):
        key = f"{split}_dataset"
        if key not in config:
            continue

        ds_config = config[key]  # config pertaining to this dataset split

        # look for the files on the system
        files = ds_config["files"]
        missing = [f for f in files if not Path(f).exists()]

        if missing:
            # if the files can't be found
            # return the config dict for this dataset
            result[key] = ds_config
        else:
            # if all files are found, reconstruct the SpectrumDataset
            result[key] = SpectrumDataset(
                files=files,
                x=ds_config["x"],
                y=ds_config["y"],
                forward_pipeline=pipelines[split] or None,
                variable_input=ds_config["variable_input"],
                tiling=ds_config["tiling"],
                allow_pickle=ds_config["allow_pickle"],
            )

    return result
