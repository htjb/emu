"""Testing the serialisation utilities."""

import glob
import tempfile
from pathlib import Path

import jax.numpy as jnp
import pytest

from astroemu._version import __version__
from astroemu.dataloaders import SpectrumDataset
from astroemu.normalisation import log_base_10
from astroemu.serialisation import load, save
from astroemu.train import train

_FILES = sorted(glob.glob("tests/example_data/sample_*.npz"))
_TRAIN_FILES = _FILES[:70]
_VAL_FILES = _FILES[70:85]
_TEST_FILES = _FILES[85:]

_DATASET_KWARGS: dict = dict(
    x="k",
    y="power",
    variable_input=["astro_params", "cosmo_params"],
    tiling=True,
    allow_pickle=True,
)

# All required save() arguments (excluding path, params, losses, datasets).
_SAVE_KWARGS: dict = dict(
    hidden_size=8,
    nlayers=1,
    act="relu",
    epochs=2,
    patience=50,
    learning_rate=1e-3,
    weight_decay=1e-4,
    loss="mse",
)


def _make_datasets() -> tuple[
    SpectrumDataset, SpectrumDataset, SpectrumDataset
]:
    return (
        SpectrumDataset(files=_TRAIN_FILES, **_DATASET_KWARGS),
        SpectrumDataset(files=_VAL_FILES, **_DATASET_KWARGS),
        SpectrumDataset(files=_TEST_FILES, **_DATASET_KWARGS),
    )


def _run_short_training(
    train_ds: SpectrumDataset, val_ds: SpectrumDataset
) -> tuple[dict, list[float], list[float]]:
    return train(
        train_ds,
        val_ds,
        hidden_size=8,
        nlayers=1,
        epochs=2,
        patience=50,
        batch_size=16,
        key=0,
    )


def test_save_appends_extension() -> None:
    """save() appends .astroemu when the extension is not provided."""
    train_ds, val_ds, test_ds = _make_datasets()
    params, train_losses, val_losses = _run_short_training(train_ds, val_ds)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Pass a path without the extension.
        bare_path = str(Path(tmpdir) / "emulator")
        save(
            bare_path,
            params,
            train_losses,
            val_losses,
            **_SAVE_KWARGS,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
        )

        assert not Path(bare_path).exists(), (
            "File should not exist at the bare path."
        )
        assert Path(bare_path + ".astroemu").exists(), (
            "File should exist at the path with .astroemu appended."
        )

        # Pass a path that already has the extension — no double-extension.
        full_path = str(Path(tmpdir) / "emulator2.astroemu")
        save(
            full_path,
            params,
            train_losses,
            val_losses,
            **_SAVE_KWARGS,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
        )

        assert Path(full_path).exists(), (
            "File should exist at the path when .astroemu is already present."
        )
        assert not Path(full_path + ".astroemu").exists(), (
            "Extension should not be appended twice."
        )


def test_save_load_roundtrip_params() -> None:
    """Loaded params arrays are numerically identical to the saved ones."""
    train_ds, val_ds, test_ds = _make_datasets()
    params, train_losses, val_losses = _run_short_training(train_ds, val_ds)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "emulator.astroemu")
        save(
            path,
            params,
            train_losses,
            val_losses,
            **_SAVE_KWARGS,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
        )
        result = load(path)

    loaded_params = result["params"]
    assert set(loaded_params.keys()) == set(params.keys()), (
        "Loaded params should have the same keys as the saved params."
    )
    for key in params:
        assert jnp.allclose(loaded_params[key], params[key]), (
            f"Loaded params['{key}'] should be numerically identical."
        )


def test_save_load_roundtrip_metadata() -> None:
    """Loaded metadata matches what was saved."""
    train_ds, val_ds, test_ds = _make_datasets()
    params, train_losses, val_losses = _run_short_training(train_ds, val_ds)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "emulator.astroemu")
        save(
            path,
            params,
            train_losses,
            val_losses,
            **_SAVE_KWARGS,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
        )
        result = load(path)

    assert result["version"] == __version__, (
        "Loaded version should match the current package version."
    )
    assert result["loss"] == "mse", "Loaded loss name should match."
    assert result["train_losses"] == train_losses, (
        "Loaded train_losses should be identical."
    )
    assert result["val_losses"] == val_losses, (
        "Loaded val_losses should be identical."
    )
    hp = result["hyperparams"]
    assert hp["hidden_size"] == 8
    assert hp["nlayers"] == 1
    assert hp["act"] == "relu"
    assert hp["learning_rate"] == pytest.approx(1e-3)


def test_save_load_with_datasets() -> None:
    """Datasets are reconstructed as SpectrumDataset when files exist."""
    train_ds, val_ds, test_ds = _make_datasets()
    params, train_losses, val_losses = _run_short_training(train_ds, val_ds)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "emulator.astroemu")
        save(
            path,
            params,
            train_losses,
            val_losses,
            **_SAVE_KWARGS,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
        )
        result = load(path)

    assert isinstance(result["train_dataset"], SpectrumDataset), (
        "train_dataset should be reconstructed as a SpectrumDataset."
    )
    assert isinstance(result["val_dataset"], SpectrumDataset), (
        "val_dataset should be reconstructed as a SpectrumDataset."
    )
    assert isinstance(result["test_dataset"], SpectrumDataset), (
        "test_dataset should be reconstructed as a SpectrumDataset."
    )
    assert result["train_dataset"].files == _TRAIN_FILES
    assert result["val_dataset"].files == _VAL_FILES
    assert result["test_dataset"].files == _TEST_FILES


def test_load_missing_files_returns_config_dict() -> None:
    """Warning is raised and config dict returned when data files missing."""
    train_ds, val_ds, test_ds = _make_datasets()
    params, train_losses, val_losses = _run_short_training(train_ds, val_ds)

    fake_ds = SpectrumDataset(
        files=["/nonexistent/path/sample_000000.npz"],
        **_DATASET_KWARGS,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "emulator.astroemu")
        save(
            path,
            params,
            train_losses,
            val_losses,
            **_SAVE_KWARGS,
            train_dataset=fake_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
        )
        result = load(path)

    assert isinstance(result["train_dataset"], dict), (
        "train_dataset should fall back to a"
        + "config dict when files are missing."
    )
    assert "files" in result["train_dataset"], (
        "Config dict should contain the 'files' key."
    )


def test_save_load_with_pipeline() -> None:
    """Normalisation pipeline instances survive a save/load roundtrip."""
    pipeline = log_base_10(log_all_x=True, log_all_y=True, log_all_params=True)
    train_ds = SpectrumDataset(
        files=_TRAIN_FILES, forward_pipeline=pipeline, **_DATASET_KWARGS
    )
    val_ds = SpectrumDataset(
        files=_VAL_FILES, forward_pipeline=pipeline, **_DATASET_KWARGS
    )
    test_ds = SpectrumDataset(
        files=_TEST_FILES, forward_pipeline=pipeline, **_DATASET_KWARGS
    )
    params, train_losses, val_losses = _run_short_training(train_ds, val_ds)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "emulator.astroemu")
        save(
            path,
            params,
            train_losses,
            val_losses,
            **_SAVE_KWARGS,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
        )
        result = load(path)

    for split in ("train", "val", "test"):
        pipeline_key = f"{split}_pipeline"
        assert len(result[pipeline_key]) == 1, (
            f"Loaded {pipeline_key} should contain one pipeline instance."
        )
        assert isinstance(result[pipeline_key][0], log_base_10), (
            f"Loaded {pipeline_key} should be a log_base_10 instance."
        )
        assert result[pipeline_key][0].log_all_x is True
        assert result[pipeline_key][0].log_all_y is True
