"""
Smoke tests for the Stage 1 training loop.

Tests run on CPU with stub data and stub models (src.models not required).
These tests verify stability (no NaN), checkpoint creation, and CSV logging.
"""

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

# Ensure project root is on path when running from any directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


class TestStage1SmokeFiveEpochs(unittest.TestCase):
    """Smoke test: 5 epochs on stub data, no NaN, checkpoint saved, CSV written."""

    def test_smoke_5_epochs(self):
        from stubs.data_stubs import write_stub_dataset
        from src.training.train_stage1 import train

        with tempfile.TemporaryDirectory() as tmp:
            # Write synthetic stub dataset
            paths = write_stub_dataset(
                base_dir=tmp,
                n_subjects=4,
                n_epochs_per_subject=40,
                n_channels_eeg=64,
                n_channels_meg=306,
                n_voxels=1000,
                n_timepoints=175,
                n_words=20,
                modalities=["eeg", "meg", "fmri"],
            )

            # Build a minimal config that overrides file paths and cuts epochs to 5
            ckpt_dir = str(Path(tmp) / "checkpoints" / "stage1")
            config_override = {
                "split_file": paths["split_path"],
                "vocab_embeddings": paths["vocab_path"],
                "processed_base_dir": paths["processed_dir"],
                "modalities": ["eeg", "meg", "fmri"],
                "embed_dim": 768,
                "eeg_channels": 64,
                "eeg_timepoints": 175,
                "meg_channels": 306,
                "meg_timepoints": 175,
                "fmri_voxels": 1000,
                "subject_embed_dim": 64,
                "n_subjects": 4,
                "lambda_cross_modal": 0.1,
                "lambda_subject_adversarial": 0.1,
                "batch_size": 8,
                "lr": 3e-4,
                "weight_decay": 1e-4,
                "n_epochs": 5,
                "val_every_n_epochs": 5,
                "checkpoint_dir": ckpt_dir,
                "log_every_n_steps": 1,
                "device": "cpu",
            }

            # Write config to temp YAML so train(config_path) can load it
            config_path = Path(tmp) / "test_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_override, f)

            # Run training
            train(str(config_path))

            # --- Assert 1: CSV log written with correct columns ---
            log_path = Path(ckpt_dir) / "train_log.csv"
            self.assertTrue(
                log_path.exists(),
                f"train_log.csv not found at {log_path}",
            )
            with open(log_path, newline="") as f:
                reader = csv.DictReader(f)
                expected_cols = {
                    "epoch", "step", "total_loss",
                    "eeg_loss", "meg_loss", "fmri_loss", "adversarial_loss",
                }
                self.assertTrue(
                    expected_cols.issubset(set(reader.fieldnames or [])),
                    f"CSV missing columns. Got: {reader.fieldnames}",
                )
                rows = list(reader)
                self.assertGreater(len(rows), 0, "CSV has no data rows")

                # --- Assert 2: No NaN or Inf in any loss column ---
                loss_cols = ["total_loss", "eeg_loss", "meg_loss", "fmri_loss", "adversarial_loss"]
                for row in rows:
                    for col in loss_cols:
                        val = float(row[col])
                        self.assertFalse(
                            math.isnan(val) or math.isinf(val),
                            f"NaN/Inf found in {col} at epoch={row['epoch']}, step={row['step']}: {val}",
                        )

            # --- Assert 3: At least one .pt checkpoint file exists ---
            ckpt_files = list(Path(ckpt_dir).glob("*.pt"))
            self.assertGreater(
                len(ckpt_files),
                0,
                f"No .pt checkpoint files found in {ckpt_dir}",
            )


if __name__ == "__main__":
    unittest.main()
