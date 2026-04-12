"""Import-safe tests for the Modal training runner."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.modal_train import _write_runtime_config


def test_write_runtime_config_overrides_device_and_checkpoint(tmp_path):
    src = tmp_path / "config.yaml"
    src.write_text(
        "device: cpu\nstage1_checkpoint: old.pt\n",
        encoding="utf-8",
    )

    out = _write_runtime_config(
        str(src),
        device="cuda",
        stage1_checkpoint="checkpoints/stage1/best.pt",
    )

    text = Path(out).read_text(encoding="utf-8")
    assert "device: cuda" in text
    assert "stage1_checkpoint: checkpoints/stage1/best.pt" in text
