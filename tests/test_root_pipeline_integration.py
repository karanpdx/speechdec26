"""Integration smoke tests for the root train-ready pipeline wrappers."""

from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_root_modules_import():
    modules = [
        "src.data.align_splits",
        "src.data.preprocess_eeg",
        "src.data.preprocess_meg",
        "src.data.preprocess_fmri",
        "src.data.vocab_embeddings",
        "src.models.encoders",
        "src.models.decoders",
        "src.training.dataset",
        "src.training.losses",
        "src.training.train_stage1",
        "src.training.train_stage2",
        "src.evaluation.retrieval",
        "src.evaluation.reconstruction",
    ]
    for module_name in modules:
        import_module(module_name)


def test_stage_train_scripts_import():
    import_module("scripts.train_stage1")
    import_module("scripts.train_stage2")
    import_module("scripts.run_alignment")
    import_module("scripts.sync_s3_data")
    import_module("scripts.modal_train")
