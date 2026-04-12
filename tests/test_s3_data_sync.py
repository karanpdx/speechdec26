"""Tests for S3 raw-data staging helpers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.sync_s3_data import build_sync_command, build_sync_plan


def test_build_sync_plan_from_bucket_and_prefix():
    config = {
        "bucket": "example-bucket",
        "prefix": "project/raw",
        "datasets": {
            "eeg": {"source": "eeg-data", "local_path": "data/raw/eeg-data"},
            "fmri": {"source": "fmri-data", "local_path": "data/raw/fmri-data"},
        },
    }
    plan = build_sync_plan(config, ["eeg", "fmri"])
    assert plan == [
        {
            "modality": "eeg",
            "source_uri": "s3://example-bucket/project/raw/eeg-data",
            "local_path": "data/raw/eeg-data",
        },
        {
            "modality": "fmri",
            "source_uri": "s3://example-bucket/project/raw/fmri-data",
            "local_path": "data/raw/fmri-data",
        },
    ]


def test_build_sync_plan_accepts_full_s3_uri():
    config = {
        "datasets": {
            "meg": {"source": "s3://example-bucket/masc_meg", "local_path": "data/raw/masc_meg"},
        },
    }
    plan = build_sync_plan(config, ["meg"])
    assert plan[0]["source_uri"] == "s3://example-bucket/masc_meg"


def test_build_sync_command_includes_profile_and_region():
    config = {"aws_profile": "research", "aws_region": "us-west-2"}
    cmd = build_sync_command(config, "s3://bucket/eeg-data", "data/raw/eeg-data")
    assert cmd == [
        "aws",
        "--profile",
        "research",
        "--region",
        "us-west-2",
        "s3",
        "sync",
        "s3://bucket/eeg-data",
        "data/raw/eeg-data",
    ]
