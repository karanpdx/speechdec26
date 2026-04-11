"""Root wrapper for modality decoders."""

from workers.worker_models.src.models.decoders import (  # noqa: F401
    EEGDecoder,
    MEGDecoder,
    _TemporalDecoder,
    fMRIDecoder,
)
