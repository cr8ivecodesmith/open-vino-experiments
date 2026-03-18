"""Shared parameter dataclasses and the Backend protocol.

All backend implementations must satisfy the :class:`Backend` protocol so they
can be used interchangeably by the command layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

VALID_TASKS = ("text_generation", "embeddings", "rerank", "image_generation")
VALID_PIPELINE_TYPES = ("LM", "LM_CB", "VLM", "VLM_CB", "AUTO")
VALID_DEVICES = ("CPU", "GPU", "NPU", "AUTO", "MULTI", "HETERO")


@dataclass
class PullParams:
    """Parameters forwarded to ``ovms --pull``.

    Attributes:
        source_model: HuggingFace model ID (e.g. ``OpenVINO/whisper-base-fp16-ov``).
        model_repository_path: Local directory for the model repository.
        task: OVMS task type.
        model_name: Name exposed externally by the server. Defaults to *source_model*.
        target_device: Inference device. Defaults to ``CPU``.
        weight_format: Weight precision for non-OpenVINO org models (passed to optimum-cli).
            ``None`` means the flag is omitted entirely, which is required for pre-converted
            OpenVINO models (e.g. from the ``OpenVINO/`` HuggingFace org).
        overwrite: If True, overwrite an existing model with the same name.
        pipeline_type: Pipeline sub-type (text_generation only).
        cache_size: KV cache size in GB (text_generation only).
        hf_token: HuggingFace authentication token.
        extra_quantization_params: Extra params forwarded to optimum-cli.
    """

    source_model: str
    model_repository_path: Path
    task: str
    model_name: str = ""
    target_device: str = "CPU"
    weight_format: str | None = None
    overwrite: bool = False
    pipeline_type: str | None = None
    cache_size: int | None = None
    hf_token: str | None = None
    extra_quantization_params: str | None = None

    def __post_init__(self) -> None:
        """Default model_name to source_model if not provided."""
        if not self.model_name:
            self.model_name = self.source_model


@dataclass
class RegisterParams:
    """Parameters forwarded to ``ovms --add_to_config``.

    Attributes:
        config_json_path: Full path to the OVMS config.json file.
            Passed as ``--config_path`` to the ``--add_to_config`` flag.
        model_repository_path: Local directory for the model repository.
        model_name: Name as exposed externally by the server.
        model_path: Path within the repository (defaults to model_name).
    """

    config_json_path: Path
    model_repository_path: Path
    model_name: str
    model_path: str = field(default="")

    def __post_init__(self) -> None:
        """Default model_path to model_name if not provided."""
        if not self.model_path:
            self.model_path = self.model_name


@dataclass
class RemoveParams:
    """Parameters forwarded to ``ovms --remove_from_config``.

    Attributes:
        config_json_path: Full path to the OVMS config.json file.
            Passed as ``--config_path`` to the ``--remove_from_config`` flag.
        model_name: Name of the model to remove from config.
    """

    config_json_path: Path
    model_name: str


@runtime_checkable
class Backend(Protocol):
    """Protocol that all backend implementations must satisfy."""

    def pull(self, params: PullParams) -> None:
        """Download a model from HuggingFace into the model repository.

        Args:
            params: Pull parameters.
        """
        ...

    def register(self, params: RegisterParams) -> None:
        """Register a model in the OVMS config.json.

        Args:
            params: Registration parameters.
        """
        ...

    def remove(self, params: RemoveParams) -> None:
        """Unregister a model from the OVMS config.json.

        Args:
            params: Remove parameters.
        """
        ...
