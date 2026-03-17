from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, List, Literal, Optional, Union
import warnings

from olmo_core.data.types import NumpyDatasetDType
from pydantic import BaseModel, Field, model_validator
import pydantic

try:
    from beaker import Priority
except ModuleNotFoundError:
    # Local-only flows (e.g. generate-mixes / run_local_variant) do not need beaker.
    Priority = str  # type: ignore[assignment]

try:
    from olmo_core.launch.beaker import BeakerLaunchConfig
except ModuleNotFoundError:
    BeakerLaunchConfig = Any  # type: ignore[assignment]

PathType = Union[Path, PathLike[Any], str]


class TrainType(Enum):
    pretrain = "pretrain"
    anneal = "anneal"


class TopicConfig(BaseModel):
    name: str
    paths: List[str]
    max_repetition_factor: float = 1.0
    max_topic_ratio: float = 1.0
    weight: Optional[float] = None


class SourceConfig(BaseModel):
    name: str
    paths: Optional[List[str]] = None
    topics: Optional[List[TopicConfig]] = None
    max_repetition_factor: float = 1.0
    max_source_ratio: float = 1.0


class SourceInstance(BaseModel):
    name: str
    paths: list[str]
    ratio: float
    repetition_factor: float = 1.0


class ExperimentConfig(BaseModel):
    name: str
    description: str
    budget: str
    workspace: str
    variants: int
    nodes: int
    gpus: int
    max_tokens: int
    sequence_length: int
    seed: int
    cluster: str
    tokenizer: str
    priority: Priority
    sources: list[SourceConfig]
    tokenizer: str
    proxy_model_id: str
    minimum_weight: Optional[float] = None
    minimum_source_weight: Optional[float] = None
    minimum_topic_weight: Optional[float] = None
    checkpoint_path: Optional[str] = None
    train_type: TrainType = TrainType.pretrain
    allow_repetition: bool = True
    dtype: NumpyDatasetDType = NumpyDatasetDType.uint32
    mix_temperature: float = 1.0
    temperature: Optional[float] = None
    source_mix_temperature: Optional[float] = None
    topic_mix_temperature: Optional[float] = None
    preemptible: bool = True
    shared_filesystem: bool = False
    weka: bool = False
    min_strength: float = 0.1
    max_strength: float = 5.0
    min_source_strength: Optional[float] = None
    max_source_strength: Optional[float] = None
    min_topic_strength: Optional[float] = None
    max_topic_strength: Optional[float] = None
    nonzero_weight: Optional[list[str]] = None
    fixed_source_weights: Optional[dict[str, float]] = None
    device_batch_size: int = 4
    global_batch_size: Optional[int] = None
    manual_prior: Optional[dict[str, float]] = None
    sample_multiplier: Optional[int] = None
    max_repetition_ratio: float = 5.0
    candidate_sampling_strategy: Literal["dirichlet", "uniform", "vertex_biased", "mixed"] = "dirichlet"
    candidate_pool_size: Optional[int] = None
    candidate_random_seed: Optional[int] = None
    candidate_vertex_prob: float = 0.3
    candidate_min_dominant_weight: float = 0.7
    candidate_min_config_distance: float = 0.0
    design_selection_method: Literal["random", "d_opt"] = "random"
    design_basis: str = "quadratic_logratio_2d"
    design_random_seed: Optional[int] = None
    design_ridge: float = 1e-6
    design_log_floor: float = 1e-4
    design_min_selected_distance: float = 0.0
    design_anchor_points: list[str] = Field(default_factory=list)
    wandb_debug: bool = False
    existing_mix_file: Optional[str] = None
    # TODO(undfined): Add field validation for weka/cluster/train_type here

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, values: Any) -> Any:
        values = dict(values or {})

        legacy_temperature = values.get("temperature")
        if legacy_temperature is not None:
            if values.get("mix_temperature") is None:
                values["mix_temperature"] = legacy_temperature
            warnings.warn(
                "ExperimentConfig.temperature is deprecated; use mix_temperature instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        return values

    @model_validator(mode="after")
    def _validate_design_fields(self) -> "ExperimentConfig":
        variants = int(self.variants)
        if variants < 1:
            raise ValueError("variants must be >= 1")

        candidate_pool_size = self.candidate_pool_size
        if candidate_pool_size is not None and int(candidate_pool_size) < variants:
            raise ValueError("candidate_pool_size must be >= variants when provided")

        max_repetition_ratio = float(self.max_repetition_ratio)
        if max_repetition_ratio < 1.0:
            raise ValueError("max_repetition_ratio must be >= 1.0")

        if float(self.candidate_min_dominant_weight) <= 0.0:
            raise ValueError("candidate_min_dominant_weight must be > 0")

        if not 0.0 <= float(self.candidate_vertex_prob) <= 1.0:
            raise ValueError("candidate_vertex_prob must be in [0, 1]")

        if float(self.candidate_min_config_distance) < 0.0:
            raise ValueError("candidate_min_config_distance must be >= 0")

        if float(self.design_log_floor) <= 0.0:
            raise ValueError("design_log_floor must be > 0")

        if float(self.design_ridge) <= 0.0:
            raise ValueError("design_ridge must be > 0")

        if float(self.design_min_selected_distance) < 0.0:
            raise ValueError("design_min_selected_distance must be >= 0")

        return self


class ExperimentInstance(BaseModel):
    name: str
    sources: list[SourceInstance]


class ExperimentGroup(BaseModel):
    config: ExperimentConfig
    group_id: str
    instances: list[ExperimentInstance]


class LaunchGroup(BaseModel):
    instances: list[BeakerLaunchConfig]
