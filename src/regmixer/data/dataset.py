import inspect
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlparse

import s3fs
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
)
try:
    from olmo_core.data.source_mixture import SourceMixtureList
except ImportError:
    SourceMixtureList = None
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.io import is_url

from regmixer.aliases import PathType, SourceInstance
from regmixer.data.parquet_utils import materialize_parquet_paths


_SOURCE_MIXTURE_DATASET_PARAMS = inspect.signature(SourceMixtureDatasetConfig).parameters
_USES_SOURCE_LIST_API = "source_list" in _SOURCE_MIXTURE_DATASET_PARAMS


@dataclass
class MixtureBuilder:
    sources: List[SourceInstance]
    max_tokens: int
    sequence_length: int
    seed: int
    dtype: NumpyDatasetDType
    tokenizer: str
    global_batch_size: Optional[int] = None
    processes: int = 1
    fs: s3fs.S3FileSystem = field(default_factory=lambda: s3fs.S3FileSystem())

    def expand_globs(self, paths: List[str]) -> Any:
        results = []
        for path in paths:
            if is_url(path):
                parsed = urlparse(str(path))
                if parsed.scheme in ("s3", "r2", "weka"):
                    results.extend([f"s3://{obj}" for obj in self.fs.glob(path)])
                elif parsed.scheme == "gs":
                    raise NotImplementedError("'gs' types are not currently supported")
                elif parsed.scheme in ("http", "https"):
                    raise NotImplementedError("'http' types are not currently supported")
                elif parsed.scheme == "file":
                    raise NotImplementedError("'file' types are not currently supported")
                else:
                    raise NotImplementedError(
                        f"Glob expansion is not currently supported for '{parsed.scheme}' files"
                    )
            else:
                matches = sorted(glob(path, recursive=True))
                if not matches:
                    raise FileNotFoundError(path)
                results.extend([str(Path(match).resolve()) for match in matches])

        return results

    def build(self) -> SourceMixtureDatasetConfig:
        source_configs: List[SourceMixtureConfig] = []
        for source in self.sources:
            globs = [path for path in source.paths if "*" in path]
            paths = [path for path in source.paths if path not in globs]
            resolved_paths = paths + self.expand_globs(globs)
            resolved_paths = materialize_parquet_paths(
                resolved_paths,
                dtype=self.dtype,
                tokenizer=self.tokenizer,
            )
            source_configs.append(
                SourceMixtureConfig(
                    source_name=source.name,
                    paths=resolved_paths,
                    target_ratio=source.ratio,
                    max_repetition_ratio=source.repetition_factor,
                )
            )

        if _USES_SOURCE_LIST_API:
            if SourceMixtureList is None:  # pragma: no cover - depends on external package shape
                raise ImportError(
                    "olmo_core.data.source_mixture is missing SourceMixtureList for the active API"
                )

            return SourceMixtureDatasetConfig(
                source_list=SourceMixtureList(sources=source_configs),
                requested_tokens=self.max_tokens,
                global_batch_size=self.global_batch_size or self.sequence_length,
                processes=self.processes,
                seed=self.seed,
            )

        return SourceMixtureDatasetConfig(
            source_configs=source_configs,
            max_tokens=self.max_tokens,
            sequence_length=self.sequence_length,
            seed=self.seed,
            dtype=self.dtype,
            processes=self.processes,
        )
