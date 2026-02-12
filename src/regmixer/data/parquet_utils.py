import hashlib
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.data.utils import write_array_to_disk
from olmo_core.io import is_url

from regmixer.model.aliases import SupportedTokenizers

logger = logging.getLogger(__name__)

PARQUET_CACHE_DIR_ENV = "REGMIXER_PARQUET_CACHE_DIR"
PARQUET_TEXT_COLUMN_ENV = "REGMIXER_PARQUET_TEXT_COLUMN"
DEFAULT_PARQUET_CACHE_DIR = "cache/parquet_npy"
DEFAULT_TEXT_COLUMNS = ("text", "content", "document")
DEFAULT_INPUT_IDS_COLUMNS = ("input_ids", "token_ids", "tokens")
PARQUET_CACHE_FORMAT_VERSION = "rawbin-v1"


def is_parquet_path(path: str) -> bool:
    return path.lower().endswith(".parquet")


def resolve_tokenizer_identifier(tokenizer: Optional[str]) -> Optional[str]:
    if tokenizer is None:
        return None
    if "/" in tokenizer:
        return tokenizer
    if tokenizer in SupportedTokenizers.__members__:
        tokenizer_config = SupportedTokenizers[tokenizer].value
        if tokenizer_config.identifier is None:
            return None
        return str(tokenizer_config.identifier)
    return tokenizer


def materialize_parquet_paths(
    paths: Sequence[str],
    dtype: NumpyDatasetDType,
    tokenizer: Optional[str] = None,
) -> list[str]:
    tokenizer_identifier = resolve_tokenizer_identifier(tokenizer)
    converted_paths: list[str] = []
    for path in paths:
        if is_parquet_path(path):
            converted_paths.append(
                convert_parquet_to_npy(
                    parquet_path=path,
                    dtype=dtype,
                    tokenizer_identifier=tokenizer_identifier,
                )
            )
        else:
            converted_paths.append(path)
    return converted_paths


def convert_parquet_to_npy(
    parquet_path: str,
    dtype: NumpyDatasetDType,
    tokenizer_identifier: Optional[str] = None,
) -> str:
    if is_url(parquet_path):
        raise NotImplementedError(
            f"Parquet conversion currently supports local paths only, got remote path '{parquet_path}'."
        )

    source = Path(parquet_path)
    if not source.exists():
        raise FileNotFoundError(parquet_path)

    cache_root = Path(os.environ.get(PARQUET_CACHE_DIR_ENV, DEFAULT_PARQUET_CACHE_DIR))
    cache_root.mkdir(parents=True, exist_ok=True)

    stat = source.stat()
    dtype_name = str(dtype.value if hasattr(dtype, "value") else dtype)
    cache_key = "|".join(
        [
            str(source.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            dtype_name,
            tokenizer_identifier or "",
            PARQUET_CACHE_FORMAT_VERSION,
        ]
    )
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    output_path = cache_root / f"{source.stem}-{digest}.npy"

    if output_path.exists():
        return str(output_path)

    logger.info("Converting parquet to npy cache: %s", source)
    token_array = _read_tokens_from_parquet(source, dtype.as_np_dtype(), tokenizer_identifier)
    tmp_path = output_path.with_suffix(f".{os.getpid()}.tmp.npy")
    # NOTE: olmo-core reads these arrays as raw contiguous bytes (np.memmap layout),
    # not .npy-header format. write_array_to_disk keeps the expected wire format.
    write_array_to_disk(token_array, tmp_path)
    os.replace(tmp_path, output_path)
    logger.info("Saved parquet cache %s (%d tokens)", output_path, int(token_array.size))
    return str(output_path)


def _read_tokens_from_parquet(
    source: Path,
    dtype: np.dtype,
    tokenizer_identifier: Optional[str],
) -> np.ndarray:
    parquet_file = pq.ParquetFile(source)
    column_names = set(parquet_file.schema_arrow.names)

    input_ids_column = next((c for c in DEFAULT_INPUT_IDS_COLUMNS if c in column_names), None)
    if input_ids_column is not None:
        return _read_input_ids_column(parquet_file, input_ids_column, dtype)

    requested_text_column = os.environ.get(PARQUET_TEXT_COLUMN_ENV)
    text_column = None
    if requested_text_column and requested_text_column in column_names:
        text_column = requested_text_column
    if text_column is None:
        text_column = next((c for c in DEFAULT_TEXT_COLUMNS if c in column_names), None)

    if text_column is None:
        raise ValueError(
            f"No tokenizable column found in parquet '{source}'. "
            f"Tried input columns {DEFAULT_INPUT_IDS_COLUMNS} and text columns {DEFAULT_TEXT_COLUMNS}."
        )
    if tokenizer_identifier is None:
        raise ValueError(
            f"Parquet file '{source}' requires text tokenization, but tokenizer identifier is missing."
        )

    return _read_text_column(parquet_file, text_column, dtype, tokenizer_identifier)


def _read_input_ids_column(parquet_file: pq.ParquetFile, column: str, dtype: np.dtype) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx, columns=[column])
        arrow_column = table.column(0)
        for chunk in arrow_column.chunks:
            values = chunk.values.to_numpy(zero_copy_only=False)
            if values.size:
                arrays.append(values.astype(dtype, copy=False))

    if not arrays:
        return np.array([], dtype=dtype)
    return np.concatenate(arrays)


def _read_text_column(
    parquet_file: pq.ParquetFile,
    column: str,
    dtype: np.dtype,
    tokenizer_identifier: str,
) -> np.ndarray:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(tokenizer_identifier)
    arrays: list[np.ndarray] = []

    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx, columns=[column])
        arrow_column = table.column(0)
        for chunk in arrow_column.chunks:
            for text in chunk.to_pylist():
                if text is None:
                    continue
                token_ids = tokenizer.encode(str(text)).ids
                if token_ids:
                    arrays.append(np.asarray(token_ids, dtype=dtype))

    if not arrays:
        return np.array([], dtype=dtype)
    return np.concatenate(arrays)
