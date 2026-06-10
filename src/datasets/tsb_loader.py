"""Flexible loaders for TSB-UAD/TSB-AD-like time-series files.

This module turns one or many CSV/out files into a unified series-level representation.
It supports nested dataset folders, one file per series, alternate column names, and
binary labels encoded as 0/1, True/False, or normal/anomaly strings.
"""

from __future__ import annotations

import fnmatch
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .event_extractor import labels_to_events
from .load_options import DatasetLoadOptions


TIMESTAMP_CANDIDATES = ("timestamp", "time", "date", "datetime", "t", "index", "time_index")
LABEL_CANDIDATES = (
    "label",
    "labels",
    "anomaly",
    "is_anomaly",
    "ground_truth",
    "target",
    "gt",
    "class",
    "y",
)
VALUE_CANDIDATES = ("value", "values", "data", "feature", "signal", "x", "reading")
SERIES_ID_CANDIDATES = ("series_id", "series", "id", "sequence_id", "ts_id", "name")

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".out", ""}

SKIP_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "docs",
    "result",
    "results",
    "outputs",
    "node_modules",
}

POSITIVE_LABEL_STRINGS = frozenset(
    {
        "1",
        "true",
        "yes",
        "y",
        "anomaly",
        "anomalous",
        "abnormal",
        "positive",
        "pos",
        "outlier",
    }
)
NEGATIVE_LABEL_STRINGS = frozenset(
    {
        "0",
        "false",
        "no",
        "n",
        "normal",
        "negative",
        "neg",
        "nominal",
    }
)


@dataclass
class TimeSeriesRecord:
    """One time series instance with values, labels, and optional timestamps."""

    series_id: str
    values: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray | None
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _detect_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def normalize_binary_labels(raw: pd.Series) -> np.ndarray:
    """Convert common label encodings into a binary 0/1 numpy array."""
    if raw.empty:
        return np.asarray([], dtype=int)

    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(False).astype(int).to_numpy(dtype=int)

    if pd.api.types.is_numeric_dtype(raw):
        numeric = pd.to_numeric(raw, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return (numeric > 0).astype(int)

    normalized: list[int] = []
    for value in raw.fillna("normal").astype(str).str.strip().str.lower():
        if value in POSITIVE_LABEL_STRINGS:
            normalized.append(1)
        elif value in NEGATIVE_LABEL_STRINGS:
            normalized.append(0)
        else:
            try:
                normalized.append(1 if float(value) > 0 else 0)
            except ValueError:
                normalized.append(0)
    return np.asarray(normalized, dtype=int)


def _looks_like_label_column(series: pd.Series) -> bool:
    if series.empty:
        return False
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            return False
        unique = np.unique(values)
        if unique.size <= 2 and np.all(np.isin(unique, [0.0, 1.0])):
            return True
        return False
    lowered = series.fillna("normal").astype(str).str.strip().str.lower()
    allowed = POSITIVE_LABEL_STRINGS | NEGATIVE_LABEL_STRINGS
    return float(lowered.isin(allowed).mean()) >= 0.8


def _looks_like_value_column(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not _looks_like_label_column(series)


def _candidate_csv_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    candidates: list[Path] = []
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        if item.name.startswith("."):
            continue
        if item.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if any(part in SKIP_DIR_NAMES for part in item.parts):
            continue
        candidates.append(item)
    return candidates


def _series_id_from_path(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
        return str(relative.with_suffix("")).replace("\\", "/")
    except ValueError:
        return path.stem


def _parent_folder_from_series_id(series_id: str, path: Path, root: Path) -> str:
    normalized = series_id.replace("\\", "/")
    if "/" in normalized:
        return normalized.split("/", 1)[0]
    try:
        relative_parent = path.relative_to(root).parent
        if str(relative_parent) not in {"", "."}:
            return str(relative_parent).replace("\\", "/")
    except ValueError:
        pass
    return path.parent.name or "root"


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() in {".csv", ".txt", ".out", ""}:
        try:
            frame = pd.read_csv(path)
            if frame.shape[1] >= 2:
                return frame
        except pd.errors.ParserError:
            pass
        return pd.read_csv(path, header=None)
    return pd.read_csv(path)


def _parse_headerless_frame(frame: pd.DataFrame, path: Path, root: Path) -> pd.DataFrame:
    if frame.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in headerless file: {path}")

    first_col = frame.iloc[:, 0]
    last_col = frame.iloc[:, -1]
    middle_cols = frame.iloc[:, 1:-1] if frame.shape[1] > 2 else None

    if frame.shape[1] == 2:
        value_col = frame.iloc[:, 0]
        label_col = frame.iloc[:, 1]
        parsed = pd.DataFrame(
            {
                "series_id": _series_id_from_path(path, root),
                "time_index": np.arange(len(frame), dtype=int),
                "value": pd.to_numeric(value_col, errors="coerce"),
                "label": label_col,
            }
        )
        return parsed.dropna(subset=["value"])

    label_is_last = _looks_like_label_column(last_col)
    label_is_first = _looks_like_label_column(first_col) and (
        middle_cols is None or all(_looks_like_value_column(middle_cols.iloc[:, idx]) for idx in range(middle_cols.shape[1]))
    )

    if label_is_last and not label_is_first:
        value_cols = frame.iloc[:, :-1]
        label_col = last_col
    elif label_is_first:
        value_cols = frame.iloc[:, 1:]
        label_col = first_col
    elif _looks_like_label_column(last_col):
        value_cols = frame.iloc[:, :-1]
        label_col = last_col
    else:
        value_cols = frame.iloc[:, :-1]
        label_col = last_col

    value_frame = value_cols.apply(pd.to_numeric, errors="coerce")
    if value_frame.shape[1] == 1:
        values = value_frame.iloc[:, 0]
    else:
        values = value_frame.to_numpy(dtype=float)

    parsed = pd.DataFrame(
        {
            "series_id": _series_id_from_path(path, root),
            "time_index": np.arange(len(frame), dtype=int),
            "value": values,
            "label": label_col,
        }
    )
    if isinstance(parsed["value"].iloc[0], np.ndarray):
        parsed = parsed.dropna(subset=["value"])
    else:
        parsed = parsed.dropna(subset=["value"])
    return parsed


def _infer_label_column(frame: pd.DataFrame) -> str:
    label_column = _detect_column(list(frame.columns), LABEL_CANDIDATES)
    if label_column is not None:
        return label_column
    for column in reversed(frame.columns):
        if _looks_like_label_column(frame[column]):
            return str(column)
    if frame.shape[1] >= 2:
        return str(frame.columns[-1])
    raise ValueError("Could not detect label column.")


def _detect_numeric_value_columns(frame: pd.DataFrame, excluded: set[str]) -> list[str]:
    numeric_columns = [
        column
        for column in frame.columns
        if column not in excluded and _looks_like_value_column(frame[column])
    ]
    preferred = _detect_column([str(column) for column in numeric_columns], VALUE_CANDIDATES)
    if preferred is not None:
        return [preferred]
    if numeric_columns:
        return [str(column) for column in numeric_columns]
    raise ValueError("Could not detect any numeric value column.")


def _frame_to_records(frame: pd.DataFrame, path: Path, root: Path) -> list[TimeSeriesRecord]:
    if frame.empty:
        return []

    columns = [str(column) for column in frame.columns]
    if set(columns) >= {"series_id", "value", "label"}:
        label_column = "label"
        timestamp_column = "time_index" if "time_index" in columns else _detect_column(columns, TIMESTAMP_CANDIDATES)
        series_id_column = "series_id"
        value_columns = ["value"]
    else:
        label_column = _infer_label_column(frame)
        timestamp_column = _detect_column(columns, TIMESTAMP_CANDIDATES)
        series_id_column = _detect_column(columns, SERIES_ID_CANDIDATES)
        if series_id_column is None:
            frame = frame.copy()
            frame["_series_id"] = _series_id_from_path(path, root)
            series_id_column = "_series_id"
        excluded = {label_column, series_id_column}
        if timestamp_column is not None:
            excluded.add(timestamp_column)
        value_columns = _detect_numeric_value_columns(frame, excluded)

    records: list[TimeSeriesRecord] = []
    for raw_series_id, group in frame.groupby(series_id_column, sort=True):
        sort_columns = [timestamp_column] if timestamp_column is not None else None
        ordered = group.sort_values(sort_columns) if sort_columns else group.reset_index(drop=True)
        value_array = ordered[value_columns].to_numpy(dtype=float)
        if value_array.ndim == 2 and value_array.shape[1] == 1:
            value_array = value_array[:, 0]
        labels = normalize_binary_labels(ordered[label_column])
        timestamps = ordered[timestamp_column].to_numpy() if timestamp_column is not None else None
        records.append(
            TimeSeriesRecord(
                series_id=str(raw_series_id),
                values=value_array,
                labels=labels,
                timestamps=timestamps,
                source_path=str(path),
                metadata={
                    "value_columns": value_columns,
                    "label_column": label_column,
                    "timestamp_column": timestamp_column,
                    "series_id_column": series_id_column,
                    "num_points": int(len(ordered)),
                    "num_dimensions": int(value_array.shape[1]) if getattr(value_array, "ndim", 1) == 2 else 1,
                    "parent_folder": _parent_folder_from_series_id(str(raw_series_id), path, root),
                    "relative_path": _series_id_from_path(path, root),
                },
            )
        )
    return records


def _read_single_file(path: Path, root: Path) -> list[TimeSeriesRecord]:
    frame = _read_table(path)
    unnamed = all(str(column).startswith("Unnamed:") or str(column).isdigit() for column in frame.columns)
    if unnamed or path.suffix.lower() == ".out":
        frame = _parse_headerless_frame(frame, path, root)
    return _frame_to_records(frame, path, root)


def _matches_series_patterns(series_id: str, options: DatasetLoadOptions) -> bool:
    if options.exclude_patterns and any(fnmatch.fnmatchcase(series_id, pattern) for pattern in options.exclude_patterns):
        return False
    if options.include_patterns and not any(fnmatch.fnmatchcase(series_id, pattern) for pattern in options.include_patterns):
        return False
    return True


def filter_records(records: list[TimeSeriesRecord], options: DatasetLoadOptions | None) -> tuple[list[TimeSeriesRecord], dict[str, Any]]:
    """Apply length/event-count filters and optional max-series cap."""
    options = options or DatasetLoadOptions()
    kept: list[TimeSeriesRecord] = []
    dropped_short = 0
    dropped_few_events = 0
    dropped_high_event_ratio = 0
    dropped_pattern = 0

    for record in records:
        if not _matches_series_patterns(record.series_id, options):
            dropped_pattern += 1
            continue
        length = int(len(record.labels))
        if length < int(options.min_length):
            dropped_short += 1
            continue
        if options.max_length is not None and length > int(options.max_length):
            dropped_short += 1
            continue
        num_events = len(
            labels_to_events(
                record.labels,
                min_length=int(options.min_event_length),
                merge_gap=int(options.merge_gap),
            )
        )
        if num_events < int(options.min_event_windows):
            dropped_few_events += 1
            continue
        event_ratio = float(np.sum(record.labels > 0) / max(length, 1))
        if options.max_event_ratio is not None and event_ratio > float(options.max_event_ratio):
            dropped_high_event_ratio += 1
            continue
        kept.append(record)

    kept.sort(key=lambda item: item.series_id)
    if options.max_series is not None and len(kept) > int(options.max_series):
        rng = random.Random(int(options.seed))
        if options.group_by_parent_folder:
            groups: dict[str, list[TimeSeriesRecord]] = defaultdict(list)
            for record in kept:
                metadata = getattr(record, "metadata", None) or {}
                parent = str(metadata.get("parent_folder", record.series_id))
                groups[parent].append(record)
            group_names = sorted(groups.keys())
            rng.shuffle(group_names)
            queues: dict[str, list[TimeSeriesRecord]] = {
                name: sorted(groups[name], key=lambda item: item.series_id) for name in group_names
            }
            for queue in queues.values():
                rng.shuffle(queue)
            selected: list[TimeSeriesRecord] = []
            active = list(group_names)
            while len(selected) < int(options.max_series) and active:
                next_active: list[str] = []
                for name in active:
                    queue = queues[name]
                    if not queue:
                        continue
                    selected.append(queue.pop(0))
                    if len(selected) >= int(options.max_series):
                        break
                    if queue:
                        next_active.append(name)
                active = next_active
            kept = selected
        else:
            indices = list(range(len(kept)))
            rng.shuffle(indices)
            selected = sorted(indices[: int(options.max_series)])
            kept = [kept[idx] for idx in selected]
        kept.sort(key=lambda item: item.series_id)

    summary = {
        "num_files_seen": len({record.source_path for record in records}),
        "num_series_loaded": len(records),
        "num_series_kept": len(kept),
        "dropped_too_short": dropped_short,
        "dropped_few_events": dropped_few_events,
        "dropped_high_event_ratio": dropped_high_event_ratio,
        "dropped_pattern_mismatch": dropped_pattern,
        "max_series": options.max_series,
        "min_length": options.min_length,
        "max_length": options.max_length,
        "min_event_windows": options.min_event_windows,
        "max_event_ratio": options.max_event_ratio,
        "include_patterns": options.include_patterns,
        "exclude_patterns": options.exclude_patterns,
        "group_by_parent_folder": options.group_by_parent_folder,
    }
    return kept, summary


def _iter_loadable_paths(input_path: Path, root: Path, options: DatasetLoadOptions | None) -> list[Path]:
    paths = _candidate_csv_paths(input_path)
    if options is None:
        return paths
    return [path for path in paths if _matches_series_patterns(_series_id_from_path(path, root), options)]


def load_tsb_records(
    path: str | Path,
    *,
    options: DatasetLoadOptions | None = None,
) -> list[TimeSeriesRecord]:
    """Load one file or a directory tree into filtered series-level records."""
    input_path = Path(path)
    root = input_path if input_path.is_dir() else input_path.parent
    records: list[TimeSeriesRecord] = []
    errors: list[str] = []

    for file_path in _iter_loadable_paths(input_path, root, options):
        try:
            records.extend(_read_single_file(file_path, root))
        except Exception as exc:  # noqa: BLE001 - collect and report per-file load failures
            errors.append(f"{file_path}: {exc}")

    if not records:
        detail = "; ".join(errors[:5]) if errors else "no readable files"
        raise ValueError(f"No time series records found under {input_path}. {detail}")

    filtered, _ = filter_records(records, options)
    if not filtered:
        raise ValueError(
            f"All {len(records)} loaded series were filtered out under {input_path}. "
            f"Try lowering --min-length or --min-event-windows."
        )
    return filtered


def load_tsb_records_with_summary(
    path: str | Path,
    *,
    options: DatasetLoadOptions | None = None,
) -> tuple[list[TimeSeriesRecord], dict[str, Any]]:
    """Load records and return a load/filter summary for inspection outputs."""
    input_path = Path(path)
    root = input_path if input_path.is_dir() else input_path.parent
    records: list[TimeSeriesRecord] = []
    file_errors: list[str] = []

    candidate_paths = _candidate_csv_paths(input_path)
    load_paths = _iter_loadable_paths(input_path, root, options)
    for file_path in load_paths:
        try:
            records.extend(_read_single_file(file_path, root))
        except Exception as exc:  # noqa: BLE001
            file_errors.append(f"{file_path}: {exc}")

    if not records:
        detail = "; ".join(file_errors[:5]) if file_errors else "no readable files"
        raise ValueError(f"No time series records found under {input_path}. {detail}")

    filtered, summary = filter_records(records, options)
    summary["file_errors"] = file_errors
    summary["num_files_seen"] = len(candidate_paths)
    summary["num_files_loaded"] = len(load_paths)
    summary["dataset_name"] = (options.dataset_name if options else None) or input_path.name
    if not filtered:
        raise ValueError(
            f"All {len(records)} loaded series were filtered out under {input_path}. "
            f"Try lowering --min-length or --min-event-windows."
        )
    return filtered, summary


def load_tsb_like_csv(path: Path) -> pd.DataFrame:
    """Backward-compatible helper for smoke scripts that expect a single dataframe."""
    return pd.read_csv(path)


def timelines_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a pointwise dataframe into timeline dicts for simple experiments."""
    if "series_id" not in frame.columns:
        raise ValueError("Expected `series_id` column in frame.")
    timelines: list[dict[str, Any]] = []
    for series_id, group in frame.groupby("series_id", sort=True):
        value_columns = [column for column in group.columns if column not in {"series_id", "label", "time_index"}]
        if "value" in value_columns:
            value_columns = ["value"]
        values = group[value_columns].to_numpy(dtype=float)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        timelines.append(
            {
                "series_id": str(series_id),
                "values": values.tolist(),
                "labels": normalize_binary_labels(group["label"]).tolist(),
                "time_index": group["time_index"].astype(int).tolist() if "time_index" in group.columns else list(range(len(group))),
            }
        )
    return timelines
