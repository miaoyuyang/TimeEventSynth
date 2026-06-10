"""Synthetic benchmark records for smoke tests and ablations."""

from __future__ import annotations

import numpy as np

from .tsb_loader import TimeSeriesRecord


def make_synthetic_records(
    seed: int = 42,
    *,
    num_series: int = 30,
    events_per_series: int = 10,
) -> list[TimeSeriesRecord]:
    rng = np.random.default_rng(seed)
    domains = ["domain_a", "domain_b", "domain_c"]
    records: list[TimeSeriesRecord] = []
    for series_idx in range(num_series):
        domain = domains[series_idx % len(domains)]
        length = 120
        values = np.sin(np.linspace(0, 4 * np.pi, length)) + 0.1 * rng.normal(size=length)
        labels = np.zeros(length, dtype=int)
        for event_idx in range(events_per_series):
            start = min(length - 6, max(1, 8 + event_idx * 10 + int(rng.integers(-2, 3))))
            end = min(start + int(rng.integers(5, 9)), length)
            values[start:end] += 2.5 + 0.5 * rng.normal(size=end - start)
            labels[start:end] = 1
        series_id = f"{domain}/series_{series_idx:03d}"
        records.append(
            TimeSeriesRecord(
                series_id=series_id,
                values=values.astype(float),
                labels=labels.astype(int),
                timestamps=None,
                source_path="synthetic",
                metadata={
                    "synthetic": True,
                    "parent_folder": domain,
                },
            )
        )
    return records
