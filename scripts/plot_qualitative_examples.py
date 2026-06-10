from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_series(values: str) -> list[float]:
    cleaned = str(values).strip()
    if not cleaned:
        return []
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    if not cleaned:
        return []
    return [float(token.strip()) for token in cleaned.split(",") if token.strip()]


def _plot_example(row: pd.Series, out_path: Path) -> None:
    original = _parse_series(row.get("original", ""))
    masked = _parse_series(row.get("masked", ""))
    reconstructed = _parse_series(row.get("reconstructed", ""))
    fig, ax = plt.subplots(figsize=(10, 4))
    if original:
        ax.plot(original, label="original", linewidth=2)
    if masked:
        ax.plot(masked, label="masked", linewidth=2)
    if reconstructed:
        ax.plot(reconstructed, label="reconstructed", linewidth=2)
    ax.set_title(str(row.get("example_id", out_path.stem)))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot masked-completion qualitative examples.")
    parser.add_argument("--input-glob", default="outputs/**/*qualitative*.csv")
    parser.add_argument("--output-dir", default="outputs/summary/qualitative_plots")
    args = parser.parse_args()

    input_paths = sorted(PROJECT_ROOT.glob(args.input_glob))
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_paths:
        print(f"No qualitative CSVs found for glob: {args.input_glob}")
        print(str(output_dir))
        return

    created = 0
    for csv_path in input_paths:
        frame = pd.read_csv(csv_path)
        required = {"original", "masked", "reconstructed"}
        if not required.issubset(frame.columns):
            continue
        for idx, row in frame.iterrows():
            example_id = str(row.get("example_id", f"{csv_path.stem}_{idx}"))
            _plot_example(row, output_dir / example_id)
            created += 1
    print(f"created={created}")
    print(str(output_dir))


if __name__ == "__main__":
    main()
