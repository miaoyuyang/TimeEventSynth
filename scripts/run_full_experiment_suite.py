#!/usr/bin/env python3
"""Run the full TimeEventSynth experiment suite across multiple streams."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.experiments.audit_sanity import validate_suite_manifest

ALL_STREAMS = (
    "tests",
    "inspect",
    "ablation",
    "low_label",
    "low_label_sweep",
    "masked_completion",
    "analyze_sweep",
    "real_only",
)


def _run_command(cmd: list[str], *, cwd: Path, stream: str) -> dict[str, Any]:
    print(f"\n=== [{stream}] {' '.join(cmd)} ===", flush=True)
    completed = subprocess.run(cmd, cwd=cwd, check=False)
    return {
        "stream": stream,
        "command": cmd,
        "exit_code": int(completed.returncode),
        "success": completed.returncode == 0,
    }


def _latest_output_dir(base: Path) -> Path | None:
    if not base.exists():
        return None
    candidates = sorted([path for path in base.iterdir() if path.is_dir()], key=lambda path: path.name)
    return candidates[-1] if candidates else None


def _resolve_profile_args(profile: str, args: argparse.Namespace) -> dict[str, Any]:
    if profile == "synthetic":
        return {"use_synthetic": True, "dataset_label": "synthetic", "cli_data": None, "max_series": None}
    if profile == "tsb":
        return {
            "use_synthetic": False,
            "dataset_label": "TSB-UAD-Public-v2",
            "cli_data": args.data,
            "max_series": args.max_series,
        }
    raise ValueError(f"Unknown profile: {profile}")


def run_suite(
    *,
    profiles: list[str],
    streams: list[str],
    config: Path,
    low_label_config: Path,
    masked_config: Path,
    data: Path,
    max_series: int | None,
    skip_tests: bool,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "started_at_utc": started_at,
        "profiles": {},
        "streams_requested": streams,
    }

    for profile in profiles:
        profile_args = _resolve_profile_args(profile, argparse.Namespace(data=data, max_series=max_series))
        profile_entry: dict[str, Any] = {"dataset_label": profile_args["dataset_label"], "runs": []}
        common: list[str] = ["--config", str(config)]
        if profile_args["use_synthetic"]:
            common.append("--use-synthetic")
        else:
            common.extend(["--data", str(profile_args["cli_data"]), "--dataset-name", profile_args["dataset_label"]])
            if profile_args["max_series"] is not None:
                common.extend(["--max-series", str(profile_args["max_series"])])

        active_streams = list(streams)
        if skip_tests and "tests" in active_streams:
            active_streams.remove("tests")

        for stream in active_streams:
            if stream == "tests":
                result = _run_command([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, stream=stream)
                profile_entry["runs"].append(result)
                continue

            if stream == "inspect" and profile_args["use_synthetic"]:
                profile_entry["runs"].append(
                    {"stream": stream, "skipped": True, "reason": "inspect not applicable for synthetic profile"}
                )
                continue

            if stream == "inspect":
                cmd = [
                    sys.executable,
                    "scripts/inspect_dataset.py",
                    "--data",
                    str(profile_args["cli_data"]),
                    "--dataset-name",
                    profile_args["dataset_label"],
                ]
                if profile_args["max_series"] is not None:
                    cmd.extend(["--max-series", str(profile_args["max_series"])])
                result = _run_command(cmd, cwd=PROJECT_ROOT, stream=stream)
                result["output_dir"] = str(PROJECT_ROOT / "outputs" / "inspect" / profile_args["dataset_label"])
                profile_entry["runs"].append(result)
                continue

            if stream == "ablation":
                cmd = [sys.executable, "-m", "src.experiments.run_ablation", *common]
                result = _run_command(cmd, cwd=PROJECT_ROOT, stream=stream)
                result["output_dir"] = str(_latest_output_dir(PROJECT_ROOT / "outputs" / "ablation" / profile_args["dataset_label"]))
                profile_entry["runs"].append(result)
                continue

            if stream == "low_label":
                cmd = [sys.executable, "-m", "src.experiments.run_low_label", *common]
                result = _run_command(cmd, cwd=PROJECT_ROOT, stream=stream)
                result["output_dir"] = str(_latest_output_dir(PROJECT_ROOT / "outputs" / "low_label" / profile_args["dataset_label"]))
                profile_entry["runs"].append(result)
                continue

            if stream == "low_label_sweep":
                cmd = [sys.executable, "-m", "src.experiments.run_low_label_sweep", *common]
                result = _run_command(cmd, cwd=PROJECT_ROOT, stream=stream)
                sweep_dir = _latest_output_dir(PROJECT_ROOT / "outputs" / "low_label_sweep" / profile_args["dataset_label"])
                result["output_dir"] = str(sweep_dir)
                profile_entry["runs"].append(result)
                profile_entry["latest_sweep_dir"] = str(sweep_dir)
                continue

            if stream == "masked_completion":
                cmd = [
                    sys.executable,
                    "-m",
                    "src.experiments.run_masked_completion",
                    "--config",
                    str(masked_config),
                ]
                if profile_args["use_synthetic"]:
                    cmd.append("--use-synthetic")
                else:
                    cmd.extend(
                        [
                            "--data",
                            str(profile_args["cli_data"]),
                            "--dataset-name",
                            profile_args["dataset_label"],
                        ]
                    )
                    if profile_args["max_series"] is not None:
                        cmd.extend(["--max-series", str(profile_args["max_series"])])
                result = _run_command(cmd, cwd=PROJECT_ROOT, stream=stream)
                result["output_dir"] = str(
                    _latest_output_dir(PROJECT_ROOT / "outputs" / "masked_completion" / profile_args["dataset_label"])
                )
                profile_entry["runs"].append(result)
                continue

            if stream == "analyze_sweep":
                sweep_dir = profile_entry.get("latest_sweep_dir")
                if sweep_dir is None:
                    sweep_dir = str(_latest_output_dir(PROJECT_ROOT / "outputs" / "low_label_sweep" / profile_args["dataset_label"]))
                sweep_path = Path(sweep_dir) if sweep_dir else None
                comparison = sweep_path / "low_label_sweep_comparison.csv" if sweep_path else None
                if comparison is None or not comparison.exists():
                    profile_entry["runs"].append(
                        {
                            "stream": stream,
                            "skipped": True,
                            "reason": "low_label_sweep outputs not found; run low_label_sweep first",
                        }
                    )
                    continue
                out_md = sweep_path / "analysis_report.md"
                cmd = [
                    sys.executable,
                    "scripts/analyze_low_label_sweep.py",
                    "--comparison",
                    str(comparison),
                    "--per-series",
                    str(sweep_path / "low_label_sweep_per_series.csv"),
                    "--summary",
                    str(sweep_path / "low_label_sweep_summary.json"),
                    "--out",
                    str(out_md),
                ]
                result = _run_command(cmd, cwd=PROJECT_ROOT, stream=stream)
                result["output_dir"] = str(sweep_path)
                result["analysis_report_md"] = str(out_md)
                result["analysis_report_json"] = str(sweep_path / "analysis_report.json")
                profile_entry["runs"].append(result)
                continue

            if stream == "real_only":
                cmd = [sys.executable, "-m", "src.experiments.run_real_only", *common]
                result = _run_command(cmd, cwd=PROJECT_ROOT, stream=stream)
                result["output_dir"] = str(_latest_output_dir(PROJECT_ROOT / "outputs" / "real_only" / profile_args["dataset_label"]))
                profile_entry["runs"].append(result)
                continue

            profile_entry["runs"].append({"stream": stream, "skipped": True, "reason": f"unknown stream: {stream}"})

        profile_entry["success"] = all(
            run.get("success", run.get("skipped", False)) for run in profile_entry["runs"] if not run.get("skipped")
        )
        manifest["profiles"][profile] = profile_entry

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["success"] = all(entry.get("success", False) for entry in manifest["profiles"].values())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full TimeEventSynth multi-stream experiment suite.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=["synthetic", "tsb", "all"],
        default=["synthetic"],
        help="Dataset profiles to run (default: synthetic). 'all' runs synthetic then TSB-UAD.",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=ALL_STREAMS,
        default=["tests", "ablation", "low_label_sweep", "masked_completion", "analyze_sweep"],
        help="Experiment streams to execute.",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment_low_label.yaml")
    parser.add_argument("--masked-config", type=Path, default=PROJECT_ROOT / "configs" / "experiment_masked_completion.yaml")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "raw" / "TSB-UAD-Public-v2")
    parser.add_argument("--max-series", type=int, default=50)
    parser.add_argument(
        "--skip-tsb",
        action="store_true",
        help="When using --profiles all, skip the TSB-UAD profile (230k-point ECG series are very slow).",
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for suite_manifest.json (default: outputs/experiment_suite/<timestamp>/)",
    )
    args = parser.parse_args()

    profiles = ["synthetic", "tsb"] if "all" in args.profiles else list(args.profiles)
    if args.skip_tsb and "tsb" in profiles:
        profiles.remove("tsb")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = args.out_dir or (PROJECT_ROOT / "outputs" / "experiment_suite" / timestamp)
    suite_dir.mkdir(parents=True, exist_ok=True)

    manifest = run_suite(
        profiles=profiles,
        streams=args.streams,
        config=args.config,
        low_label_config=args.config,
        masked_config=args.masked_config,
        data=args.data,
        max_series=args.max_series,
        skip_tests=args.skip_tests,
    )
    manifest["suite_dir"] = str(suite_dir)
    manifest_path = suite_dir / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    missing = validate_suite_manifest(
        manifest,
        streams_requested=list(args.streams),
        fail_on_missing_stream=bool(getattr(args, "fail_on_missing_stream", False)),
    )
    for warning in missing:
        print(f"WARNING: {warning}", flush=True)

    print(f"\nSuite manifest: {manifest_path}")
    print(f"Overall success: {manifest['success']}")
    if not manifest["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
