#!/usr/bin/env python3
"""
Convenience workflow for Data Selector.

Usage:
  python3 Data_selector/src/run_selector_workflow.py eric/white_smash_01

It will:
1) Launch interactive selector UI
2) Save selection json under Data_selector/
3) Apply selection and write <dataset>_filtered
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run select/apply workflow with repo id (e.g. eric/white_smash_01)."
    )
    parser.add_argument("repo_id", help="Dataset repo id under HF_LEROBOT_HOME, e.g. eric/white_smash_01")
    parser.add_argument(
        "--hf-home",
        default="/home/ckyljl/.cache/huggingface/lerobot",
        help="Root path that contains <repo_id> datasets",
    )
    parser.add_argument(
        "--only",
        choices=["select", "apply", "all"],
        default="all",
        help="Run only selector UI, only apply, or both",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Selection json path. Supports shorthand '/Data_selector/xxx.json'. "
            "Default: /Data_selector/selection_<dataset_name>.json"
        ),
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    hf_home = Path(args.hf_home).expanduser()
    dataset_path = hf_home / args.repo_id

    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    dataset_name = args.repo_id.split("/")[-1]
    default_selection = workspace_root / "Data_selector" / f"selection_{dataset_name}.json"
    if args.output is None:
        selection_path = default_selection
    else:
        out = args.output
        if out.startswith("/Data_selector/"):
            out = str(workspace_root / out.lstrip("/"))
        selection_path = Path(out).expanduser()
    output_path = hf_home / f"{args.repo_id}_filtered"

    data_selector_script = workspace_root / "Data_selector" / "src" / "data_selector.py"
    apply_script = workspace_root / "Data_selector" / "src" / "apply_selection.py"

    if args.only in ("select", "all"):
        cmd = [
            sys.executable,
            str(data_selector_script),
            "--dataset",
            str(dataset_path),
            "--output",
            str(selection_path),
        ]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(workspace_root))

    if args.only in ("apply", "all"):
        if not selection_path.exists():
            print(f"[ERROR] Selection file not found: {selection_path}", file=sys.stderr)
            print("        Run selector first or provide --only select before apply.", file=sys.stderr)
            return 2
        cmd = [
            sys.executable,
            str(apply_script),
            "--dataset",
            str(dataset_path),
            "--selection",
            str(selection_path),
            "--output",
            str(output_path),
        ]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(workspace_root))

    print("\n[DONE]")
    print(f"  dataset   : {dataset_path}")
    print(f"  selection : {selection_path}")
    print(f"  filtered  : {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
