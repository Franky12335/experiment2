"""Restore a saved checkpoint and evaluate the full SearchQA test split."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen2.5_3b", "qwen3_1.7b"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--devices")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/run_experiment.py"),
               "--model", args.model, "--checkpoint", str(args.checkpoint.expanduser().resolve())]
    for name in ("devices", "run_name"):
        if getattr(args, name):
            command += ["--" + name.replace("_", "-"), getattr(args, name)]
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
