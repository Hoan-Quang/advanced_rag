"""Run all declared configurations sequentially. Does not rebuild indexes implicitly."""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--generation", action="store_true", help="Include answer generation; default retrieval only")
    args = parser.parse_args()
    for name in ("naive", "advanced_chunking", "advanced_query", "advanced_hybrid", "advanced_rerank", "advanced_all"):
        command = [sys.executable, str(ROOT / "experiments/run.py"), "run", "--config", str(ROOT / f"configs/{name}.yaml")]
        if not args.generation:
            command.append("--retrieval-only")
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        print(f"Running {name}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
