"""Run and score the declared pipelines sequentially, keeping a session status file."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.common import ROOT, load_config

NAMES = ("naive", "advanced_query", "advanced_hybrid", "advanced_rerank", "advanced_all")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-dir", help="Reuse or wait for an already started baseline run")
    parser.add_argument("--generation", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    status = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
              "generation": args.generation, "status": "running", "pipelines": {},
              "label_note": "Retrieval uses draft automatic labels for 229/400 questions."}
    snapshots = output / "configs"
    snapshots.mkdir(exist_ok=True)
    for name in NAMES:
        frozen = load_config(ROOT / f"configs/{name}.yaml")
        (snapshots / f"{name}.yaml").write_text(yaml.safe_dump(frozen, allow_unicode=True), encoding="utf-8")

    def save():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(status_path)

    def run(command, logfile, watch=None):
        with logfile.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen([sys.executable, *command], cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT)
            while process.poll() is None:
                if watch:
                    watch()
                time.sleep(3)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)

    save()
    successful = []
    for name in NAMES:
        record = {"status": "running"}
        status["pipelines"][name] = record
        status["current_pipeline"] = name
        save()
        log = output / f"{name}.log"
        try:
            if name == "naive" and args.baseline_run_dir:
                folder = Path(args.baseline_run_dir).resolve()
                record["run_dir"] = str(folder)
                save()
                while True:
                    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
                    if manifest["status"] != "running":
                        break
                    time.sleep(3)
                if manifest["status"] != "complete":
                    raise RuntimeError("Existing baseline did not complete")
                if args.generation and manifest["retrieval_only"]:
                    raise ValueError("Cannot reuse retrieval-only baseline for Generation evaluation")
            else:
                parent = ROOT / "results" / name
                before = set(parent.glob("*/manifest.json")) if parent.exists() else set()
                command = [str(ROOT / "experiments/run.py"), "run", "--config", str(snapshots / f"{name}.yaml")]
                if not args.generation:
                    command.append("--retrieval-only")
                def progress():
                    found = set(parent.glob("*/manifest.json")) - before
                    if len(found) == 1:
                        manifest_path = next(iter(found))
                        current = json.loads(manifest_path.read_text(encoding="utf-8"))
                        record.update(run_dir=str(manifest_path.parent), completed_queries=current["completed"],
                                      total_queries=current["query_count"])
                        save()
                run(command, log, watch=progress)
                created = set(parent.glob("*/manifest.json")) - before
                if len(created) != 1:
                    raise RuntimeError("Could not unambiguously identify the new run")
                folder = created.pop().parent
                record["run_dir"] = str(folder)
                save()
            run([str(ROOT / "experiments/evaluate.py"), "--run-dir", str(folder),
                 "--k", "5", "--allow-partial"], log)
            if args.generation:
                run([str(ROOT / "experiments/evaluate_answers.py"), "--run-dir", str(folder)], log)
            metrics = json.loads((folder / "retrieval_metrics_at_5.json").read_text(encoding="utf-8"))
            record.update(status="complete", scored_queries=metrics["scored_queries"],
                          total_queries=metrics["total_queries"], metrics=metrics["metrics"])
            if args.generation:
                answer_metrics = json.loads((folder / "answer_metrics.json").read_text(encoding="utf-8"))
                record["generation_metrics"] = answer_metrics["metrics"]
            successful.append(str(folder))
        except Exception as error:
            record.update(status="failed", error=str(error), log=str(log))
        save()
        print(f"{name}: {record['status']}", flush=True)
    if successful:
        run([str(ROOT / "experiments/compare.py"), "--runs", *successful,
             "--output", str(output / "comparison.csv")], output / "comparison.log")
    status["status"] = "complete" if all(row["status"] == "complete" for row in status["pipelines"].values()) else "completed_with_errors"
    status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(f"Evaluation saved: {output}", flush=True)


if __name__ == "__main__":
    main()
