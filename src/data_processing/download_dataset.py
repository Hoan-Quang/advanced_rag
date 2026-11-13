"""Download a ViAdminRAG snapshot, validate it and export UTF-8 JSONL."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

DATASET_ID = "nhokjack/ViAdminRAG"
ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "raw" / "viadminrag"


def main():
    with urlopen(f"https://huggingface.co/api/datasets/{DATASET_ID}", timeout=60) as response:
        info = json.load(response)
    revision = info["sha"]
    files = [
        item["rfilename"] for item in info["siblings"]
        if item["rfilename"] in {"ViAdminRag.json", "README.md"}
    ]
    if "ViAdminRag.json" not in files:
        raise ValueError("ViAdminRag.json is missing from the source snapshot")

    manifest_files = []
    rows = []
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid source path: {name}")
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{revision}/{name}"
        with urlopen(url, timeout=60) as response:
            content = response.read()
        target.write_bytes(content)
        manifest_files.append({
            "path": name, "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
        if name == "ViAdminRag.json":
            rows = json.loads(content.decode("utf-8-sig"))

    required = {"ID", "Content", "Source URL", "Category", "Question type",
                "Difficulty", "input", "expected_output", "context"}
    types = {"Summarization", "Multi-hop QA"}
    if not isinstance(rows, list) or len(rows) != 400:
        raise ValueError("Expected a JSON array with 400 rows; review the source")
    seen_ids = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f"Missing fields at row {index}")
        for key in required - {"context"}:
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Invalid {key} at row {index}")
        if row["ID"] in seen_ids:
            raise ValueError(f"Duplicate ID at row {index}")
        seen_ids.add(row["ID"])
        contexts = row["context"]
        if not isinstance(contexts, list) or not contexts or not all(
            isinstance(text, str) and text.strip() for text in contexts
        ):
            raise ValueError(f"Invalid contexts at row {index}")
        if row["Question type"] not in types:
            raise ValueError(f"Unknown question type at row {index}")
        if row["Difficulty"] not in {"Easy", "Medium", "Hard"}:
            raise ValueError(f"Unknown difficulty at row {index}")

    # Export all rows without changing the original split or field names.
    export = DEST / "train.jsonl"
    export.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "rows": len(rows),
        "question_types": dict(Counter(row["Question type"] for row in rows)),
        "difficulty": dict(Counter(row["Difficulty"] for row in rows)),
        "categories": dict(Counter(row["Category"] for row in rows)),
        "duplicate_questions": len(rows) - len({row["input"].strip() for row in rows}),
        "unique_source_urls": len({row["Source URL"] for row in rows}),
        "unique_contents": len({row["Content"] for row in rows}),
        "unique_reference_contexts": len({text for row in rows for text in row["context"]}),
    }
    manifest = {
        "dataset_id": DATASET_ID, "revision": revision,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": manifest_files, "summary": summary,
    }
    (DEST / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
