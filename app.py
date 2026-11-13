"""Serve the RAG web UI and API on one port: uvicorn app:app --port 8000."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.common import ROOT, load_config
from src.ui_service import BusyError, check_connection, run_question
from src.reranking import reranker_metadata

app = FastAPI(title="ViAdminRAG", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
PipelineName = Literal["naive", "advanced_chunking", "advanced_query", "advanced_hybrid", "advanced_rerank", "advanced_all"]
PIPELINES = {name: ROOT / "configs" / f"{name}.yaml" for name in PipelineName.__args__}


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=10)
    retrieval_only: bool = False
    pipeline: PipelineName = "naive"

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value):
        if not value.strip():
            raise ValueError("Bạn chưa nhập câu hỏi.")
        return value.strip()


def settings():
    return load_config(ROOT / "configs/naive.yaml")


def pipeline_settings(name):
    return settings() if name == "naive" else load_config(PIPELINES[name])


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "web/index.html")


@app.get("/api/config")
def config_info(pipeline: PipelineName = "naive"):
    config = pipeline_settings(pipeline)
    return {
        "pipeline": pipeline, "available_pipelines": list(PIPELINES),
        "embedding_model": config["embedding"]["model"],
        "generation_model": config["generation"]["model"],
        "top_k": config["retrieval"]["top_k"],
        "index_ready": (Path(config["index_dir"]) / "manifest.json").is_file(),
        "reranker_ready": not config.get("reranking", {}).get("enabled") or (ROOT / config["reranking"]["model_path"] / "config.json").is_file(),
        "query_processing": config.get("query_processing", {}).get("enabled", False),
    }


@app.get("/api/health")
def health(retrieval_only: bool = False, pipeline: PipelineName = "naive"):
    try:
        models = check_connection(pipeline_settings(pipeline), retrieval_only)
        return {"status": "ok", "models": models}
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        raise HTTPException(503, detail=f"Chưa kết nối được Ollama: {error}") from error


@app.post("/api/ask")
def ask(payload: AskRequest):
    try:
        config = pipeline_settings(payload.pipeline)
        if not (Path(config["index_dir"]) / "manifest.json").is_file():
            raise HTTPException(409, detail=f"Chưa có Dense index. Chạy: python experiments/run.py index --config configs/{payload.pipeline}.yaml")
        top_k = payload.top_k if payload.top_k is not None else config["retrieval"]["top_k"]
        result = run_question(config, payload.question, top_k, payload.retrieval_only)
        return {"result": result, "run_settings": {
            "pipeline": config["name"], "embedding_model": config["embedding"]["model"],
            "generation_model": config["generation"]["model"], "top_k": top_k,
            "retrieval_only": payload.retrieval_only, "chunking": config["chunking"],
            "generation": config["generation"],
            "query_processing": config.get("query_processing", {}),
            "retrieval": dict(config["retrieval"], top_k=top_k),
            "reranking": config.get("reranking", {}),
            "reranker_checkpoint": reranker_metadata(config),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }}
    except BusyError as error:
        raise HTTPException(429, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(409, detail=str(error)) from error
    except (RuntimeError, OSError, KeyError) as error:
        raise HTTPException(503, detail=f"Không xử lý được câu hỏi: {error}") from error
