"""FastAPI/static UI checks. Ollama is mocked; no GPU inference."""

import copy
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app
from src.common import ROOT, load_config
from src.ui_service import _request_lock


class WebTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.result = {
            "input": "Câu hỏi thử", "answer": "Trả lời có bằng chứng [c1].",
            "retrieved_chunks": [{"chunk_id": "c1", "doc_id": "d1", "text": "Bằng chứng",
                                  "source_url": "https://example.com", "score": 0.8,
                                  "start_char": 0, "end_char": 10}],
            "retrieval_seconds": 0.1, "generation_seconds": 0.2, "total_seconds": 0.3,
        }

    def test_serves_page_assets_config_and_docs_without_model_calls(self):
        with patch("src.ui_service.OllamaClient") as ollama:
            for path in ("/", "/static/style.css", "/static/app.js", "/api/config", "/docs"):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200, path)
            ollama.assert_not_called()
        self.assertIn("Hỏi tài liệu.", self.client.get("/").text)
        self.assertNotIn("localhost:11434", self.client.get("/static/app.js").text)

    def test_ask_passes_only_question_and_settings_and_returns_export(self):
        with patch("app.run_question", return_value=self.result) as run:
            response = self.client.post("/api/ask", json={"question": " Câu hỏi thử ", "top_k": 3})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(run.call_args.args[1:], ("Câu hỏi thử", 3, False))
            self.assertEqual(response.json()["result"], self.result)
            self.assertEqual(response.json()["run_settings"]["top_k"], 3)
            retrieval = copy.deepcopy(self.result)
            retrieval.pop("answer")
            run.return_value = retrieval
            response = self.client.post("/api/ask", json={"question": "Thử", "retrieval_only": True})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(run.call_args.args[3])
            self.assertNotIn("answer", response.json()["result"])

    def test_rejects_invalid_inputs_and_gold_fields(self):
        with patch("app.run_question") as run:
            for body in ({"question": "   "}, {"question": "x", "top_k": 0},
                         {"question": "x", "top_k": 11}, {"question": "x" * 4001},
                         {"question": "x", "context": ["gold evidence"]}):
                self.assertEqual(self.client.post("/api/ask", json=body).status_code, 422)
            run.assert_not_called()

    def test_model_errors_missing_index_and_busy_status(self):
        with patch("app.run_question", side_effect=RuntimeError("Ollama offline")):
            response = self.client.post("/api/ask", json={"question": "Test"})
            self.assertEqual(response.status_code, 503)
            self.assertIn("Ollama offline", response.json()["detail"])
        config = load_config(ROOT / "configs/naive.yaml")
        config["index_dir"] = str(ROOT / ".test_tmp" / "missing_web_index")
        with patch("app.settings", return_value=config):
            self.assertEqual(self.client.post("/api/ask", json={"question": "Test"}).status_code, 409)
        _request_lock.acquire()
        try:
            self.assertEqual(self.client.post("/api/ask", json={"question": "Test"}).status_code, 429)
        finally:
            _request_lock.release()

    def test_health_connection_failure_is_readable(self):
        with patch("app.check_connection", side_effect=RuntimeError("offline")):
            response = self.client.get("/api/health")
            self.assertEqual(response.status_code, 503)
            self.assertIn("offline", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
