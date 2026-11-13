import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", timeout=300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, route, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(self.base_url + route, data=data,
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. Start Ollama and check model names. {error}"
            ) from error

    def model_info(self, name):
        for model in self.request("/api/tags").get("models", []):
            if model.get("name") == name or model.get("model") == name:
                return {"name": name, "digest": model["digest"]}
        raise RuntimeError(f"Model not installed: {name}. Run: ollama pull {name}")

    def embed(self, model, texts):
        response = self.request("/api/embed", {
            "model": model, "input": texts, "truncate": False,
        })
        return response["embeddings"]

    def chat(self, model, messages, options):
        response = self.request("/api/chat", {
            "model": model, "messages": messages, "options": options, "stream": False,
        })
        if not response.get("done") or not response.get("message", {}).get("content", "").strip():
            raise RuntimeError("Ollama returned an incomplete or empty answer")
        return response
