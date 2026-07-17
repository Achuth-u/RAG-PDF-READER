"""Ollama service health checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class OllamaStatus:
    """Current Ollama reachability and installed model names."""

    reachable: bool
    models: tuple[str, ...]
    error: str | None = None


def get_ollama_status(base_url: str, timeout: float = 2.0) -> OllamaStatus:
    """Return Ollama server status using its local HTTP API."""

    tags_url = f"{base_url.rstrip('/')}/api/tags"
    request = Request(tags_url, headers={"Accept": "application/json"})

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return OllamaStatus(False, (), f"Ollama returned HTTP {exc.code}.")
    except URLError:
        return OllamaStatus(False, (), "Ollama is not running or is not reachable.")
    except TimeoutError:
        return OllamaStatus(False, (), "Ollama status check timed out.")
    except OSError as exc:
        return OllamaStatus(False, (), str(exc))
    except json.JSONDecodeError:
        return OllamaStatus(False, (), "Ollama returned an invalid status response.")

    models = tuple(
        model.get("name", "")
        for model in payload.get("models", [])
        if isinstance(model, dict) and model.get("name")
    )
    return OllamaStatus(True, models)


def is_model_available(model_name: str, installed_models: tuple[str, ...]) -> bool:
    """Return whether the selected model is installed locally."""

    requested_names = {model_name, model_name.split(":", 1)[0]}
    return any(
        requested_names.intersection({installed, installed.split(":", 1)[0]})
        for installed in installed_models
    )
