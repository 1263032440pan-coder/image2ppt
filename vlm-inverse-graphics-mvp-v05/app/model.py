"""Anthropic Messages-compatible adapter for the user-selected API gateway."""
from __future__ import annotations
import base64
import json
import os
import re
from typing import Any
import httpx

DEFAULT_URL = "https://vip.moma.cmecloud.cn/v1/messages"

class ModelAPIError(RuntimeError):
    pass


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"\{", text):
        try:
            data, _ = json.JSONDecoder().raw_decode(text[match.start():])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    raise ModelAPIError("Model did not return a valid JSON object")


def extract_text(response: dict) -> str:
    blocks = response.get("content")
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, list):
        result = "\n".join(x.get("text", "") for x in blocks if isinstance(x, dict) and x.get("type") == "text")
        if result.strip():
            return result
    try:
        result = response["choices"][0]["message"]["content"]
        if isinstance(result, str):
            return result
    except (KeyError, IndexError, TypeError):
        pass
    raise ModelAPIError("Unexpected model response format: missing text content")


def image_part(image: bytes, fmt: str) -> dict:
    encoded = base64.b64encode(image).decode("ascii")
    if fmt == "anthropic":
        return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded}}
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}}


class MessagesClient:
    def __init__(self, key: str | None = None, url: str | None = None,
                 model: str | None = None, image_format: str | None = None) -> None:
        self.key = key if key is not None else os.getenv("MOMA_API_KEY", "")
        self.url = url or os.getenv("MOMA_API_URL", DEFAULT_URL)
        self.model = model or os.getenv("MOMA_MODEL", "deepseek-v4.1-flash-1017")
        self.image_format = image_format or os.getenv("MOMA_IMAGE_FORMAT", "auto")
        self.timeout = float(os.getenv("MOMA_TIMEOUT_SECONDS", "100"))
        if self.image_format not in ("auto", "anthropic", "openai"):
            raise ValueError("MOMA_IMAGE_FORMAT must be auto, anthropic, or openai")
        if not self.url.startswith("https://") and not self.url.startswith("http://localhost"):
            raise ValueError("Model endpoint must use HTTPS")

    def _call(self, content: str | list, max_tokens: int = 4096) -> dict:
        if not self.key:
            raise ModelAPIError("MOMA_API_KEY is missing. Configure .env first, or enable demo mode.")
        body = {"model": self.model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": content}],
                "stream": False, "temperature": 0.2, "top_p": 0.9}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.url, json=body, headers={
                    "Authorization": "Bearer " + self.key,
                    "Content-Type": "application/json",
                })
            if response.status_code >= 400:
                # Include only status + small sanitized provider detail; never expose credential.
                hint = response.text[:450].replace(self.key, "[REDACTED]")
                raise ModelAPIError(f"HTTP {response.status_code} from model endpoint: {hint}")
            result = response.json()
            return parse_json(extract_text(result))
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            raise ModelAPIError(f"Model connection failed: {type(exc).__name__}") from exc
        except (ValueError, TypeError) as exc:
            raise ModelAPIError(f"Invalid model response: {type(exc).__name__}") from exc

    def json(self, prompt: str, images: list[bytes] | None = None, max_tokens: int = 4096) -> dict:
        if not images:
            return self._call(prompt, max_tokens)
        formats = [self.image_format] if self.image_format != "auto" else ["anthropic", "openai"]
        errors = []
        for format_name in formats:
            content = [{"type": "text", "text": prompt}] + [image_part(img, format_name) for img in images]
            try:
                return self._call(content, max_tokens)
            except ModelAPIError as exc:
                errors.append(f"{format_name}: {exc}")
                # On a transport/authentication error, changing image content cannot help.
                if "HTTP 400" not in str(exc) and "HTTP 415" not in str(exc) and "HTTP 422" not in str(exc):
                    raise
        raise ModelAPIError("Multimodal request was rejected in both image formats. " + " | ".join(errors))
