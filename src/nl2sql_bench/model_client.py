"""Model client over an OpenAI-compatible endpoint.

Works for a local vLLM server (`vllm serve ... --port 8000`) and for any cloud
reference model exposing the OpenAI API. Tracks token usage and wall-clock so the
runner can report cost/latency alongside accuracy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


@dataclass
class ModelClient:
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"          # vLLM ignores it; set real key for cloud ref
    temperature: float = 0.0
    max_tokens: int = 1024
    _client: OpenAI = field(init=False, repr=False)

    def __post_init__(self):
        from openai import OpenAI  # lazy: harness core imports without the SDK

        # bounded timeout + retries: a single wedged request must fail, not hang
        # the whole stage forever (one stuck query stalled an entire ablation stage).
        self._client = OpenAI(
            base_url=self.base_url, api_key=self.api_key,
            timeout=90.0, max_retries=2,
        )

    def complete(self, prompt: str, temperature: float | None = None) -> Completion:
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
        )
        dt = time.perf_counter() - t0
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            latency_s=dt,
        )
