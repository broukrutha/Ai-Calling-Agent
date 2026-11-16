"""
Check that the configured LLM providers accept your keys.

    python scripts/check_llm.py

Sends one tiny request through each provider in the fallback chain. Keys are
never printed.
"""

import asyncio
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_agent import config  # noqa: E402
from voice_agent.sarvam_client import build_provider_chain  # noqa: E402


async def main() -> int:
    providers = build_provider_chain()
    if not providers:
        print("No LLM provider is configured. Set OPENAI_API_KEY, GROQ_API_KEY or SARVAM_API_KEY in .env.")
        return 1

    failures = 0
    async with httpx.AsyncClient(timeout=30.0, trust_env=config.HTTP_TRUST_ENV) as client:
        for provider in providers:
            payload = {
                "model": provider.model,
                "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                "max_tokens": 5,
                **provider.extra_body,
            }
            try:
                resp = await client.post(provider.url, json=payload, headers=provider.headers)
            except httpx.HTTPError as exc:
                print(f"  FAIL  {provider.name}/{provider.model}: {type(exc).__name__}")
                failures += 1
                continue
            if resp.status_code == 200:
                print(f"  ok    {provider.name}/{provider.model}")
            else:
                hint = {401: "the key was rejected", 429: "rate limit or quota exceeded"}.get(resp.status_code, "")
                print(f"  FAIL  {provider.name}/{provider.model}: HTTP {resp.status_code} {hint}".rstrip())
                failures += 1
    return 1 if failures == len(providers) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
