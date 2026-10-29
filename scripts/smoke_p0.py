"""P0 exit check: one Gemini call and one Ollama call through the SAME interface.

This is the only thing that proves PRD §5 P0 is actually done. Everything else
in P0 is verified by `make test` with no network.

Usage:
    make smoke              # both providers
    make smoke ARGS=--gemini-only
    make smoke ARGS=--ollama-only

Exit code 0 = P0 exit criterion met. Non-zero = not met, reason printed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app.core.config import load_config  # noqa: E402
from app.llm.base import LLMError  # noqa: E402
from app.llm.cache import DiskCache  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.llm.limiter import SlidingWindow  # noqa: E402
from app.llm.providers.gemini import GeminiProvider  # noqa: E402
from app.llm.providers.ollama import OllamaProvider  # noqa: E402

PROMPT = (
    "Summarize in one short sentence, and nothing else:\n\n"
    "Transformers replaced recurrence with self-attention, which lets every "
    "token attend to every other token in a single layer."
)

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}  PASS{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"{RED}  FAIL{RESET}  {msg}")


async def check_provider(name: str, provider, profile, cache_dir: Path) -> bool:
    """One cold call, then one identical call that must come from cache.

    The cache dir is run-scoped: a warm cache left by a previous run would make
    the FIRST call a hit too, and the check would prove nothing about whether a
    live call still works.
    """
    print(f"\n[{name}] model={profile.model}")

    calls: list[dict] = []
    client = LLMClient(
        provider=provider,
        profile=profile,
        cache=DiskCache(cache_dir / name),
        rpm_bucket=SlidingWindow(profile.limits.rpm),
        call_log=calls.append,
    )

    try:
        first = await client.generate(PROMPT, purpose="smoke", seed=1234)
    except LLMError as exc:
        fail(f"{type(exc).__name__}: {exc}")
        return False

    if not first.text.strip():
        fail("provider returned empty text")
        return False

    ok(f"live call -> {first.tokens_in} in / {first.tokens_out} out tokens")
    print(f"{DIM}        {first.text.strip()[:160]}{RESET}")

    second = await client.generate(PROMPT, purpose="smoke", seed=1234)
    if not second.cache_hit:
        fail("identical second call was NOT served from cache (quota at risk)")
        return False
    ok("identical second call served from cache (0 quota)")

    if len(calls) != 2 or calls[0]["cache_hit"] or not calls[1]["cache_hit"]:
        fail(f"llm_calls log wrong: {calls}")
        return False
    ok(f"llm_calls logged {len(calls)} entries with correct cache flags (FR-7.4)")

    return True


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini-only", action="store_true")
    parser.add_argument("--ollama-only", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = load_config(ROOT / "configs")
    # Run-scoped so every invocation makes one genuinely live call.
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    cache_dir = Path(os.getenv("LLM_CACHE_DIR", ROOT / ".cache")) / "smoke" / run_id

    print("=" * 66)
    print("P0 EXIT CHECK - one Gemini call and one Ollama call, one interface")
    print("=" * 66)

    results: dict[str, bool] = {}

    if not args.ollama_only:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            print("\n[gemini]")
            fail("GEMINI_API_KEY is empty or missing from .env  (step M2)")
            results["gemini"] = False
        else:
            results["gemini"] = await check_provider(
                "gemini",
                GeminiProvider(api_key=key),
                cfg.profiles["gemini_full"],
                cache_dir,
            )

    if not args.gemini_only:
        results["ollama"] = await check_provider(
            "ollama",
            OllamaProvider(
                base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
                keep_alive=0,  # DECISION-3: evict from the 8 GB GPU straight away
            ),
            cfg.profiles["qwen3_8b_local"],
            cache_dir,
        )

    print("\n" + "=" * 66)
    for name, passed in results.items():
        print(f"  {name:<8} {'PASS' if passed else 'FAIL'}")

    if all(results.values()):
        print(f"\n{GREEN}P0 EXIT CRITERION MET{RESET}")
        return 0
    print(f"\n{RED}P0 exit criterion NOT met{RESET} - see failures above")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
