"""Disk cache for LLM responses. PRD §11.3, NFR-2, NFR-6.

The cache is what protects free-tier quota and makes experiment reruns
deterministic, so key stability is the property that actually matters.
"""

from app.llm.cache import DiskCache, cache_key


def test_key_is_stable_across_calls():
    a = cache_key("gemini", "gemini-2.5-flash", "hello", {"temperature": 0.2}, 1234)
    b = cache_key("gemini", "gemini-2.5-flash", "hello", {"temperature": 0.2}, 1234)
    assert a == b


def test_key_ignores_param_dict_ordering():
    a = cache_key("gemini", "m", "p", {"temperature": 0.2, "top_p": 0.9}, 1)
    b = cache_key("gemini", "m", "p", {"top_p": 0.9, "temperature": 0.2}, 1)
    assert a == b


def test_key_changes_with_every_component():
    base = ("gemini", "m", "p", {"temperature": 0.2}, 1)
    baseline = cache_key(*base)
    variants = [
        ("ollama", "m", "p", {"temperature": 0.2}, 1),
        ("gemini", "m2", "p", {"temperature": 0.2}, 1),
        ("gemini", "m", "p2", {"temperature": 0.2}, 1),
        ("gemini", "m", "p", {"temperature": 0.9}, 1),
        ("gemini", "m", "p", {"temperature": 0.2}, 2),
    ]
    for v in variants:
        assert cache_key(*v) != baseline, f"key collided for {v}"


def test_key_is_hex_sha256():
    k = cache_key("gemini", "m", "p", {}, None)
    assert len(k) == 64
    assert all(c in "0123456789abcdef" for c in k)


def test_set_then_get_round_trips(tmp_path):
    cache = DiskCache(tmp_path)
    cache.set("abc123", {"text": "hi", "tokens_in": 3})
    assert cache.get("abc123") == {"text": "hi", "tokens_in": 3}


def test_get_returns_none_on_miss(tmp_path):
    assert DiskCache(tmp_path).get("nothing-here") is None


def test_survives_a_new_cache_instance(tmp_path):
    DiskCache(tmp_path).set("k", {"text": "persisted"})
    assert DiskCache(tmp_path).get("k") == {"text": "persisted"}


def test_disabled_cache_never_stores_or_reads(tmp_path):
    cache = DiskCache(tmp_path, enabled=False)
    cache.set("k", {"text": "v"})
    assert cache.get("k") is None
    assert list(tmp_path.rglob("*.json")) == []


def test_corrupt_entry_is_a_miss_not_a_crash(tmp_path):
    cache = DiskCache(tmp_path)
    cache.set("k", {"text": "v"})
    path = next(tmp_path.rglob("*.json"))
    path.write_text("{not json", encoding="utf-8")
    assert cache.get("k") is None


def test_creates_directory_if_absent(tmp_path):
    target = tmp_path / "deep" / "nested"
    DiskCache(target).set("k", {"text": "v"})
    assert DiskCache(target).get("k") == {"text": "v"}
