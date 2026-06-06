"""Unit tests for configuration + model alias resolution."""

from __future__ import annotations

import re

from vertex_proxy.config import Settings


def test_defaults() -> None:
    s = Settings()
    assert s.host == "127.0.0.1"
    assert s.port == 8787
    assert s.anthropic_region == "us-east5"
    assert s.gemini_region == "us-central1"
    assert s.token_refresh_seconds == 3000


def test_anthropic_aliases_include_sonnet() -> None:
    s = Settings()
    assert "claude-sonnet-4-5-20250929" in s.anthropic_model_aliases
    assert s.anthropic_model_aliases["claude-sonnet-4-5-20250929"] == "claude-sonnet-4-5@20250929"


# Vertex uses two ID conventions, split at the Claude 4.6 generation:
#   * 4.6-gen and later (Opus 4.6/4.7/4.8, Sonnet 4.6) use a DATELESS bare ID;
#     appending '@<date>' makes the Vertex call 404.
#   * pre-4.6 (Sonnet 4.5/4, Opus 4.5, Haiku 4.5) carry a snapshot date that
#     Vertex separates with '@', e.g. claude-haiku-4-5@20251001.
_DATELESS_46_GEN = {
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
}


def test_anthropic_46_gen_aliases_are_dateless() -> None:
    """4.6-generation Vertex IDs are bare (no '@', no date); a suffix 404s."""
    s = Settings()
    for model in _DATELESS_46_GEN:
        vertex_id = s.anthropic_model_aliases[model]
        assert vertex_id == model, f"4.6-gen alias {model!r} must map to bare id, got {vertex_id!r}"
        assert "@" not in vertex_id, (
            f"4.6-gen alias {model!r} must NOT carry '@', got {vertex_id!r}"
        )


def test_anthropic_pre_46_aliases_use_at_sign_date() -> None:
    """Pre-4.6 Vertex IDs separate the snapshot date with '@', never '-'."""
    s = Settings()
    for alias, vertex_id in s.anthropic_model_aliases.items():
        if vertex_id in _DATELESS_46_GEN:
            continue
        assert "@" in vertex_id, f"pre-4.6 alias {alias!r} → {vertex_id!r} missing '@' date sep"
        # The eight digits after '@' must be the snapshot date.
        head, _, date = vertex_id.partition("@")
        assert date.isdigit() and len(date) == 8, f"{alias!r} → {vertex_id!r} bad date segment"
        # The head (model name) must NOT carry a '-YYYYMMDD' dash-date; on Vertex
        # the date is always '@'-separated, never '-' (that is the Claude-API form).
        assert not re.search(r"-\d{8}$", head), f"{alias!r} has a '-' date in {vertex_id!r}"


def test_sonnet_4_maps_to_at_sign_vertex_id() -> None:
    """Sonnet 4 (pre-4.6) must use the '@' Vertex id, not the Claude-API '-' id."""
    s = Settings()
    assert s.anthropic_model_aliases["claude-sonnet-4-20250514"] == "claude-sonnet-4@20250514"


def test_opus_45_and_haiku_45_use_correct_snapshot_dates() -> None:
    """Opus 4.5 = @20251101 and Haiku 4.5 = @20251001 on Vertex (not @20250929)."""
    s = Settings()
    assert s.anthropic_model_aliases["claude-opus-4-5"] == "claude-opus-4-5@20251101"
    assert s.anthropic_model_aliases["claude-haiku-4-5"] == "claude-haiku-4-5@20251001"


def test_gemini_aliases_include_pro_and_flash() -> None:
    s = Settings()
    assert "gemini-2.5-pro" in s.gemini_model_aliases
    assert "gemini-2.5-flash" in s.gemini_model_aliases


def test_maas_aliases_have_publisher_path_shape() -> None:
    """MaaS aliases must follow 'publishers/{vendor}/models/{id}' shape."""
    s = Settings()
    for alias, path in s.maas_model_aliases.items():
        parts = path.split("/")
        assert len(parts) == 4, f"maas alias {alias!r} → {path!r} not 4 segments"
        assert parts[0] == "publishers", f"maas alias {alias!r} must start with 'publishers/'"
        assert parts[2] == "models", f"maas alias {alias!r} must have 'models/' segment"


def test_known_maas_vendors_present() -> None:
    """Spot-check that each known vendor has at least one model."""
    s = Settings()
    vendors_seen = {path.split("/")[1] for path in s.maas_model_aliases.values()}
    expected = {"moonshotai", "zhipu", "minimax", "qwen", "xai"}
    missing = expected - vendors_seen
    assert not missing, f"missing MaaS vendor(s): {missing}"


def test_env_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("VERTEX_PROXY_PORT", "9999")
    monkeypatch.setenv("VERTEX_PROXY_ANTHROPIC_REGION", "europe-west4")
    s = Settings()
    assert s.port == 9999
    assert s.anthropic_region == "europe-west4"
