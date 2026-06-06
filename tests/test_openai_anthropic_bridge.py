"""Unit tests for the OpenAI ↔ Anthropic bridge translation functions."""

from __future__ import annotations

import json

from vertex_proxy.openai_anthropic_bridge import (
    anthropic_stream_to_openai_stream,
    anthropic_to_openai_response,
    openai_to_anthropic_body,
)

# --- Group A — openai_to_anthropic_body -------------------------------------


def test_body_extracts_and_joins_system_messages() -> None:
    body = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
        ],
    }
    out = openai_to_anthropic_body(body)
    assert out["system"] == "You are helpful.\n\nBe terse."
    # System messages are removed from the messages array.
    assert out["messages"] == [{"role": "user", "content": "hi"}]


def test_body_system_structured_text_parts() -> None:
    body = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "S1"}]},
            {"role": "user", "content": "hi"},
        ],
    }
    out = openai_to_anthropic_body(body)
    assert out["system"] == "S1"


def test_body_no_system_omits_system_key() -> None:
    body = {"messages": [{"role": "user", "content": "hi"}]}
    out = openai_to_anthropic_body(body)
    assert "system" not in out


def test_body_max_tokens_defaults_to_4096() -> None:
    body = {"messages": [{"role": "user", "content": "hi"}]}
    out = openai_to_anthropic_body(body)
    assert out["max_tokens"] == 4096


def test_body_max_tokens_explicit_wins() -> None:
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 256}
    out = openai_to_anthropic_body(body)
    assert out["max_tokens"] == 256


def test_body_max_completion_tokens_maps_to_max_tokens() -> None:
    body = {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 512}
    out = openai_to_anthropic_body(body)
    assert out["max_tokens"] == 512


def test_body_temperature_and_top_p_passed_through() -> None:
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.3,
        "top_p": 0.9,
    }
    out = openai_to_anthropic_body(body)
    assert out["temperature"] == 0.3
    assert out["top_p"] == 0.9


def test_body_stop_string_becomes_stop_sequences_list() -> None:
    body = {"messages": [{"role": "user", "content": "hi"}], "stop": "END"}
    out = openai_to_anthropic_body(body)
    assert out["stop_sequences"] == ["END"]


def test_body_stop_list_passed_through() -> None:
    body = {"messages": [{"role": "user", "content": "hi"}], "stop": ["A", "B"]}
    out = openai_to_anthropic_body(body)
    assert out["stop_sequences"] == ["A", "B"]


def test_body_stream_flag_passed_through() -> None:
    body = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    out = openai_to_anthropic_body(body)
    assert out["stream"] is True


def test_body_assistant_tool_calls_become_tool_use_blocks() -> None:
    body = {
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "let me check",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Oslo"}',
                        },
                    }
                ],
            },
        ],
    }
    out = openai_to_anthropic_body(body)
    assistant_msg = out["messages"][1]
    assert assistant_msg["role"] == "assistant"
    blocks = assistant_msg["content"]
    # Leading text block, then the tool_use block.
    assert blocks[0] == {"type": "text", "text": "let me check"}
    tool_block = blocks[1]
    assert tool_block["type"] == "tool_use"
    assert tool_block["id"] == "call_1"
    assert tool_block["name"] == "get_weather"
    # arguments JSON string is parsed into an input dict.
    assert tool_block["input"] == {"city": "Oslo"}


def test_body_assistant_tool_call_invalid_json_args_wrapped() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {"name": "f", "arguments": "not-json"},
                    }
                ],
            }
        ],
    }
    out = openai_to_anthropic_body(body)
    tool_block = out["messages"][0]["content"][0]
    assert tool_block["input"] == {"raw": "not-json"}


def test_body_tool_role_becomes_tool_result() -> None:
    body = {
        "messages": [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "sunny, 18C",
            }
        ],
    }
    out = openai_to_anthropic_body(body)
    msg = out["messages"][0]
    assert msg["role"] == "user"
    block = msg["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_1"
    assert block["content"] == "sunny, 18C"


# --- Group B — anthropic_to_openai_response ---------------------------------


def test_response_extracts_text() -> None:
    resp = {
        "id": "msg_abc",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    out = anthropic_to_openai_response(resp, "claude-opus-4-8")
    choice = out["choices"][0]
    assert choice["message"]["content"] == "hello"
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] == "stop"
    assert out["model"] == "claude-opus-4-8"
    assert out["object"] == "chat.completion"
    assert out["id"] == "chatcmpl-msg_abc"


def test_response_tool_use_maps_to_tool_calls() -> None:
    resp = {
        "id": "msg_t",
        "content": [
            {"type": "text", "text": "checking"},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Oslo"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 7},
    }
    out = anthropic_to_openai_response(resp, "claude-opus-4-8")
    choice = out["choices"][0]
    msg = choice["message"]
    # Text preserved alongside tool_calls.
    assert msg["content"] == "checking"
    tool_calls = msg["tool_calls"]
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["id"] == "toolu_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    # arguments must be a JSON string, not a dict.
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"city": "Oslo"}
    assert choice["finish_reason"] == "tool_calls"


def test_response_tool_use_only_sets_content_none() -> None:
    resp = {
        "id": "msg_t2",
        "content": [
            {"type": "tool_use", "id": "toolu_2", "name": "f", "input": {}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    out = anthropic_to_openai_response(resp, "m")
    msg = out["choices"][0]["message"]
    assert msg["content"] is None
    assert msg["tool_calls"][0]["function"]["arguments"] == "{}"
    assert out["choices"][0]["finish_reason"] == "tool_calls"


def test_response_no_tool_calls_key_when_absent() -> None:
    resp = {
        "id": "msg_x",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    out = anthropic_to_openai_response(resp, "m")
    assert "tool_calls" not in out["choices"][0]["message"]


def test_response_usage_mapping() -> None:
    resp = {
        "id": "msg_u",
        "content": [{"type": "text", "text": "x"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 11, "output_tokens": 4},
    }
    out = anthropic_to_openai_response(resp, "m")
    usage = out["usage"]
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == 15


def test_response_finish_reason_map() -> None:
    cases = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
    }
    for anthropic_reason, openai_reason in cases.items():
        resp = {
            "id": "m",
            "content": [{"type": "text", "text": "x"}],
            "stop_reason": anthropic_reason,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        out = anthropic_to_openai_response(resp, "m")
        assert out["choices"][0]["finish_reason"] == openai_reason


# --- Group C — anthropic_stream_to_openai_stream ----------------------------


def _parse_sse_data(out: bytes) -> dict:
    """Extract the first `data: {json}` payload from an SSE byte string."""
    text = out.decode("utf-8")
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        if chunk.startswith("data: ") and not chunk.endswith("[DONE]"):
            return json.loads(chunk[len("data: ") :])
    raise AssertionError(f"no JSON data line in {out!r}")


def test_stream_content_block_delta_text_becomes_chunk() -> None:
    line = b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}'
    out = anthropic_stream_to_openai_stream(line, "m", stream_id="chatcmpl-fixed")
    assert out is not None
    payload = _parse_sse_data(out)
    assert payload["object"] == "chat.completion.chunk"
    assert payload["id"] == "chatcmpl-fixed"
    assert payload["choices"][0]["delta"]["content"] == "Hi"
    assert payload["choices"][0]["finish_reason"] is None


def test_stream_message_stop_emits_only_done() -> None:
    line = b'data: {"type":"message_stop"}'
    out = anthropic_stream_to_openai_stream(line, "m", stream_id="chatcmpl-fixed")
    assert out == b"data: [DONE]\n\n"


def test_stream_message_delta_carries_finish_reason() -> None:
    line = b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}'
    out = anthropic_stream_to_openai_stream(line, "m", stream_id="chatcmpl-fixed")
    payload = _parse_sse_data(out)
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["choices"][0]["delta"] == {}
    assert payload["id"] == "chatcmpl-fixed"


def test_stream_message_delta_max_tokens_maps_to_length() -> None:
    line = b'data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"}}'
    out = anthropic_stream_to_openai_stream(line, "m", stream_id="chatcmpl-fixed")
    payload = _parse_sse_data(out)
    assert payload["choices"][0]["finish_reason"] == "length"


def test_stream_finish_reason_emitted_exactly_once() -> None:
    """Regression: message_delta is the sole finish_reason carrier; message_stop
    must NOT emit a second finish_reason chunk (only the [DONE] sentinel)."""
    delta_line = b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}'
    stop_line = b'data: {"type":"message_stop"}'
    delta_out = anthropic_stream_to_openai_stream(delta_line, "m", stream_id="s")
    stop_out = anthropic_stream_to_openai_stream(stop_line, "m", stream_id="s")
    # message_delta has exactly one non-null finish_reason.
    assert b'"finish_reason": "stop"' in delta_out or b'"finish_reason":"stop"' in delta_out
    # message_stop produces no chunk JSON, only the DONE sentinel.
    assert stop_out == b"data: [DONE]\n\n"
    assert b"chat.completion.chunk" not in stop_out


def test_stream_done_passthrough() -> None:
    out = anthropic_stream_to_openai_stream(b"data: [DONE]", "m", stream_id="s")
    assert out == b"data: [DONE]\n\n"


def test_stream_event_type_lines_skipped() -> None:
    assert anthropic_stream_to_openai_stream(b"event: message_start", "m") is None


def test_stream_blank_and_unknown_lines_skipped() -> None:
    assert anthropic_stream_to_openai_stream(b"", "m") is None
    assert anthropic_stream_to_openai_stream(b'data: {"type":"ping"}', "m") is None


def test_stream_id_is_stable_across_chunks_when_passed() -> None:
    sid = "chatcmpl-abc123"
    a = anthropic_stream_to_openai_stream(
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"a"}}',
        "m",
        stream_id=sid,
    )
    b = anthropic_stream_to_openai_stream(
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"b"}}',
        "m",
        stream_id=sid,
    )
    assert _parse_sse_data(a)["id"] == sid
    assert _parse_sse_data(b)["id"] == sid
