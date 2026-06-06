"""OpenAI → Anthropic bridge functions for vertex-proxy.

Limitation: streaming tool calls are not translated. Anthropic streams tool
use as content_block_start + input_json_delta events, which require stateful
accumulation the stateless line translator does not do; those events are
skipped. Non-streaming requests return tool calls correctly. Use a
non-streaming request when you need tool_calls back.
"""

import json
import time
import uuid
from typing import Any


def openai_to_anthropic_body(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions request body to Anthropic Messages format."""
    messages = body.get("messages", [])

    # Extract system message(s) — Anthropic wants them as a top-level 'system' field.
    system_parts = []
    non_system_messages = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                # Handle structured content
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part["text"])
                    elif isinstance(part, str):
                        system_parts.append(part)
        else:
            # Map 'tool' role to Anthropic format if needed
            anthropic_msg = _convert_message(msg)
            if anthropic_msg:
                non_system_messages.append(anthropic_msg)

    anthropic_body: dict[str, Any] = {}

    if system_parts:
        anthropic_body["system"] = "\n\n".join(system_parts)

    anthropic_body["messages"] = non_system_messages

    # Map parameters
    if "max_tokens" in body:
        anthropic_body["max_tokens"] = body["max_tokens"]
    elif "max_completion_tokens" in body:
        anthropic_body["max_tokens"] = body["max_completion_tokens"]
    else:
        anthropic_body["max_tokens"] = 4096  # Anthropic requires max_tokens

    if "temperature" in body:
        anthropic_body["temperature"] = body["temperature"]
    if "top_p" in body:
        anthropic_body["top_p"] = body["top_p"]
    if "stop" in body:
        stop = body["stop"]
        if isinstance(stop, str):
            anthropic_body["stop_sequences"] = [stop]
        elif isinstance(stop, list):
            anthropic_body["stop_sequences"] = stop

    # Stream passthrough
    if body.get("stream"):
        anthropic_body["stream"] = True

    return anthropic_body


def _convert_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a single OpenAI message to Anthropic format."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "assistant":
        anthropic_msg: dict[str, Any] = {"role": "assistant"}
        if content:
            anthropic_msg["content"] = content
        # Handle tool_calls from assistant
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            blocks = []
            if content and isinstance(content, str):
                blocks.append({"type": "text", "text": content})
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args,
                    }
                )
            anthropic_msg["content"] = blocks
        return anthropic_msg

    elif role == "tool":
        # OpenAI tool result → Anthropic tool_result
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content if isinstance(content, str) else json.dumps(content),
                }
            ],
        }

    elif role == "user":
        return {"role": "user", "content": content if content else ""}

    return None


def anthropic_to_openai_response(anthropic_resp: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert an Anthropic Messages response to OpenAI Chat Completions format."""
    content_blocks = anthropic_resp.get("content", [])
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content_blocks:
        if isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )
        elif isinstance(block, str):
            text_parts.append(block)

    text_content = "\n".join(text_parts) if text_parts else None

    # Map stop reason. If Anthropic emitted tool_use blocks, force tool_calls
    # regardless of stop_reason so finish_reason and the tool_calls array agree.
    stop_reason = anthropic_resp.get("stop_reason", "end_turn")
    finish_reason_map = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }
    finish_reason = finish_reason_map.get(stop_reason, "stop")
    if tool_calls:
        finish_reason = "tool_calls"

    usage = anthropic_resp.get("usage", {})
    openai_usage = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
    }

    message: dict[str, Any] = {
        "role": "assistant",
        # OpenAI convention: content is null when only tool_calls are present.
        "content": text_content,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{anthropic_resp.get('id', uuid.uuid4().hex[:24])}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": openai_usage,
    }


def anthropic_stream_to_openai_stream(
    line: bytes, model: str, stream_id: str | None = None
) -> bytes | None:
    """Convert a single Anthropic SSE line to OpenAI SSE format.

    Returns None if the line should be skipped. `stream_id` should be a single
    id generated once per stream and passed for every line; if omitted a stable
    fallback id is generated per call (acceptable but not cross-chunk-stable).
    """
    _stream_id = stream_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    text = line.decode("utf-8", errors="replace").strip()
    if not text or not text.startswith("data: "):
        if text.startswith("event: "):
            return None  # Skip Anthropic event type lines
        return None

    data_str = text[6:]  # Remove "data: " prefix
    if data_str == "[DONE]":
        return b"data: [DONE]\n\n"

    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return None

    event_type = data.get("type", "")

    if event_type == "content_block_delta":
        delta = data.get("delta", {})
        if delta.get("type") == "text_delta":
            openai_chunk = {
                "id": _stream_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": delta.get("text", "")},
                        "finish_reason": None,
                    }
                ],
            }
            return f"data: {json.dumps(openai_chunk)}\n\n".encode()
        # NOTE: input_json_delta (streamed tool_use args) is intentionally not
        # translated; see module docstring. Falls through to `return None`.

    elif event_type == "message_stop":
        # Terminal sentinel only. finish_reason was already emitted on the
        # preceding message_delta chunk; emitting it here too would be a
        # second (invalid) finish_reason. Just close the stream.
        return b"data: [DONE]\n\n"

    elif event_type == "message_delta":
        # Carries the final stop_reason (and usage). This is the single chunk
        # that sets finish_reason for the whole OpenAI stream.
        stop_reason = data.get("delta", {}).get("stop_reason", "end_turn")
        finish_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }
        openai_chunk = {
            "id": _stream_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason_map.get(stop_reason, "stop"),
                }
            ],
        }
        return f"data: {json.dumps(openai_chunk)}\n\n".encode()

    return None
