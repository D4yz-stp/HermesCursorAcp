"""Rewrite flat tool-call objects into the function-call shape Hermes parses."""

from __future__ import annotations

import json

_OPEN_TAG = "<tool_call>"
_CLOSE_TAG = "</tool_call>"


def normalize_tool_call_text(text: str) -> str:
    """Return text with each flat tool-call object rewritten."""
    decoder = json.JSONDecoder()
    parts: list[str] = []
    cursor = 0
    while True:
        start = text.find(_OPEN_TAG, cursor)
        if start < 0:
            parts.append(text[cursor:])
            break
        parts.append(text[cursor:start])
        block_end, replacement = _rewrite_block(text, start, decoder)
        if replacement is None:
            parts.append(text[start:block_end])
        else:
            parts.append(replacement)
        cursor = block_end
    return "".join(parts)


def _rewrite_block(
    text: str, start: int, decoder: json.JSONDecoder
) -> tuple[int, str | None]:
    """Return ``(end, replacement)`` for the block at ``start``.

    ``replacement`` is None when the block stays as it is. ``end`` is the
    index just after the closing tag, or ``len(text)`` when the tag never closes.
    """
    body = start + len(_OPEN_TAG)
    index = body
    while index < len(text) and text[index].isspace():
        index += 1
    if index < len(text) and text[index] == "{":
        try:
            obj, value_end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            obj = None
            value_end = None
        if value_end is not None:
            after = value_end
            while after < len(text) and text[after].isspace():
                after += 1
            close_end = after + len(_CLOSE_TAG)
            if text.startswith(_CLOSE_TAG, after):
                rewritten = _flat_object_as_function_call(obj)
                if rewritten is None:
                    return close_end, None
                return close_end, f"{_OPEN_TAG}{rewritten}{_CLOSE_TAG}"
    close_at = text.find(_CLOSE_TAG, body)
    if close_at < 0:
        return len(text), None
    return close_at + len(_CLOSE_TAG), None


def _flat_object_as_function_call(obj: object) -> str | None:
    """Compact function-call JSON, or None when ``obj`` is not a flat call."""
    if not isinstance(obj, dict) or "function" in obj:
        return None
    name = obj.get("name")
    if not isinstance(name, str):
        return None
    if "arguments" not in obj:
        arguments = "{}"
    elif isinstance(obj["arguments"], str):
        arguments = obj["arguments"]
    else:
        arguments = json.dumps(obj["arguments"], separators=(",", ":"))
    return json.dumps(
        {"type": "function", "function": {"name": name, "arguments": arguments}},
        separators=(",", ":"),
    )
