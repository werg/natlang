"""Render a small typed document to HTML or plain text with host UI events."""
from __future__ import annotations

import html


class DocumentRenderer:
    def __init__(self):
        self.events = []

    def render(self, blocks: list[dict], fmt: str) -> str:
        if fmt not in ("html", "text"):
            raise ValueError("unsupported document format")
        parts = []
        for block in blocks:
            kind, value = block.get("kind"), block.get("text")
            if kind not in ("heading", "paragraph") or not isinstance(value, str):
                raise ValueError("invalid typed document block")
            if fmt == "html":
                tag = "h1" if kind == "heading" else "p"
                parts.append(f"<{tag}>{html.escape(value)}</{tag}>")
            else:
                parts.append(value)
        output = "\n".join(parts)
        self.events.append({"kind": "render", "format": fmt, "blocks": len(blocks), "bytes": len(output.encode())})
        return output

    def emit(self, event: str, payload: dict):
        self.events.append({"kind": "ui", "event": event, "payload": dict(payload)})
