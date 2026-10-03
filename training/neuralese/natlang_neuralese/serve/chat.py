"""Chat requests to model input, and model output to chat messages, with Neuralese blocks (S4 §3–4).

Requests are OpenAI-style. Wherever the runtime carries a soft value, a message's `content` or a tool call's
`function.arguments` is an array of parts, `{"type": "text", "text": …}` and `{"type": "neuralese", "id": …}`
(ts-host/src/contracts.ts). Tool-call arguments travel as JSON text, so their parts are pieces of that text.

Rendering replaces every block with a private-use placeholder, applies the backbone's native chat template (LFM2.5
renders tool calls in its Pythonic form, `<|tool_call_start|>[eval(code='…')]<|tool_call_end|>`, so a literal sits
inside the quoted code string), then cuts the rendered text at the placeholders: text runs are tokenized, blocks
become `<|neuralese|>` + one position per vector + `<|/neuralese|>` for the read port.

Escaping (spec §3.3): control-token text inside message content or tool-call arguments is ordinary text. Rendering
wraps every occurrence of a special-token string in content between two private-use escape marks; the engine
tokenizes the wrapped text with special-token splitting, so only structure the template itself inserted becomes
special tokens.

Parsing does the reverse on what the model produced: blocks are placeholders while the Pythonic tool calls are
parsed, then every string that holds a placeholder becomes a part array again.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

_PH_OPEN, _PH_CLOSE = "", ""
_PH = re.compile(_PH_OPEN + r"(\d+)" + _PH_CLOSE)
ESC_OPEN, ESC_CLOSE = "\ue012", "\ue013"
_ESC = re.compile(ESC_OPEN + "(.*?)" + ESC_CLOSE, re.S)
_CALLS = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.S)
_THINK = re.compile(r"<think>(.*?)</think>", re.S)


class RequestError(ValueError):
    """A malformed request (HTTP 400). `code` is a stable diagnostic name."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


def placeholder(index: int) -> str:
    return f"{_PH_OPEN}{index}{_PH_CLOSE}"


def _is_parts(value) -> bool:
    return isinstance(value, list) and value and all(
        isinstance(p, dict) and p.get("type") in ("text", "neuralese") for p in value)


@dataclass
class Rendered:
    segments: list  # str (template text) or int (index into `blocks`)
    blocks: list[str]  # block IDs in order of appearance


def escape_specials(text: str, specials) -> str:
    """Wrap every special-token string in `text` in escape marks (longest first)."""
    if not specials or "<" not in text:
        return text
    pattern = _specials_pattern(tuple(specials))
    return pattern.sub(lambda m: ESC_OPEN + m.group(0) + ESC_CLOSE, text)


_PATTERNS: dict = {}


def _specials_pattern(specials: tuple):
    found = _PATTERNS.get(specials)
    if found is None:
        found = re.compile("|".join(re.escape(s) for s in sorted(set(specials), key=len, reverse=True) if s))
        _PATTERNS[specials] = found
    return found


def split_escaped(text: str) -> list[tuple[str, bool]]:
    """Template text → [(run, escaped)], escaped runs being content that must tokenize as plain text."""
    runs, last = [], 0
    for match in _ESC.finditer(text):
        if match.start() > last:
            runs.append((text[last:match.start()], False))
        runs.append((match.group(1), True))
        last = match.end()
    if last < len(text):
        runs.append((text[last:], False))
    return runs


def _escape_value(value, specials):
    if isinstance(value, str):
        return escape_specials(value, specials)
    if isinstance(value, list):
        return [_escape_value(v, specials) for v in value]
    if isinstance(value, dict):
        return {k: _escape_value(v, specials) for k, v in value.items()}
    return value


def render_messages(messages: list[dict], tools: list | None, apply_template, specials=()) -> Rendered:
    """Messages with block parts → template text cut into text runs and block references.

    `apply_template(messages, tools)` renders plain messages with the model's chat template and the generation prompt.
    `specials` are the tokenizer's special-token strings; occurrences inside content are escaped (spec §3.3).
    """
    blocks: list[str] = []

    def flatten(parts, escape: bool = True) -> str:
        out = []
        for part in parts:
            if part["type"] == "text":
                out.append(escape_specials(part.get("text") or "", specials) if escape else (part.get("text") or ""))
            else:
                block_id = part.get("id")
                if not isinstance(block_id, str) or not block_id.startswith("nz1_"):
                    raise RequestError("neuralese-bad-part", f"invalid block part {part!r}")
                out.append(placeholder(len(blocks)))
                blocks.append(block_id)
        return "".join(out)

    plain = []
    for message in messages:
        message = dict(message)
        if _is_parts(message.get("content")):
            message["content"] = flatten(message["content"])
        elif isinstance(message.get("content"), str):
            message["content"] = escape_specials(message["content"], specials)
        calls = []
        for call in message.get("tool_calls") or []:
            call = json.loads(json.dumps(call))
            fn = call.get("function", call)
            arguments = fn.get("arguments")
            if _is_parts(arguments):
                arguments = flatten(arguments, escape=False)
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError as error:
                    raise RequestError("tool-arguments", f"tool-call arguments are not JSON: {error}") from error
            fn["arguments"] = _escape_value(arguments, specials)
            calls.append(call)
        if calls:
            message["tool_calls"] = calls
        plain.append(message)
    text = apply_template(plain, tools)
    segments: list = []
    last = 0
    for match in _PH.finditer(text):
        if match.start() > last:
            segments.append(text[last:match.start()])
        segments.append(int(match.group(1)))
        last = match.end()
    if last < len(text):
        segments.append(text[last:])
    seen = sorted(s for s in segments if isinstance(s, int))
    if seen != list(range(len(blocks))):
        raise RequestError("neuralese-render", "the chat template dropped or duplicated a block")
    return Rendered(segments, blocks)


def _to_parts(text: str, block_ids: list[str]):
    """A string with placeholders → a part array; a string without them stays a string."""
    if not _PH.search(text):
        return text
    parts, last = [], 0
    for match in _PH.finditer(text):
        if match.start() > last:
            parts.append({"type": "text", "text": text[last:match.start()]})
        parts.append({"type": "neuralese", "id": block_ids[int(match.group(1))]})
        last = match.end()
    if last < len(text):
        parts.append({"type": "text", "text": text[last:]})
    return parts


def _restore(value, block_ids):
    if isinstance(value, str):
        return _to_parts(value, block_ids)
    if isinstance(value, list):
        return [_restore(v, block_ids) for v in value]
    if isinstance(value, dict):
        return {k: _restore(v, block_ids) for k, v in value.items()}
    return value


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_call_name(node.value)}.{node.attr}"
    raise ValueError("unsupported call target")


def parse_pythonic_calls(body: str) -> list[tuple[str, dict]]:
    """`[name(arg=literal, …), …]` → [(name, {arg: value})]. Raises ValueError on anything else."""
    tree = ast.parse(body.strip(), mode="eval").body
    nodes = tree.elts if isinstance(tree, (ast.List, ast.Tuple)) else [tree]
    calls = []
    for node in nodes:
        if not isinstance(node, ast.Call) or node.args:
            raise ValueError("tool calls take keyword arguments only")
        calls.append((_call_name(node.func), {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}))
    return calls


def build_message(text: str, block_ids: list[str], call_prefix: str = "call") -> dict:
    """Model output (with block placeholders) → an assistant message with parts where blocks are."""
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    reasoning = None
    think = _THINK.search(text)
    if think:
        reasoning = think.group(1).strip()
        text = text[:think.start()] + text[think.end():]
    tool_calls = []
    for match in _CALLS.finditer(text):
        try:
            calls = parse_pythonic_calls(match.group(1))
        except (ValueError, SyntaxError):
            continue  # left in the content; the runtime treats leaked markup as a malformed call
        for name, arguments in calls:
            tool_calls.append({
                "id": f"{call_prefix}_{len(tool_calls)}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(_restore(arguments, block_ids))},
            })
    content = _CALLS.sub(lambda m: "" if _parses(m.group(1)) else m.group(0), text).strip()
    message = {"role": "assistant", "content": _to_parts(content, block_ids) if content else None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _parses(body: str) -> bool:
    try:
        parse_pythonic_calls(body)
        return True
    except (ValueError, SyntaxError):
        return False
