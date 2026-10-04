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
import base64
import json
import re
import secrets
from dataclasses import dataclass

_PH_OPEN, _PH_CLOSE = "", ""
_PH = re.compile(_PH_OPEN + r"(\d+)" + _PH_CLOSE)
ESC_OPEN, ESC_CLOSE = "\ue012", "\ue013"
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
    escape_nonce: str = ""


def escape_specials(text: str, specials, nonce: str = "") -> str:
    """Wrap every special-token string in `text` in escape marks (longest first)."""
    if not specials or "<" not in text:
        return text
    pattern = _specials_pattern(tuple(specials))
    return pattern.sub(lambda m: ESC_OPEN + nonce + ':' + base64.b64encode(m.group(0).encode()).decode() + ESC_CLOSE, text)


_PATTERNS: dict = {}


def _specials_pattern(specials: tuple):
    found = _PATTERNS.get(specials)
    if found is None:
        found = re.compile("|".join(re.escape(s) for s in sorted(set(specials), key=len, reverse=True) if s))
        _PATTERNS[specials] = found
    return found


def split_escaped(text: str, nonce: str = "") -> list[tuple[str, bool]]:
    """Template text → [(run, escaped)], escaped runs being content that must tokenize as plain text."""
    runs, last = [], 0
    pattern = re.compile(re.escape(ESC_OPEN + nonce + ':') + r'([A-Za-z0-9+/]*={0,2})' + re.escape(ESC_CLOSE))
    for match in pattern.finditer(text):
        if match.start() > last:
            runs.append((text[last:match.start()], False))
        runs.append((base64.b64decode(match.group(1), validate=True).decode(), True))
        last = match.end()
    if last < len(text):
        runs.append((text[last:], False))
    return runs


def _escape_value(value, specials, nonce=""):
    if isinstance(value, str):
        return escape_specials(value, specials, nonce)
    if isinstance(value, list):
        return [_escape_value(v, specials, nonce) for v in value]
    if isinstance(value, dict):
        return {k: _escape_value(v, specials, nonce) for k, v in value.items()}
    return value


def render_messages(messages: list[dict], tools: list | None, apply_template, specials=()) -> Rendered:
    """Messages with block parts → template text cut into text runs and block references.

    `apply_template(messages, tools)` renders plain messages with the model's chat template and the generation prompt.
    `specials` are the tokenizer's special-token strings; occurrences inside content are escaped (spec §3.3).
    """
    blocks: list[str] = []
    # An input cannot manufacture our internal placeholders. Choose a nonce
    # absent from every supplied string; private-use characters remain ordinary text.
    supplied = json.dumps([messages, tools], ensure_ascii=False)
    nonce = secrets.token_hex(16)
    while nonce in supplied:
        nonce = secrets.token_hex(16)
    input_prefix = _PH_OPEN + nonce + ':'
    input_pattern = re.compile(re.escape(input_prefix) + r'(\d+)' + re.escape(_PH_CLOSE))

    def flatten(parts, escape: bool = True) -> str:
        out = []
        for part in parts:
            if part["type"] == "text":
                out.append(escape_specials(part.get("text") or "", specials, nonce) if escape else (part.get("text") or ""))
            else:
                block_id = part.get("id")
                if not isinstance(block_id, str) or not block_id.startswith("nz1_"):
                    raise RequestError("neuralese-bad-part", f"invalid block part {part!r}")
                out.append(f"{input_prefix}{len(blocks)}{_PH_CLOSE}")
                blocks.append(block_id)
        return "".join(out)

    plain = []
    for message in messages:
        message = dict(message)
        if _is_parts(message.get("content")):
            message["content"] = flatten(message["content"])
        elif isinstance(message.get("content"), str):
            message["content"] = escape_specials(message["content"], specials, nonce)
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
            fn["arguments"] = _escape_value(arguments, specials, nonce)
            calls.append(call)
        if calls:
            message["tool_calls"] = calls
        plain.append(message)
    text = apply_template(plain, tools)
    segments: list = []
    last = 0
    for match in input_pattern.finditer(text):
        if match.start() > last:
            segments.append(text[last:match.start()])
        segments.append(int(match.group(1)))
        last = match.end()
    if last < len(text):
        segments.append(text[last:])
    seen = sorted(s for s in segments if isinstance(s, int))
    if seen != list(range(len(blocks))):
        raise RequestError("neuralese-render", "the chat template dropped or duplicated a block")
    return Rendered(segments, blocks, nonce)


_VALUE = "natlangValue7f3a9c"  # printable, so no template escapes it; stands for a call argument's value while the template renders the call


def call_reply(apply_template, name: str, arguments: dict, argument: str = "value", quoted: bool = True) -> tuple[str, str]:
    """The model's own rendering of an assistant reply that calls `name` with `arguments` and then `argument`, cut at
    that argument's value: (prefix, suffix). `apply_template(messages, add_generation_prompt)` is the chat template.

    With `quoted` the value is a string, as a written block's placeholder is: the prefix ends with the opening quote
    and the suffix starts with the closing one and runs through the end of the turn. Without, the prefix ends where
    the value starts (for decoding the value) and the suffix is empty. Template readout (the `neuralese_template`
    request field) and the trajectory trainer's write sites both force replies cut this way, so what runs is what
    is trained."""
    opening = [{"role": "user", "content": "x"}]
    call = {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": name, "arguments": {**arguments, argument: _VALUE}}}]}
    prompt = apply_template(opening, True)
    full = apply_template(opening + [call], False)
    if not full.startswith(prompt) or _VALUE not in full[len(prompt):]:
        raise RequestError("neuralese-template", "the chat template renders the call's reply differently")
    reply = full[len(prompt):]
    at = reply.index(_VALUE)
    prefix, suffix = reply[:at], reply[at + len(_VALUE):]
    if quoted:
        return prefix, suffix
    if not prefix or prefix[-1] not in "'\"":
        raise RequestError("neuralese-template", "the chat template does not quote string arguments")
    return prefix[:-1], ""


def _to_parts(text: str, block_ids: list[str]):
    """A string with placeholders → a part array; a string without them stays a string."""
    if not _PH.search(text):
        return text
    parts, last = [], 0
    for match in _PH.finditer(text):
        if int(match.group(1)) >= len(block_ids):
            continue  # Literal/model-written private-use text is not a block reference.
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
