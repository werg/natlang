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
_JSON_CALLS = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)  # Qwen-family templates (Maple)


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


def render_messages(messages: list[dict], tools: list | None, apply_template, specials=(), block_type=None) -> Rendered:
    """Messages with block parts → template text cut into text runs and block references.

    `apply_template(messages, tools)` renders plain messages with the model's chat template and the generation prompt.
    `specials` are the tokenizer's special-token strings; occurrences inside content are escaped (spec §3.3).
    """
    blocks: list[str] = []
    value_types, native_values = {}, set()
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
                value_types[len(blocks)] = part.get("value_type") or (block_type(block_id) if block_type else None)
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

            def inline(value):
                # An argument value that is a part list holding a block (as replies return a written value:
                # `"value": [{"type": "neuralese", "id": …}]`) renders as that block inside the value's string.
                if _is_parts(value) and any(part["type"] == "neuralese" for part in value):
                    value = flatten(value, escape=False)
                    marker = input_pattern.fullmatch(value)
                    if marker and value_types.get(int(marker.group(1))) == "unknown":
                        native_values.add(value)
                    return value
                if isinstance(value, str):
                    marker = input_pattern.fullmatch(value)
                    if marker and value_types.get(int(marker.group(1))) == "unknown":
                        native_values.add(value)
                    return value
                if isinstance(value, list):
                    return [inline(item) for item in value]
                if isinstance(value, dict):
                    return {key: inline(item) for key, item in value.items()}
                return value

            fn["arguments"] = _escape_value(inline(arguments), specials, nonce)
            calls.append(call)
        if calls:
            message["tool_calls"] = calls
        plain.append(message)
    text = apply_template(plain, tools)
    for marker in native_values:
        text = re.sub(r"([\"'])" + re.escape(marker) + r"\1", lambda match: marker, text)
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



_REASONING = "\u0001reasoning\u0001"  # stands for an empty reasoning block while a template renders a turn


def opens_thinking(generation_prompt: str) -> bool:
    """Thinking templates (Qwen3-family, Maple) open `<think>` in the generation prompt."""
    return generation_prompt.rstrip("\n").endswith("<think>")


def render_with_empty_thought(render, messages: list[dict]) -> str:
    """`render(messages)` with a final assistant turn that has no reasoning given an empty think block. Thinking
    templates render such a past turn bare, but on policy the model closes an empty `<think>` block before replying,
    so a teacher-forced reply must continue the generation prompt. Templates that ignore reasoning are unchanged."""
    last = messages[-1]
    if last.get("role") != "assistant" or last.get("reasoning_content") or "</think>" in str(last.get("content") or ""):
        return render(messages)
    text = render(messages[:-1] + [{**last, "reasoning_content": _REASONING}])
    return text.replace(_REASONING, "") if _REASONING in text else render(messages)


HISTORY_REASONING_POLICIES = ("keep", "last_turn_only")
BACKBONE_HISTORY_REASONING = {"lfm2": "last_turn_only", "mellum": "keep"}
"""Declared, backbone-inherent history-reasoning policy per backbone (`config.model_type`; owner 2026-10-10). The
reasoning of the turn being trained or generated is rendered for every line; what differs is earlier assistant
turns' reasoning in the history, which follows how each backbone was post-trained with its own template default:
LFM2.5 drops it (`last_turn_only`: its template keeps only the last assistant turn's reasoning unless
`preserve_thinking`/`keep_past_thinking` is set, which natlang never sets); Mellum2.1 keeps it (`keep`: its
template renders every assistant turn's reasoning after the last user query, without a switch). Training text and
serving prompts render through `bind_history_reasoning`, which pins the template switch to the declared policy and
asserts with a rendering probe that the template behaves as declared."""
_HISTORY_REASONING_SWITCHES = ("preserve_thinking", "keep_past_thinking")
_PROBE_EARLIER, _PROBE_LATEST = "natlang-history-reasoning-probe-earlier", "natlang-history-reasoning-probe-latest"
_PROBE_EXPECTED = {"keep": "keep", "last_turn_only": "drops-history"}


def history_reasoning_kwargs(chat_template, policy: str) -> dict:
    """The chat-template keyword arguments that pin `chat_template` to `policy`: a template with a switch gets it
    set explicitly (true for 'keep', false for 'last_turn_only'); a template without one gets none."""
    if policy not in HISTORY_REASONING_POLICIES:
        raise ValueError(f"unknown history_reasoning policy {policy!r} (declared: {HISTORY_REASONING_POLICIES})")
    if isinstance(chat_template, str):
        for name in _HISTORY_REASONING_SWITCHES:
            if name in chat_template:
                return {name: policy == "keep"}
    return {}


def declared_history_reasoning(model_type) -> str | None:
    """The declared policy of a backbone (`config.model_type`), or None for an undeclared one."""
    return BACKBONE_HISTORY_REASONING.get(model_type) if isinstance(model_type, str) else None


def tokenizer_model_type(tokenizer) -> str | None:
    """`model_type` of the checkpoint a tokenizer was loaded from (its directory's config.json), when it has one."""
    import pathlib
    path = getattr(tokenizer, "name_or_path", None)
    config = pathlib.Path(path) / "config.json" if isinstance(path, str) and path else None
    if config is None or not config.is_file():
        return None
    try:
        return json.loads(config.read_text(encoding="utf-8")).get("model_type")
    except (OSError, ValueError):
        return None


def probe_history_reasoning(render) -> str:
    """Render a tool-loop conversation (task, reasoned step, tool result, reasoned step) with `render(messages)` and
    classify it: 'keep' (both reasonings rendered), 'drops-history' (only the latest turn's reasoning rendered) or
    'not-rendered' (the template renders no reasoning)."""
    text = render([{"role": "user", "content": "probe task"},
                   {"role": "assistant", "reasoning_content": _PROBE_EARLIER, "content": "probe step"},
                   {"role": "tool", "content": "probe result"},
                   {"role": "assistant", "reasoning_content": _PROBE_LATEST, "content": "probe done"}])
    if _PROBE_EARLIER in text and _PROBE_LATEST in text:
        return "keep"
    return "drops-history" if _PROBE_LATEST in text else "not-rendered"


def bind_history_reasoning(tokenizer, policy: str | None = None, *, model_type: str | None = None,
                           require_declared: bool = False):
    """Bind a backbone's declared history-reasoning policy to `tokenizer.apply_chat_template` (idempotent) and record
    it as `tokenizer.natlang_history_reasoning` ({"policy", "backbone", "template_kwargs", "probe"}).

    `policy` defaults to the declaration for `model_type` (default: the tokenizer checkpoint's config.json). The
    template switch, when it has one, is pinned to the policy, and a caller passing a contrary value raises. A
    template whose probe contradicts the declared policy raises ValueError. An undeclared backbone (`policy`
    "undeclared") keeps its template's behaviour, recorded by the probe; `require_declared` makes it an error."""
    model_type = model_type if model_type is not None else tokenizer_model_type(tokenizer)
    declared = policy if policy is not None else declared_history_reasoning(model_type)
    bound = getattr(tokenizer, "natlang_history_reasoning", None)
    if isinstance(bound, dict):
        if declared is not None and bound.get("policy") != declared:
            raise ValueError(f"tokenizer is bound to history_reasoning={bound.get('policy')!r}, not {declared!r}")
        if require_declared and bound.get("policy") == "undeclared":
            raise ValueError(f"backbone {model_type!r} declares no history_reasoning policy (chat.BACKBONE_HISTORY_REASONING)")
        return tokenizer
    if declared is None and require_declared:
        raise ValueError(f"backbone {model_type!r} declares no history_reasoning policy (chat.BACKBONE_HISTORY_REASONING)")
    kwargs = {} if declared is None else history_reasoning_kwargs(getattr(tokenizer, "chat_template", None), declared)
    original = tokenizer.apply_chat_template

    def apply_chat_template(conversation, *args, **options):
        for name, value in kwargs.items():
            if options.get(name, value) != value:
                raise ValueError(f"history_reasoning={declared} pins chat-template {name}={value!r}")
        return original(conversation, *args, **{**options, **kwargs})

    probe = probe_history_reasoning(lambda m: apply_chat_template(m, tokenize=False, add_generation_prompt=False))
    if declared is not None and probe != _PROBE_EXPECTED[declared]:
        raise ValueError(f"chat template of backbone {model_type!r} renders history reasoning as {probe!r}, "
                         f"not as its declared history_reasoning={declared}")
    tokenizer.apply_chat_template = apply_chat_template
    tokenizer.natlang_history_reasoning = {"policy": declared or "undeclared", "backbone": model_type,
                                           "template_kwargs": dict(kwargs), "probe": probe}
    return tokenizer


def assistant_reply(apply_template, message: dict) -> str | None:
    """The text a model generates for assistant `message` after the generation prompt of a one-turn conversation, or
    None when the template's rendering of the turn does not continue that prompt. Thinking templates get the empty
    think block a reasoning-free reply closes on policy (`render_with_empty_thought`)."""
    opening = [{"role": "user", "content": "x"}]
    prompt = apply_template(opening, True)
    full = apply_template(opening + [message], False)
    if not full.startswith(prompt) and opens_thinking(prompt):
        full = render_with_empty_thought(lambda m: apply_template(m, False), opening + [message])
    return full[len(prompt):] if full.startswith(prompt) else None


def assistant_reply_segments(apply_template, message: dict, specials=()) -> list[tuple[str, bool]] | None:
    """Render an assistant reply after its serving generation prompt.

    Special-token spellings in assistant-authored content are escaped before
    applying the template, then returned as ``(text, escaped)`` runs so callers
    can tokenize structure and quoted content with their respective policies.
    ``apply_template(messages, add_generation_prompt)`` is the same contract as
    :func:`assistant_reply`.
    """
    if not specials:
        reply = assistant_reply(apply_template, message)
        return None if reply is None else [(reply, False)]

    supplied = json.dumps(message, ensure_ascii=False)
    nonce = secrets.token_hex(16)
    while nonce in supplied:
        nonce = secrets.token_hex(16)

    def escape_value(value):
        if isinstance(value, str):
            return escape_specials(value, specials, nonce)
        if isinstance(value, list):
            return [escape_value(child) for child in value]
        if isinstance(value, dict):
            return {key: escape_value(child) for key, child in value.items()}
        return value

    prepared = dict(message)
    content = prepared.get("content")
    if isinstance(content, str):
        prepared["content"] = escape_specials(content, specials, nonce)
    elif _is_parts(content):
        prepared["content"] = [dict(part, **({"text": escape_specials(part.get("text") or "", specials, nonce)}
                                              if part.get("type") == "text" else {}))
                                for part in content]
    calls = []
    for call in prepared.get("tool_calls") or []:
        copied = json.loads(json.dumps(call))
        fn = copied.get("function", copied)
        arguments = fn.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as error:
                raise RequestError("tool-arguments", f"tool-call arguments are not JSON: {error}") from error
        fn["arguments"] = escape_value(arguments)
        calls.append(copied)
    if calls:
        prepared["tool_calls"] = calls

    reply = assistant_reply(apply_template, prepared)
    return None if reply is None else split_escaped(reply, nonce)


def _at_value_path(value, path):
    for key in path:
        if isinstance(value, dict) and isinstance(key, str) and key in value:
            value = value[key]
        elif isinstance(value, list) and type(key) is int and 0 <= key < len(value):
            value = value[key]
        else:
            raise RequestError("neuralese-template", "the selected write value path is absent or ambiguous")
    return value


def _replace_value_path(value, path, replacement):
    if not path:
        return replacement
    key, *rest = path
    if isinstance(value, dict) and isinstance(key, str) and key in value:
        return {**value, key: _replace_value_path(value[key], rest, replacement)}
    if isinstance(value, list) and type(key) is int and 0 <= key < len(value):
        return [(_replace_value_path(item, rest, replacement) if i == key else item)
                for i, item in enumerate(value)]
    raise RequestError("neuralese-template", "the selected write value path is absent or ambiguous")


def call_reply(apply_template, name: str, arguments: dict, argument: str = "value", quoted: bool = True,
               argument_path: tuple[str | int, ...] | list[str | int] | None = None) -> tuple[str, str]:
    """The model's own rendering of an assistant reply that calls `name` with `arguments` and then `argument`, cut at
    that argument's value: (prefix, suffix). `apply_template(messages, add_generation_prompt)` is the chat template.

    With `quoted` the value is a string, as a written block's placeholder is: the prefix ends with the opening quote
    and the suffix starts with the closing one and runs through the end of the turn. Without, the prefix ends where
    the value starts (for decoding the value) and the suffix is empty. Template readout (the `neuralese_template`
    request field) and the trajectory trainer's write sites both force replies cut this way, so what runs is what
    is trained."""
    if argument_path is None:
        call_arguments = {**arguments, argument: _VALUE}
    else:
        path = tuple(argument_path)
        if not path or path[0] != argument:
            raise RequestError("neuralese-template", "write path must begin at the declared argument")
        call_arguments = _replace_value_path(arguments, path, _VALUE)
    call = {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": name, "arguments": call_arguments}}]}
    reply = assistant_reply(apply_template, call)
    if reply is None or _VALUE not in reply:
        raise RequestError("neuralese-template", "the chat template renders the call's reply differently")
    at = reply.index(_VALUE)
    prefix, suffix = reply[:at], reply[at + len(_VALUE):]
    if quoted:
        return prefix, suffix
    if not prefix or prefix[-1] not in "'\"":
        raise RequestError("neuralese-template", "the chat template does not quote string arguments")
    return prefix[:-1], ""


def write_reply(apply_template, name: str, arguments: dict, argument: str = "value",
                value_type: str = "string", argument_path: tuple[str | int, ...] | list[str | int] | None = None) -> tuple[str, str]:
    """Keep native typed-value syntax in the model cache; opaque wire placeholders are quoted separately."""
    prefix, suffix = call_reply(apply_template, name, arguments, argument, argument_path=argument_path)
    if value_type == "string":
        return prefix, suffix
    if value_type != "unknown":
        raise RequestError("neuralese-template", "write value_type must be string or unknown")
    if not prefix or prefix[-1] not in "\"'" or not suffix.startswith(prefix[-1]):
        raise RequestError("neuralese-template", "the chat template does not quote string arguments")
    return prefix[:-1], suffix[1:]



def write_value_text(apply_template, name: str, arguments: dict, argument: str,
                     value, value_type: str = "string", argument_path: tuple[str | int, ...] | list[str | int] | None = None) -> str:
    """Gold value span in the same native template as generated writing.

    Source serialization is not the output syntax: JSON spacing and quoted string
    escaping can change its token count. Require the exact write boundary instead
    of assuming a textual representation or silently clipping the gold value.
    """
    prefix, suffix = write_reply(apply_template, name, arguments, argument, value_type, argument_path)
    call_arguments = ({**arguments, argument: value} if argument_path is None else
                      _replace_value_path(arguments, tuple(argument_path), value))
    call = {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": name, "arguments": call_arguments}}]}
    reply = assistant_reply(apply_template, call)
    if reply is None:
        raise RequestError("neuralese-template", "gold write reply differs from generation prompt")
    if not reply.startswith(prefix) or not reply.endswith(suffix):
        raise RequestError("neuralese-template", "gold value does not match the native write boundary")
    end = len(reply) - len(suffix) if suffix else len(reply)
    return reply[len(prefix):end]


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
        calls.append((_call_name(node.func), {kw.arg: _literal(kw.value) for kw in node.keywords}))
    return calls


_JSON_NAMES = {"true": True, "false": False, "null": None, "True": True, "False": False, "None": None}


def _literal(node):
    """A Python literal, also accepting JSON's true/false/null: LFM2.5's template renders top-level arguments in
    Python form and nested values as JSON (`flag=True, value={"a": false}`), and models write either."""
    if isinstance(node, ast.Name) and node.id in _JSON_NAMES:
        return _JSON_NAMES[node.id]
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Constant):
        return node.value
    raise ValueError(f"not a literal: {ast.dump(node)[:80]}")


def arguments_text(arguments) -> str:
    """A tool call's `function.arguments` as both servers answer it (spec "Response fields"): compact JSON text
    (no spaces after separators), non-ASCII characters as UTF-8 rather than escapes, keys in the order the model
    wrote them. The llama.cpp fork answers the same text (nlohmann ordered_json `dump()`)."""
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def build_message(text: str, block_ids: list[str], call_prefix: str = "call") -> dict:
    """Model output (with block placeholders) → an assistant message with parts where blocks are."""
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    reasoning = None
    think = _THINK.search(text)
    if think:
        reasoning = think.group(1).strip()
        text = text[:think.start()] + text[think.end():]
    elif "</think>" in text:  # thinking templates open the block in the generation prompt
        head, _, text = text.partition("</think>")
        reasoning = head.strip()
    tool_calls = []
    for pattern, parse in ((_CALLS, parse_pythonic_calls), (_JSON_CALLS, parse_json_call)):
        for match in pattern.finditer(text):
            try:
                calls = parse(match.group(1))
            except (ValueError, SyntaxError):
                continue  # left in the content; the runtime treats leaked markup as a malformed call
            for name, arguments in calls:
                tool_calls.append({
                    "id": f"{call_prefix}_{len(tool_calls)}", "type": "function",
                    "function": {"name": name, "arguments": arguments_text(_restore(arguments, block_ids))},
                })
    content = _CALLS.sub(lambda m: "" if _parses(m.group(1)) else m.group(0), text)
    content = _JSON_CALLS.sub(lambda m: "" if _parses(m.group(1), parse_json_call) else m.group(0), content).strip()
    message = {"role": "assistant", "content": _to_parts(content, block_ids) if content else None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def parse_json_call(body: str) -> list[tuple[str, dict]]:
    """A Qwen-family `<tool_call>` body, `{"name": …, "arguments": {…}}` → [(name, arguments)]. Raises ValueError on
    anything else."""
    call = json.loads(body)
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        raise ValueError("a tool call names its function")
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("tool call arguments are an object")
    return [(call["name"], arguments)]


def _parses(body: str, parse=None) -> bool:
    try:
        (parse or parse_pythonic_calls)(body)
        return True
    except (ValueError, SyntaxError):
        return False
