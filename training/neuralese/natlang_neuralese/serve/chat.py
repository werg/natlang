"""Chat requests to model input, and model output to chat messages, with native shared state (S4 §3–4).

Requests are OpenAI-style. Wherever the runtime carries shared state, a message's `content` or a tool call's
`function.arguments` is an array of parts, `{"type": "text", "text": …}` and `{"type": "native_state", "id": …}`.
Tool-call arguments travel as JSON text, so their parts are pieces of that text.

Rendering replaces every block with a private-use placeholder, applies the backbone's native chat template (LFM2.5
renders tool calls in its Pythonic form, `<|tool_call_start|>[eval(code='…')]<|tool_call_end|>`, so a literal sits
inside the quoted code string), then cuts the rendered text at the placeholders: text runs are tokenized and
native-state references become position-preserving imports.

Escaping (spec §3.3): control-token text inside message content or tool-call arguments is ordinary text. Rendering
wraps every occurrence of a special-token string in content between two private-use escape marks; the engine
tokenizes the wrapped text with special-token splitting, so only structure the template itself inserted becomes
special tokens.

Parsing does the reverse on what the model produced: state references are placeholders while the Pythonic tool calls are
parsed, then every string that holds a placeholder becomes a part array again.
"""

from __future__ import annotations

import ast
import base64
import json
import re
import secrets
from dataclasses import dataclass

from ..common.hashing import canonical_json_sha256_hex

_PH_OPEN, _PH_CLOSE = "", ""
_PH = re.compile(_PH_OPEN + r"(\d+)" + _PH_CLOSE)
ESC_OPEN, ESC_CLOSE = "\ue012", "\ue013"
_CALLS = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.S)
_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_JSON_CALLS = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)  # Qwen-family templates (Maple)

TOOL_CALL_FORMATS = {"lfm2": "pythonic", "lfm2_moe": "pythonic", "lfm3": "xml", "qwen3_5": "xml",
                     "mellum": "json", "maple": "json", "qwen3": "json", "qwen2": "json"}
"""Declared tool-call serialization of each backbone's chat template (`config.model_type`), never inferred from the
vocabulary (LFM3 keeps LFM2's unused `<|tool_call_start|>` token):

- `pythonic` (LFM2/LFM2.5): `<|tool_call_start|>[name(key='value', …)]<|tool_call_end|>`; string values are quoted
  Python literals, so a write site's value sits between two quotes.
- `json` (Qwen3/Mellum/Maple): `<tool_call>\\n{"name": …, "arguments": {…}}\\n</tool_call>`; quoted JSON strings.
- `xml` (LFM3, Qwen3.5; the Qwen3-Coder form): `<tool_call>\\n<function=NAME>\\n<parameter=KEY>\\nVALUE\\n</parameter>\\n…
  </function>\\n</tool_call>`. VALUE is unquoted: a string verbatim, anything else its JSON text. A value ends at the
  first `XML_PARAMETER_CLOSE` after its opening line (the rule of llama.cpp's Qwen3-Coder/LFM3 parser), so a string
  is representable exactly when that delimiter does not occur in `VALUE + XML_PARAMETER_CLOSE` before its end
  (`xml_value_representable`). Types come from the tool schema: a string-only parameter is verbatim text, any other
  declared type is JSON first, and an undeclared parameter is JSON when its text parses as JSON, else a string."""
XML_PARAMETER_CLOSE = "\n</parameter>\n"
_XML_FUNCTION_OPEN = re.compile(r"<function=([^>\n]+)>\n")
_XML_PARAMETER_OPEN = re.compile(r"<parameter=([^>\n]+)>\n")


def tool_call_format(model_type) -> str | None:
    """The declared tool-call serialization of a backbone (`config.model_type`), or None when undeclared."""
    return TOOL_CALL_FORMATS.get(model_type) if isinstance(model_type, str) else None


def xml_value_representable(text: str) -> bool:
    """`text` as an XML parameter value parses back to itself: the first `XML_PARAMETER_CLOSE` in the rendered
    `text + XML_PARAMETER_CLOSE` is the one the template wrote."""
    return (text + XML_PARAMETER_CLOSE).find(XML_PARAMETER_CLOSE) == len(text)


def xml_value_text(value) -> str:
    """The text the LFM3 / Qwen3-Coder templates render for a parameter value (`render_value`, jinja `tojson`)."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    return str(value)


def xml_wire_arguments(name, arguments, tools):
    """Encode strings where a parameter's schema also permits JSON values."""
    properties = _tool_parameter_schemas(tools).get(name, {})
    return {key: (json.dumps(value, ensure_ascii=False)
                  if isinstance(value, str) and _schema_types(properties.get(key)) != {"string"}
                  else value) for key, value in arguments.items()}


def check_xml_call(name, arguments, tools=None, *, types: bool = True) -> None:
    """Raise RequestError('tool-call-unrepresentable') when an XML tool call would not parse back exactly: a function
    or parameter name holding `>` or a newline, a value holding the parameter delimiter, or (with `types`) a value
    whose type the rendering loses under `tools`' schema.

    Prompts check structure only (`types=False`): their tokens are the rendering whatever its parse. Training
    targets check types too, because the model learns the text and the parser decides what it means."""
    if not isinstance(name, str) or not name or re.search(r"[>\n]", name):
        raise RequestError("tool-call-unrepresentable", f"function name {name!r} cannot be written as <function=…>")
    if not isinstance(arguments, dict):
        raise RequestError("tool-call-unrepresentable", f"arguments of {name!r} are not an object")
    properties = _tool_parameter_schemas(tools).get(name, {})
    for key, value in arguments.items():
        if not isinstance(key, str) or not key or re.search(r"[>\n]", key):
            raise RequestError("tool-call-unrepresentable", f"parameter name {key!r} cannot be written as <parameter=…>")
        text = xml_value_text(xml_wire_arguments(name, {key: value}, tools)[key])
        if not xml_value_representable(text):
            raise RequestError("tool-call-unrepresentable",
                               f"parameter {key!r} of {name!r} contains the XML parameter delimiter "
                               f"{XML_PARAMETER_CLOSE!r}; the template has no escape for it")
        if not types:
            continue
        try:
            parsed = xml_value(text, properties.get(key))
        except ValueError:
            parsed = _UNPARSED
        if parsed is _UNPARSED or json.dumps(parsed, ensure_ascii=False) != json.dumps(value, ensure_ascii=False):
            raise RequestError("tool-call-unrepresentable",
                               f"parameter {key!r} of {name!r} does not keep its type in the XML rendering")


_UNPARSED = object()


def _schema_types(schema) -> set[str] | None:
    """JSON-schema types a parameter declares, or None when it declares none (unknown)."""
    if not isinstance(schema, dict):
        return None
    kind = schema.get("type")
    if isinstance(kind, str):
        return {kind}
    if isinstance(kind, list) and all(isinstance(k, str) for k in kind):
        return set(kind)
    for key in ("anyOf", "oneOf"):
        if isinstance(schema.get(key), list):
            found = set()
            for option in schema[key]:
                types = _schema_types(option)
                if types is None:
                    return None
                found |= types
            return found or None
    return None


def xml_value(text: str, schema=None):
    """An XML parameter's text → its value, typed by the parameter's schema (see TOOL_CALL_FORMATS)."""
    types = _schema_types(schema)
    if types == {"string"}:
        return text
    try:
        value = json.loads(text)
    except ValueError:
        if types is not None and "string" not in types:
            raise ValueError(f"parameter value is not JSON of its declared type {sorted(types)}")
        return text
    if types is None:
        # Ambiguous strings are explicitly JSON quoted by the shared renderer;
        # canonical non-string values retain the template's existing representation.
        if isinstance(value, str) and json.dumps(value, ensure_ascii=False) == text:
            return value
        return value if not isinstance(value, str) and xml_value_text(value) == text else text
    allowed = {"object": isinstance(value, dict), "array": isinstance(value, list),
               "number": isinstance(value, (int, float)) and not isinstance(value, bool),
               "integer": isinstance(value, int) and not isinstance(value, bool),
               "boolean": isinstance(value, bool), "null": value is None,
               "string": isinstance(value, str) and json.dumps(value, ensure_ascii=False) == text}
    if any(allowed.get(kind) for kind in types):
        return value
    if "string" in types:
        return text
    raise ValueError(f"parameter value is not of its declared type {sorted(types)}")


def _tool_parameter_schemas(tools) -> dict:
    """{function name: {parameter: schema}} of OpenAI-style tool declarations."""
    out = {}
    for tool in tools or ():
        fn = tool.get("function", tool) if isinstance(tool, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            properties = (fn.get("parameters") or {}).get("properties")
            out[fn["name"]] = properties if isinstance(properties, dict) else {}
    return out


@dataclass
class XmlParameter:
    key: str
    value_start: int
    value_end: int | None  # None: the value is still open (partial reply)


@dataclass
class XmlCall:
    start: int  # offset of `<tool_call>`
    name: str | None  # None: the function line is not complete yet
    name_start: int
    parameters: list
    end: int | None  # offset after `</tool_call>`; None: incomplete


def scan_xml_calls(text: str) -> list[XmlCall]:
    """Every `<tool_call>` of `text` scanned structurally (no value escapes, a value ends at its first
    `XML_PARAMETER_CLOSE`). A call that is malformed or still being written has `end` None and stops the scan of that
    call; complete calls are what `parse_xml_calls` returns. Guidance reads the partial structure while generating."""
    calls, at = [], 0
    while True:
        start = text.find("<tool_call>", at)
        if start < 0:
            return calls
        i = start + len("<tool_call>")
        while i < len(text) and text[i] in " \t\r\n":
            i += 1
        function = _XML_FUNCTION_OPEN.match(text, i)
        call = XmlCall(start, function.group(1) if function else None, i + len("<function="), [], None)
        calls.append(call)
        if function is None:
            at = i
            continue
        i = function.end()
        while True:
            parameter = _XML_PARAMETER_OPEN.match(text, i)
            if parameter is None:
                break
            close = text.find(XML_PARAMETER_CLOSE, parameter.end())
            call.parameters.append(XmlParameter(parameter.group(1), parameter.end(), None if close < 0 else close))
            if close < 0:
                return calls
            i = close + len(XML_PARAMETER_CLOSE)
        tail = re.compile(r"</function>\s*</tool_call>").match(text, i)
        if tail is None:
            at = i
            continue
        call.end = tail.end()
        at = tail.end()


def parse_xml_calls(text: str, tools=None) -> list[tuple[str, dict]]:
    """XML tool calls (`<tool_call>` … `</tool_call>`, one or more) → [(name, arguments)], typed by `tools`' schemas.
    Raises ValueError unless `text` is exactly complete calls separated by whitespace."""
    schemas = _tool_parameter_schemas(tools)
    calls, last = [], 0
    for call in scan_xml_calls(text):
        if call.end is None or call.name is None or text[last:call.start].strip():
            raise ValueError("not a complete XML tool call")
        properties = schemas.get(call.name, {})
        arguments = {}
        for parameter in call.parameters:
            if parameter.key in arguments:
                raise ValueError(f"parameter {parameter.key!r} repeats")
            raw = text[parameter.value_start:parameter.value_end]
            try:
                arguments[parameter.key] = xml_value(raw, properties.get(parameter.key))
            except ValueError:
                if not _PH.search(raw):
                    raise
                arguments[parameter.key] = raw  # a written block stands for the value (native write sites)
        calls.append((call.name, arguments))
        last = call.end
    if not calls or text[last:].strip():
        raise ValueError("not a complete XML tool call")
    return calls


class RequestError(ValueError):
    """A malformed request (HTTP 400). `code` is a stable diagnostic name."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


def placeholder(index: int) -> str:
    return f"{_PH_OPEN}{index}{_PH_CLOSE}"


def _is_parts(value) -> bool:
    return isinstance(value, list) and value and all(
        isinstance(p, dict) and p.get("type") in ("text", "native_state") for p in value)


def _reject_projected_parts(value) -> None:
    if isinstance(value, dict):
        if value.get("type") == "neuralese":
            raise RequestError("neuralese-retired-part", "projected neuralese parts are retired; use native_state references")
        for nested in value.values():
            _reject_projected_parts(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_projected_parts(nested)


@dataclass
class Rendered:
    segments: list  # str (template text) or int (index into `blocks`)
    blocks: list[str]  # native state IDs in order of appearance
    escape_nonce: str = ""
    block_stop_gradients: list[bool] | None = None
    block_source_occurrences: list[str | None] | None = None


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


def normalize_tool_arguments(arguments):
    """Decode JSON-text tool arguments into the object expected by chat templates."""
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError as error:
        raise RequestError("tool-arguments", f"tool-call arguments are not JSON: {error}") from error


def normalize_tool_call_arguments(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return a copy of messages with JSON-text tool arguments decoded for chat templates.

    OpenAI-style transports carry `function.arguments` as JSON text, while native chat
    templates expect an object. Keep this conversion in one place for ordinary serving,
    training renders, and direct tokenizer evaluation. The receipt binds each conversion
    to the exact raw text and decoded value without changing the caller's messages.
    """
    normalized = json.loads(json.dumps(messages, ensure_ascii=False))
    conversions = []
    for message_index, message in enumerate(normalized):
        for call_index, call in enumerate(message.get("tool_calls") or []):
            fn = call.get("function", call)
            arguments = fn.get("arguments")
            if not isinstance(arguments, str):
                continue
            decoded = normalize_tool_arguments(arguments)
            fn["arguments"] = decoded
            path_prefix = "/function/arguments" if "function" in call else "/arguments"
            conversions.append({
                "path": f"messages/{message_index}/tool_calls/{call_index}{path_prefix}",
                "raw_text_sha256": canonical_json_sha256_hex(arguments),
                "decoded_value_sha256": canonical_json_sha256_hex(decoded),
            })
    return normalized, conversions


def render_messages(messages: list[dict], tools: list | None, apply_template, specials=(),
                    call_format: str | None = None) -> Rendered:
    """Messages with native-state parts → template text cut into text runs and state references.

    `apply_template(messages, tools)` renders plain messages with the model's chat template and the generation prompt.
    `specials` are the tokenizer's special-token strings; occurrences inside content are escaped (spec §3.3).
    `call_format` is the backbone's declared tool-call format (`tool_call_format`); under `xml` every history call
    must be structurally representable (`check_xml_call`), since the template writes values unescaped.
    """
    blocks: list[str] = []
    block_stop_gradients: list[bool] = []
    block_source_occurrences = []
    native_values = set()
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
                if not isinstance(block_id, str) or not re.fullmatch(r"nzs1_[a-z2-7]{52}", block_id):
                    raise RequestError("neuralese-bad-part", f"invalid native_state part {part!r}")
                if 'stop_gradient' in part and part['stop_gradient'] is not True:
                    raise RequestError('neuralese-bad-part','stop_gradient must be true on an actual native-state reference use')
                out.append(f"{input_prefix}{len(blocks)}{_PH_CLOSE}")
                blocks.append(block_id)
                block_stop_gradients.append(part.get('stop_gradient',False))
                source=part.get('source_occurrence')
                if 'source_occurrence' in part and (not isinstance(source,str) or not source):
                    raise RequestError('neuralese-bad-part','native source occurrence must be its nonempty actual producer reference')
                block_source_occurrences.append(source)
        return "".join(out)

    def is_native_marker(marker: str) -> bool:
        match = input_pattern.fullmatch(marker)
        return bool(match and int(match.group(1)) < len(blocks))

    plain = []
    normalized_messages, _ = normalize_tool_call_arguments(messages)
    for message in normalized_messages:
        message = dict(message)
        _reject_projected_parts(message.get("content"))
        if _is_parts(message.get("content")):
            message["content"] = flatten(message["content"])
        elif isinstance(message.get("content"), str):
            message["content"] = escape_specials(message["content"], specials, nonce)
        calls = []
        for call in message.get("tool_calls") or []:
            call = json.loads(json.dumps(call))
            fn = call.get("function", call)
            arguments = fn.get("arguments")
            _reject_projected_parts(arguments)
            if _is_parts(arguments):
                # The initial message normalization above already decodes
                # OpenAI JSON-text arguments. Only a content-part list needs
                # a second decode after its text runs have been joined. In
                # particular, a valid JSON string scalar such as `"hello"`
                # must remain the decoded string `hello`, not be parsed again
                # as fresh JSON text.
                arguments = normalize_tool_arguments(flatten(arguments, escape=False))
            def inline(value):
                # An argument value that is a part list holding state renders as an imported value in the string.
                if _is_parts(value) and any(part["type"] == "native_state" for part in value):
                    # Only a complete content-addressed native-state ID denotes
                    # an imported state. A short lookalike inside JSON tool
                    # arguments remains ordinary data for runtime validation.
                    if any(part["type"] == "native_state" and
                           (not isinstance(part.get("id"), str) or
                            not re.fullmatch(r"nzs1_[a-z2-7]{52}", part["id"]))
                           for part in value):
                        # Keep the whole heterogeneous part-list as ordinary
                        # argument data if any state reference is malformed;
                        # partially importing it would change the value shape.
                        return value
                    value = flatten(value, escape=False)
                    marker = input_pattern.fullmatch(value)
                    if marker and is_native_marker(value):
                        native_values.add(value)
                    return value
                if isinstance(value, str):
                    marker = input_pattern.fullmatch(value)
                    if marker and is_native_marker(value):
                        native_values.add(value)
                    return value
                if isinstance(value, list):
                    return [inline(item) for item in value]
                if isinstance(value, dict):
                    return {key: inline(item) for key, item in value.items()}
                return value

            inlined = inline(arguments)
            if call_format == "xml":
                check_xml_call(fn.get("name"), inlined, tools, types=False)
                inlined = xml_wire_arguments(fn.get("name"), inlined, tools)
            fn["arguments"] = _escape_value(inlined, specials, nonce)
            calls.append(call)
        if calls:
            message["tool_calls"] = calls
        plain.append(message)
    text = apply_template(plain, tools)
    for marker in native_values:
        text = re.sub(r"([\"'])" + re.escape(marker) + r"\1", lambda match: marker, text)
        # Templates commonly JSON-encode tool arguments. That escapes private
        # placeholder codepoints (`\ue010...\ue011`), which would otherwise
        # make a real block occurrence disappear before token-position restore.
        escaped = marker
        for _ in range(3):
            escaped = json.dumps(escaped, ensure_ascii=True)[1:-1]
            text = text.replace(escaped, marker)
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
        raise RequestError("neuralese-render", f"the chat template dropped or duplicated a block (expected={len(blocks)}, observed={seen})")
    return Rendered(segments, blocks, nonce, block_stop_gradients, block_source_occurrences)


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
BACKBONE_HISTORY_REASONING = {"lfm2": "last_turn_only", "mellum": "keep", "lfm3": "keep"}
"""Declared, backbone-inherent history-reasoning policy per backbone (`config.model_type`; owner 2026-10-10). The
reasoning of the turn being trained or generated is rendered for every line; what differs is earlier assistant
turns' reasoning in the history. Legacy runs use each backbone's default: LFM2.5 drops earlier reasoning
(`last_turn_only`) and Mellum2.1 keeps it (`keep`). A checkpoint or recipe may explicitly opt in to KEEP on a
switch-capable template. Training text and serving prompts render through `bind_history_reasoning`, which pins the
template switch to the checkpoint's declared policy and asserts with a rendering probe that the template behaves as
declared.

LFM3 (2026-10-28) declares `keep`, pinning its template's `preserve_thinking=true`. Its default keeps the reasoning
of assistant turns after the last user message and rewrites every earlier assistant turn to `<think></think>` plus
its content, so a new user message changes the tokens of the history before it; with `preserve_thinking` every
assistant turn renders `<think>R</think>` (`<think></think>` when it has none) exactly as it was generated after the
`<think>` generation prompt, and every rendering is a token prefix of the next (tests/neuralese/test_lfm3_format.py).
Natlang invocations (one user message, then a tool loop) render identically under both settings."""
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
                           require_declared: bool = False, allow_transition: bool = False):
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
            if not allow_transition:
                raise ValueError(f"tokenizer is bound to history_reasoning={bound.get('policy')!r}, not {declared!r}")
            original = getattr(tokenizer, "_natlang_history_reasoning_original_apply_chat_template", None)
            if original is None:
                raise ValueError("tokenizer history_reasoning transition lacks its original renderer")
            tokenizer.apply_chat_template = original
            delattr(tokenizer, "natlang_history_reasoning")
            bound = None
        if bound is None:
            pass
        elif require_declared and bound.get("policy") == "undeclared":
            raise ValueError(f"backbone {model_type!r} declares no history_reasoning policy (chat.BACKBONE_HISTORY_REASONING)")
        else:
            return tokenizer
    if declared is None and require_declared:
        raise ValueError(f"backbone {model_type!r} declares no history_reasoning policy (chat.BACKBONE_HISTORY_REASONING)")
    kwargs = {} if declared is None else history_reasoning_kwargs(getattr(tokenizer, "chat_template", None), declared)
    original = tokenizer.apply_chat_template
    if not hasattr(tokenizer, "_natlang_history_reasoning_original_apply_chat_template"):
        tokenizer._natlang_history_reasoning_original_apply_chat_template = original

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


def assistant_reply(apply_template, message: dict, opening=None) -> str | None:
    """The text a model generates for assistant `message` after the generation prompt of a one-turn conversation, or
    None when the template's rendering of the turn does not continue that prompt. Thinking templates get the empty
    think block a reasoning-free reply closes on policy (`render_with_empty_thought`)."""
    opening = [{"role": "user", "content": "x"}] if opening is None else opening
    prompt = apply_template(opening, True)
    full = apply_template(opening + [message], False)
    if not full.startswith(prompt) and opens_thinking(prompt):
        full = render_with_empty_thought(lambda m: apply_template(m, False), opening + [message])
    return full[len(prompt):] if full.startswith(prompt) else None


def assistant_reply_segments(apply_template, message: dict, specials=(), opening=None, *,
                             call_format: str | None = None, tools=None) -> list[tuple[str, bool]] | None:
    """Render an assistant reply after its serving generation prompt.

    Special-token spellings in assistant-authored content are escaped before
    applying the template, then returned as ``(text, escaped)`` runs so callers
    can tokenize structure and quoted content with their respective policies.
    ``apply_template(messages, add_generation_prompt)`` is the same contract as
    :func:`assistant_reply`. Under the declared ``xml`` call format every call
    of the reply must parse back exactly, types included (``check_xml_call``
    with ``tools``' schemas): the reply is a training target.
    """
    if call_format == "xml":
        for call in normalize_tool_call_arguments([message])[0][0].get("tool_calls") or []:
            fn = call.get("function", call)
            check_xml_call(fn.get("name"), fn.get("arguments"), tools)
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
    prepared_messages, _ = normalize_tool_call_arguments([prepared])
    prepared = prepared_messages[0]
    for call in prepared.get("tool_calls") or []:
        copied = json.loads(json.dumps(call))
        fn = copied.get("function", copied)
        arguments = fn.get("arguments")
        if call_format == "xml":
            arguments = xml_wire_arguments(fn.get("name"), arguments, tools)
        fn["arguments"] = escape_value(arguments)
        calls.append(copied)
    if calls:
        prepared["tool_calls"] = calls

    reply = assistant_reply(apply_template, prepared, opening)
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


def _xml_whole_parameter(prefix: str, suffix: str) -> bool:
    """The cut sits on a whole unquoted XML parameter value: `<parameter=KEY>\n` | value | `\n</parameter>\n`."""
    return bool(re.search(r"<parameter=[^>\n]+>\n$", prefix)) and suffix.startswith(XML_PARAMETER_CLOSE)


def call_reply(apply_template, name: str, arguments: dict, argument: str = "value", quoted: bool = True,
               argument_path: tuple[str | int, ...] | list[str | int] | None = None,
               call_format: str | None = None) -> tuple[str, str]:
    """The model's own rendering of an assistant reply that calls `name` with `arguments` and then `argument`, cut at
    that argument's value: (prefix, suffix). `apply_template(messages, add_generation_prompt)` is the chat template.

    With `quoted` the value is a string, as a written block's placeholder is: the prefix ends with the opening quote
    and the suffix starts with the closing one and runs through the end of the turn. Without, the prefix ends where
    the value starts (for decoding the value) and the suffix is empty. Template readout (the `neuralese_template`
    request field) and the trajectory trainer's write sites both force replies cut this way, so what runs is what
    is trained.

    XML templates (`call_format` "xml": LFM3, Qwen3.5) write a whole parameter value unquoted, so its boundary is the
    parameter's own lines: the prefix ends with `<parameter=KEY>\n` and the suffix starts with `XML_PARAMETER_CLOSE`,
    for a string and for any other value alike. A value nested inside a JSON-valued parameter (`argument_path` below
    the parameter) is a quoted JSON string as in the other formats. A declared format must match what the template
    renders; without one the boundary the template renders decides."""
    if argument_path is None:
        call_arguments = {**arguments, argument: _VALUE}
    else:
        path = tuple(argument_path)
        if not path or path[0] != argument:
            raise RequestError("neuralese-template", "write path must begin at the declared argument")
        call_arguments = _replace_value_path(arguments, path, _VALUE)
    call = {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": name, "arguments": call_arguments}}]}
    if call_format == "xml":
        check_xml_call(name, call_arguments, types=False)
    reply = assistant_reply(apply_template, call)
    if reply is None or _VALUE not in reply:
        raise RequestError("neuralese-template", "the chat template renders the call's reply differently")
    at = reply.index(_VALUE)
    prefix, suffix = reply[:at], reply[at + len(_VALUE):]
    whole_xml = _xml_whole_parameter(prefix, suffix)
    if call_format == "xml" and not whole_xml and (argument_path is None or len(tuple(argument_path)) == 1):
        raise RequestError("neuralese-template", "the chat template does not write the declared XML parameter lines")
    if call_format in ("pythonic", "json") and whole_xml:
        raise RequestError("neuralese-template", f"the chat template writes XML parameters, not {call_format} calls")
    if quoted:
        return prefix, suffix
    if whole_xml:
        return prefix, ""
    if not prefix or prefix[-1] not in "'\"":
        raise RequestError("neuralese-template", "the chat template does not quote string arguments")
    return prefix[:-1], ""


def write_reply(apply_template, name: str, arguments: dict, argument: str = "value",
                value_type: str = "string", argument_path: tuple[str | int, ...] | list[str | int] | None = None,
                call_format: str | None = None) -> tuple[str, str]:
    """Keep native typed-value syntax in the model cache; opaque wire placeholders are quoted separately. An XML
    parameter value has no quotes, so its string and `unknown` (any JSON value) write sites share one boundary."""
    prefix, suffix = call_reply(apply_template, name, arguments, argument, argument_path=argument_path,
                                call_format=call_format)
    if value_type == "string":
        return prefix, suffix
    if value_type != "unknown":
        raise RequestError("neuralese-template", "write value_type must be string or unknown")
    if _xml_whole_parameter(prefix, suffix):
        return prefix, suffix
    if not prefix or prefix[-1] not in "\"'" or not suffix.startswith(prefix[-1]):
        raise RequestError("neuralese-template", "the chat template does not quote string arguments")
    return prefix[:-1], suffix[1:]



def write_value_text(apply_template, name: str, arguments: dict, argument: str,
                     value, value_type: str = "string", argument_path: tuple[str | int, ...] | list[str | int] | None = None,
                     call_format: str | None = None) -> str:
    """Gold value span in the same native template as generated writing.

    Source serialization is not the output syntax: JSON spacing and quoted string
    escaping can change its token count. Require the exact write boundary instead
    of assuming a textual representation or silently clipping the gold value.
    An unquoted XML parameter value must also parse back to the gold value
    (no parameter delimiter inside it; a string value stays a string).
    """
    prefix, suffix = write_reply(apply_template, name, arguments, argument, value_type, argument_path, call_format)
    if _xml_whole_parameter(prefix, suffix):
        text = xml_value_text(value)
        if not xml_value_representable(text):
            raise RequestError("neuralese-template", "gold value contains the XML parameter delimiter")
        if value_type == "string" and not isinstance(value, str):
            raise RequestError("neuralese-template", "a string write site's gold value is not a string")
        if value_type == "unknown" and json.dumps(xml_value(text), ensure_ascii=False) != json.dumps(value, ensure_ascii=False):
            raise RequestError("neuralese-template", "gold value does not keep its type in the XML rendering")
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
        parts.append({"type": "native_state", "id": block_ids[int(match.group(1))]})
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


def _take_xml_calls(text: str, tools=None) -> tuple[list[tuple[str, dict]], str]:
    """Complete, well-typed XML calls of `text` (in order) and the text without them. A malformed or ill-typed call
    stays in the text, where the runtime treats leaked markup as a malformed call."""
    calls, kept, last = [], [], 0
    for call in scan_xml_calls(text):
        if call.end is None or call.name is None:
            continue
        try:
            parsed = parse_xml_calls(text[call.start:call.end], tools)
        except ValueError:
            continue
        calls.extend(parsed)
        kept.append(text[last:call.start])
        last = call.end
    kept.append(text[last:])
    return calls, "".join(kept)


def build_message(text: str, block_ids: list[str], call_prefix: str = "call", tools=None) -> dict:
    """Model output (with block placeholders) → an assistant message with parts where blocks are. `tools` (the
    request's declarations) type the values of XML calls (`xml_value`)."""
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
    xml_calls, text = _take_xml_calls(text, tools)
    for name, arguments in xml_calls:
        tool_calls.append({
            "id": f"{call_prefix}_{len(tool_calls)}", "type": "function",
            "function": {"name": name, "arguments": arguments_text(_restore(arguments, block_ids))},
        })
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
