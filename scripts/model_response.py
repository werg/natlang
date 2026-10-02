"""Decode model transport formats without executing text or changing tool arguments."""
import ast
import json
import math
import re
import uuid
import xml.etree.ElementTree as ET


def literal(node):
    if isinstance(node, ast.Name) and node.id in ('true', 'false', 'null'):
        return {'true': True, 'false': False, 'null': None}[node.id]
    if isinstance(node, ast.List):
        return [literal(value) for value in node.elts]
    if isinstance(node, ast.Dict):
        return {literal(key): literal(value) for key, value in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def _call(name, arguments):
    return {'id': 'call_' + uuid.uuid4().hex, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}


def _ling_xml_calls(text, tools):
    """Parse the unescaped XML-like tool transport emitted by Ling's Jinja template.

    Argument values are delimited rather than XML-decoded because the template writes
    string values verbatim. This preserves whitespace, quotes, ampersands and angle
    brackets exactly. A value containing a closing transport delimiter is ambiguous
    in the model's format and is rejected instead of guessed.
    """
    schemas = {
        tool['function']['name']: tool['function'].get('parameters', {})
        for tool in tools if isinstance(tool, dict) and isinstance(tool.get('function'), dict)
        and isinstance(tool['function'].get('name'), str)
    }
    spans = list(re.finditer(r'<tool_call>(.*?)</tool_call>', text, re.S))
    residue = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.S)
    if '<tool_call>' in residue or '</tool_call>' in residue:
        raise ValueError('incomplete Ling XML tool call')
    if not spans:
        return [], text

    calls = []
    for match in spans:
        body = match.group(1)
        first_key = body.find('<arg_key>')
        if first_key < 0:
            name, remainder = body.strip(), ''
        else:
            name, remainder = body[:first_key].strip(), body[first_key:]
        if (not name or any(ch.isspace() for ch in name) or name not in schemas):
            raise ValueError('tool is not in this request')

        parameters = schemas[name]
        properties = parameters.get('properties', {}) if isinstance(parameters, dict) else {}
        arguments = {}
        cursor = 0
        while cursor < len(remainder):
            while cursor < len(remainder) and remainder[cursor].isspace():
                cursor += 1
            if cursor == len(remainder):
                break
            key_open = '<arg_key>'
            if not remainder.startswith(key_open, cursor):
                raise ValueError('invalid Ling XML argument sequence')
            key_start = cursor + len(key_open)
            key_end = remainder.find('</arg_key>', key_start)
            if key_end < 0:
                raise ValueError('incomplete Ling XML argument key')
            key = remainder[key_start:key_end]
            if not key or key.strip() != key or any(tag in key for tag in ('<', '>')):
                raise ValueError('invalid Ling XML argument key')
            cursor = key_end + len('</arg_key>')
            while cursor < len(remainder) and remainder[cursor].isspace():
                cursor += 1
            value_open = '<arg_value>'
            if not remainder.startswith(value_open, cursor):
                raise ValueError('Ling XML argument key is not followed by a value')
            value_start = cursor + len(value_open)
            value_end = remainder.find('</arg_value>', value_start)
            if value_end < 0:
                raise ValueError('incomplete Ling XML argument value')
            raw_value = remainder[value_start:value_end]
            cursor = value_end + len('</arg_value>')

            if key in arguments or key not in properties:
                raise ValueError('unknown or duplicate Ling XML argument')
            schema = properties[key] if isinstance(properties[key], dict) else {}
            expected = schema.get('type')
            if expected == 'string':
                value = raw_value
            else:
                try:
                    value = json.loads(raw_value, parse_constant=_reject_json_constant,
                                       parse_float=_finite_json_float,
                                       object_pairs_hook=_strict_json_object)
                except json.JSONDecodeError as error:
                    allowed = _allowed_schema_types(schema)
                    if 'string' in allowed or not allowed:
                        value = raw_value
                    else:
                        raise ValueError('non-string Ling XML values must be JSON') from error
                except (TypeError, ValueError) as error:
                    raise ValueError('invalid strict JSON Ling argument value') from error
                else:
                    allowed = _allowed_schema_types(schema)
                    parsed_type = _json_type(value)
                    if not allowed:
                        raise ValueError(f'Ling XML argument {key!r} has no type; JSON-shaped value is ambiguous')
                    non_string_types = allowed - {'string'}
                    parsed_non_string_type_allowed = (
                        parsed_type in non_string_types or
                        (parsed_type == 'integer' and 'number' in non_string_types)
                    )
                    if 'string' in allowed and parsed_non_string_type_allowed:
                        raise ValueError(f'Ling XML argument {key!r} has an ambiguous string/JSON union value')
                    if parsed_type not in allowed and not (parsed_type == 'integer' and 'number' in allowed):
                        raise ValueError(f'Ling XML argument {key!r} must have JSON type in {sorted(allowed)}')
                    if parsed_type == 'string':
                        # Strings are emitted verbatim by the Jinja template, so retain
                        # the original bytes including any quote characters.
                        value = raw_value
            arguments[key] = value

        required = parameters.get('required', []) if isinstance(parameters, dict) else []
        if any(key not in arguments for key in required):
            raise ValueError('Ling XML tool call is missing a required argument')
        calls.append(_call(name, arguments))

    content = text
    for match in reversed(spans):
        content = content[:match.start()] + content[match.end():]
    return calls, content


def _reject_json_constant(value):
    raise ValueError(f'non-standard JSON number {value}')


def _finite_json_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f'non-finite JSON number {value}')
    return parsed


def _strict_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON object key {key!r}')
        result[key] = value
    return result


def _json_type(value):
    if value is None:
        return 'null'
    if type(value) is bool:
        return 'boolean'
    if type(value) is int:
        return 'integer'
    if type(value) is float:
        return 'number'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, list):
        return 'array'
    if isinstance(value, dict):
        return 'object'
    return 'unknown'


def _allowed_schema_types(schema):
    if not isinstance(schema, dict):
        return set()
    schema_type = schema.get('type')
    if isinstance(schema_type, str):
        result = {schema_type}
    elif isinstance(schema_type, list):
        result = {item for item in schema_type if isinstance(item, str)}
    else:
        result = set()
    for key in ('anyOf', 'oneOf'):
        variants = schema.get(key)
        if isinstance(variants, list):
            for variant in variants:
                result.update(_allowed_schema_types(variant))
    return result


def _first_transport_call_start(text):
    markers = ('<tool_call>', '<function', '<|tool_call_start|>')
    candidates = []
    for marker in markers:
        start = 0
        while True:
            position = text.find(marker, start)
            if position < 0:
                break
            candidates.append(position)
            start = position + len(marker)
    candidates.sort()
    if not candidates:
        return None
    think_regions = []
    start = 0
    while True:
        opening = text.find('<think>', start)
        if opening < 0:
            break
        closing = text.find('</think>', opening + len('<think>'))
        end = len(text) if closing < 0 else closing + len('</think>')
        think_regions.append((opening, end))
        start = end
        if closing < 0:
            break
    return next((candidate for candidate in candidates
                 if not any(begin <= candidate < end for begin, end in think_regions)), None)


def parse_response(text, tools=()):
    reasoning = None
    call_start = _first_transport_call_start(text)
    prefix = text if call_start is None else text[:call_start]
    suffix = '' if call_start is None else text[call_start:]
    if '<think>' in prefix:
        match = re.search(r'<think>(.*?)</think>', prefix, re.S)
        if not match:
            return {'content': '', 'reasoning_content': prefix.split('<think>', 1)[1], 'tool_calls': []}
        reasoning = match.group(1).strip()
        prefix = prefix[:match.start()] + prefix[match.end():]
    elif '</think>' in prefix:
        # Some generation prefixes already contain the opening thinking token.
        closing = prefix.index('</think>')
        reasoning, prefix = prefix[:closing].strip(), prefix[closing + len('</think>'):]
    text = prefix + suffix
    calls = []
    match = re.search(r'<\|tool_call_start\|>(.*?)<\|tool_call_end\|>', text, re.S)
    if match:
        nodes = ast.parse(match.group(1), mode='eval').body
        if not isinstance(nodes, ast.List):
            raise ValueError('tool calls must form a list')
        for node in nodes.elts:
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.args or any(key.arg is None for key in node.keywords):
                raise ValueError('invalid tool-call syntax')
            calls.append(_call(node.func.id, {key.arg: literal(key.value) for key in node.keywords}))
        text = text[:match.start()] + text[match.end():]
    elif ('<tool_call>' in text and
          ('<function' not in text or text.index('<tool_call>') < text.index('<function'))):
        ling_calls, text = _ling_xml_calls(text, tools)
        calls.extend(ling_calls)
    elif '<function' in text:
        schemas = {tool['function']['name']: tool['function'].get('parameters', {}).get('properties', {}) for tool in tools}
        fragments = re.findall(r'<function\b[^>]*>(?:<!\[CDATA\[.*?\]\]>|(?:(?!</function>).))*?</function>', text, re.S)
        if not fragments:
            raise ValueError('incomplete XML tool call')
        for fragment in fragments:
            function = ET.fromstring(fragment)
            name = function.attrib.get('name')
            if name not in schemas:
                raise ValueError('tool is not in this request')
            arguments = {}
            for parameter in function:
                if parameter.tag != 'param' or list(parameter):
                    raise ValueError('invalid XML parameter')
                key = parameter.attrib.get('name')
                if key in arguments or key not in schemas[name]:
                    raise ValueError('unknown or duplicate XML parameter')
                value = parameter.text or ''
                schema = schemas[name][key]
                if schema.get('type') == 'string':
                    arguments[key] = value
                elif schema.get('type') == 'boolean' and value.strip() in ('True', 'False'):
                    # MiniCPM emits Python boolean literals in its XML transport.
                    arguments[key] = value.strip() == 'True'
                else:
                    arguments[key] = json.loads(value)
            calls.append(_call(name, arguments))
            text = text.replace(fragment, '', 1)
    elif '</tool_call>' in text:
        raise ValueError('incomplete Ling XML tool call')
    content = re.sub(r'<\|.*?\|>', '', text).strip()
    return {'content': content, 'reasoning_content': reasoning, 'tool_calls': calls}


def tool_calls(text, tools=()):
    return parse_response(text, tools)['tool_calls']
