"""Decode model transport formats without executing text or changing tool arguments."""
import ast
import json
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


def parse_response(text, tools=()):
    reasoning = None
    if '<think>' in text:
        match = re.search(r'<think>(.*?)</think>', text, re.S)
        if not match:
            return {'content': '', 'reasoning_content': text.split('<think>', 1)[1], 'tool_calls': []}
        reasoning = match.group(1).strip()
        text = text[:match.start()] + text[match.end():]
    elif '</think>' in text:
        # Some generation prefixes already contain the opening thinking token.
        closing = text.index('</think>')
        first_call = min((text.index(marker) for marker in ('<function', '<|tool_call_start|>') if marker in text), default=len(text))
        if closing < first_call:
            reasoning, text = text[:closing].strip(), text[closing + len('</think>'):]
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
    content = re.sub(r'<\|.*?\|>', '', text).strip()
    return {'content': content, 'reasoning_content': reasoning, 'tool_calls': calls}


def tool_calls(text, tools=()):
    return parse_response(text, tools)['tool_calls']
