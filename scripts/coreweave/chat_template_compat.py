"""Preserve ordered natlang system messages in the pinned Qwen renderer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

GUARD = """        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}"""
ORDERED_SYSTEM = """        {%- if not loop.first %}
            {{- '<|im_start|>system\\n' + (render_content(message.content, false, true)|trim) + '<|im_end|>\\n' }}
        {%- endif %}"""


def preserve_ordered_system_messages(template: str) -> str:
    if template.count(GUARD) != 1:
        raise ValueError('chat template does not match the pinned later-system guard')
    return template.replace(GUARD, ORDERED_SYSTEM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != args.source_sha256:
        raise ValueError('chat template source SHA-256 differs')
    derived = preserve_ordered_system_messages(source.decode()).encode()
    if args.out.resolve() == args.source.resolve():
        raise ValueError('derived chat template must preserve its pinned source')
    if args.out.exists() and args.out.read_bytes() != derived:
        raise ValueError('derived chat template output already has different bytes')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(derived)
    print(json.dumps({'schema':'natlang.teacher-chat-template-compatibility/1',
        'source':{'path':str(args.source.resolve()),'sha256':args.source_sha256},
        'derived':{'path':str(args.out.resolve()),'sha256':hashlib.sha256(derived).hexdigest()},
        'scope':'ordered system roles; originally accepted renderings unchanged; live qualification pending'}))


if __name__ == '__main__':
    main()
