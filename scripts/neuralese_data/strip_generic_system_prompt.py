#!/usr/bin/env python3
"""Program-header twin of a rendered native-gold text file: the generic system prompt removed.

Owner decision 2026-10-11 (plans/neuralese/NO_SYSTEM_PROMPT_RENDERING_DECOMPOSITION.md): the natlang-only
models train first on documents with the full runtime system prompt (warm-up), then on documents whose GENERIC
runtime instructions and tool schemas are dropped (context distillation of the prompt into the weights).
PROGRAM-SPECIFIC content always stays: the call signature, types, instructions, inputs and skills already live in
the first user turn; the few system paragraphs that declare a property of this call (directory-reducer mode,
nesting layer, Neuralese parameter/result declarations, the edit_code capability) are kept verbatim as a short
system turn, the "program header". No new model-facing text is introduced: kept paragraphs are byte-identical
to the source; a document with none of them has no system turn at all.

Per row: the system turn's content is split into its prose paragraphs and its tool section (``List of tools:``
on LFM2.5, ``# Tools`` on the Qwen-style Mellum template); prose paragraphs whose head matches
PROGRAM_SPECIFIC_HEADS are kept, everything else is dropped. Token IDs from the first non-system turn on are the
source row's own IDs; the header is spliced in at token level. Where the source text re-tokenizes exactly to its
IDs, the script proves the splice equals re-tokenizing the new text (rows rendered with escaped special spellings
cannot be re-tokenized from text, so the splice is their only correct form). It then shifts ``supervised_suffix_start`` (and any ``context_only_spans``) by the removed length. A row
failing either proof goes to ``omissions.jsonl`` with its reason; nothing is dropped silently. Document identity
(id, split, source_groups, source_record_ids, attestations) is copied unchanged, so splits and source-group
disjointness are those of the source file.

Usage (CPU only; inside the natlang-neuralese image, under the DGX memory ledger):
  strip_generic_system_prompt.py --source-text T --tokenizer DIR --out-dir OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = 'natlang.text-program-header-twin/1'
SYSTEM_OPEN = '<|im_start|>system\n'
TURN_END = '<|im_end|>'
TOOL_SECTION_MARKERS = ('List of tools: ', '# Tools\n')
# Paragraph heads (exact prefixes) that declare a property of THIS call rather than generic runtime behaviour.
PROGRAM_SPECIFIC_HEADS = (
    'This call is a directory reducer:',
    'This call is at the ',  # nesting layer of ad hoc natural-language calls
    '\nDeclared Neuralese text-channel emulation for this run:',
    'Declared Neuralese text-channel emulation for this run:',
    'At a direct parameter declared Neuralese<string>',
    "When this call's declared result is exactly Neuralese<string>",
    'If a function you call is unclear or wrong, read it with read_code and fix it with edit_code',
)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def split_system(text: str):
    """(prefix before the system turn, system content, rest after the system turn incl. its end) or None."""
    start = text.find(SYSTEM_OPEN)
    if start < 0 or text[:start] not in ('', '<|startoftext|>'):
        return None
    end = text.find(TURN_END, start)
    if end < 0:
        return None
    after = end + len(TURN_END)
    if text[after:after + 1] == '\n':
        after += 1
    return text[:start], text[start + len(SYSTEM_OPEN):end], text[after:]


def program_header(content: str):
    """(kept paragraphs, prose chars, tool-section chars)."""
    cut = min((content.find(m) for m in TOOL_SECTION_MARKERS if m in content), default=len(content))
    prose, tools = content[:cut], content[cut:]
    kept = [p for p in prose.split('\n\n') if p.startswith(PROGRAM_SPECIFIC_HEADS)]
    return kept, len(prose), len(tools)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-text', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    a = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = (a.out_dir / 'text-data.jsonl').open('w')
    stats = (a.out_dir / 'row-stats.jsonl').open('w')
    omissions = (a.out_dir / 'omissions.jsonl').open('w')
    im_start = tok.convert_tokens_to_ids('<|im_start|>')
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    totals = {'retokenization_checked': 0, 'rows': 0, 'written': 0, 'omitted': 0, 'with_header': 0, 'source_tokens': 0, 'output_tokens': 0,
              'source_system_tokens': 0, 'header_tokens': 0, 'by_split': {}}
    tokenizer_sha = None
    with a.source_text.open() as source:
        for index, line in enumerate(source):
            if a.limit is not None and index >= a.limit:
                break
            row = json.loads(line)
            totals['rows'] += 1
            ids = list(row['token_ids'])

            def omit(reason):
                totals['omitted'] += 1
                omissions.write(json.dumps({'id': row['id'], 'split': row['split'], 'reason': reason}) + '\n')

            parts = split_system(row['text'])
            if parts is None:
                omit('no leading system turn')
                continue
            prefix, content, rest = parts
            kept, prose_chars, tool_chars = program_header(content)
            header = (SYSTEM_OPEN + '\n\n'.join(kept) + TURN_END + '\n') if kept else ''
            text = prefix + header + rest
            # Token level: the turn after the system turn starts at the second <|im_start|>; everything from there
            # is kept as the source's own IDs (escaped special spellings included). The header's content is
            # encoded with special spellings split, as the renderer does for content; structure IDs are explicit.
            starts = [i for i, t in enumerate(ids) if t == im_start]
            if len(starts) < 2 or not rest.startswith('<|im_start|>'):
                omit('no turn after the system turn')
                continue
            rest_start = starts[1]
            head_ids = ids[:starts[0]]
            header_ids = ([im_start] + tok.encode('system\n' + '\n\n'.join(kept), add_special_tokens=False,
                                                  split_special_tokens=True)
                          + [im_end] + tok.encode('\n', add_special_tokens=False)) if kept else []
            new_ids = head_ids + header_ids + ids[rest_start:]
            exact = tok.encode(row['text'], add_special_tokens=False) == ids
            if exact:
                totals['retokenization_checked'] += 1
                if tok.encode(text, add_special_tokens=False) != new_ids:
                    omit('token-level header splice differs from re-tokenizing the new text')
                    continue
            old_system = rest_start
            new_system = len(head_ids) + len(header_ids)
            shift = old_system - new_system
            sss = row.get('supervised_suffix_start')
            if sss is not None and sss < old_system:
                omit('supervised suffix starts inside the system turn')
                continue
            new_row = dict(row)
            new_row['text'] = text
            new_row['token_ids'] = new_ids
            if sss is not None:
                new_row['supervised_suffix_start'] = sss - shift
            if row.get('context_only_spans'):
                new_row['context_only_spans'] = [[b - shift, e - shift] for b, e in row['context_only_spans']]
            new_row['system_prompt_mode'] = 'program-header' if kept else 'none'
            tokenizer_sha = row.get('tokenizer_sha256')
            out.write(json.dumps(new_row, ensure_ascii=False) + '\n')
            totals['written'] += 1
            totals['with_header'] += bool(kept)
            totals['source_tokens'] += len(ids)
            totals['output_tokens'] += len(new_ids)
            totals['source_system_tokens'] += old_system
            totals['header_tokens'] += new_system
            split = totals['by_split'].setdefault(row['split'], {'rows': 0, 'source_tokens': 0, 'output_tokens': 0})
            split['rows'] += 1
            split['source_tokens'] += len(ids)
            split['output_tokens'] += len(new_ids)
            stats.write(json.dumps({'id': row['id'], 'split': row['split'], 'source_tokens': len(ids),
                                    'source_system_tokens': old_system, 'header_tokens': new_system,
                                    'header_paragraphs': len(kept), 'system_prose_chars': prose_chars,
                                    'system_tool_chars': tool_chars, 'tokens': len(new_ids),
                                    'boundary': new_row.get('supervised_suffix_start')}) + '\n')
    for handle in (out, stats, omissions):
        handle.close()
    receipt = {'schema': SCHEMA, 'source_text': {'path': str(a.source_text), 'sha256': file_sha(a.source_text)},
               'tokenizer': str(a.tokenizer), 'tokenizer_sha256': tokenizer_sha,
               'program_specific_heads': list(PROGRAM_SPECIFIC_HEADS), 'totals': totals,
               'outputs': {name: file_sha(a.out_dir / name)
                           for name in ('text-data.jsonl', 'row-stats.jsonl', 'omissions.jsonl')}}
    (a.out_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(totals))
    return 0 if totals['written'] else 1


if __name__ == '__main__':
    sys.exit(main())
