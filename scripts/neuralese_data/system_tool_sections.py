"""Program-header derivation for templates whose tool section closes the system turn (LFM3).

``strip_generic_system_prompt.strip_row`` is pinned byte-for-byte by the legacy Search renderer binding
(render_native_turn_corpus.LEGACY_SEARCH_CODE_BINDING), so it is not edited. LFM3's template writes the system turn as
``prose + "\\n\\n" + "<tools>\\n" + one JSON line per tool + "\\n</tools>"`` (or the section alone): this module removes
that closing section from the system turn, text and token IDs together (the turn is re-encoded the way the renderer
encodes structure and content), and then applies the unchanged ``strip_row`` with all of its proofs. Rows without such
a section go to ``strip_row`` unchanged.
"""
from __future__ import annotations

import json

from strip_generic_system_prompt import SYSTEM_OPEN, TURN_END, split_system, strip_row as _strip_row


def closing_tool_section(content: str):
    """(prose, section) when `content` ends with an LFM3 tool section of JSON tool lines, else None."""
    if not content.endswith('\n</tools>'):
        return None
    at = content.find('<tools>\n')
    while at >= 0:
        if at == 0 or content[at - 2:at] == '\n\n':
            body = content[at + len('<tools>\n'):-len('\n</tools>')]
            try:
                if body and all(isinstance(json.loads(line), dict) for line in body.split('\n')):
                    return content[:max(at - 2, 0)], content[at:]
            except ValueError:
                pass
        at = content.find('<tools>\n', at + 1)
    return None


def strip_row(row: dict, tok, im_start: int, im_end: int):
    """``strip_generic_system_prompt.strip_row`` after removing a closing LFM3 tool section (same contract)."""
    parts = split_system(row['text'])
    found = closing_tool_section(parts[1]) if parts else None
    if found is None:
        return _strip_row(row, tok, im_start, im_end)
    prefix, content, rest = parts
    prose, section = found
    ids = list(row['token_ids'])
    starts = [i for i, t in enumerate(ids) if t == im_start]
    if len(starts) < 2:
        return None, 'no turn after the system turn', None
    turn = SYSTEM_OPEN + content + TURN_END + '\n'
    if tok.decode(ids[starts[0]:starts[1]], skip_special_tokens=False, clean_up_tokenization_spaces=False) != turn:
        return None, 'system turn tokens do not decode to the system turn text', None
    # The system turn's IDs are replaced as a whole (strip_row discards them for the header it encodes itself). The
    # trimmed turn is encoded as plain text, so a source row that re-tokenizes exactly still does after the trim and
    # strip_row's re-tokenization proof keeps its meaning; content with escaped special spellings never re-tokenizes
    # exactly, and strip_row relies on the kept IDs after the system turn, as for every template.
    trimmed_turn = SYSTEM_OPEN + prose + TURN_END + '\n'
    trimmed_ids = tok.encode(trimmed_turn, add_special_tokens=False)
    if trimmed_ids[:1] != [im_start] or im_start in trimmed_ids[1:]:
        return None, 'trimmed system turn does not encode as one turn', None
    shift = (starts[1] - starts[0]) - len(trimmed_ids)
    trimmed = dict(row)
    trimmed['text'] = prefix + SYSTEM_OPEN + prose + TURN_END + '\n' + rest
    trimmed['token_ids'] = ids[:starts[0]] + trimmed_ids + ids[starts[1]:]
    if row.get('supervised_suffix_start') is not None:
        trimmed['supervised_suffix_start'] = row['supervised_suffix_start'] - shift
    if row.get('context_only_spans'):
        trimmed['context_only_spans'] = [[b - shift, e - shift] for b, e in row['context_only_spans']]
    new_row, reason, info = _strip_row(trimmed, tok, im_start, im_end)
    if new_row is None:
        return None, reason, None
    # Coordinates relative to the source row: the removed section counts as removed system text.
    info = {**info, 'old_system': info['old_system'] + shift, 'tool_chars': info['tool_chars'] + len(section)}
    return new_row, None, info
