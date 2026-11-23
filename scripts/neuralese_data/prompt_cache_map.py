#!/usr/bin/env python3
"""Shared-prefix (prompt cache) map of recorded model calls, in LFM3 tokens (plans/semantic-fp/02 §2.2, R0).

Every engine (SGLang's radix cache, the managed llama-server's slot cache, wllama's slot save/restore, the Python native
engine) and the learner's shared-prefix tree reuse the longest token prefix a call shares with an earlier call; all
tokens after the first difference are recomputed. For each call this reports that longest shared prefix against the
earlier calls of its episode and of its whole window (the source, in record order), and in which region of the prompt
the first differing token falls:

  system   the system prompt text          tools    the tool block (``<tools>`` ... in the system turn)
  opening  the call's opening user message history  later turns (the append-only transcript)
  end      the prompt is entirely a prefix of an earlier call's

and for a difference in the tool block, which tool's line differs first. Sources:

  --collect DIR     an expert-iteration/GRPO collection (DIR/groups/g*/rollouts.jsonl): exact actor prompt ids
  --held DIR        a held run (DIR/row-*/sample-*/results.jsonl: trajectory context + tools_offered), rendered
  --turns FILE      episodes with ``turns`` ({messages, tools}; .jsonl or .jsonl.zst; trajectory_ir episodes too)

``--rewrite`` renders every source from its messages and tools after passing them through
``ts-host/scripts/rewrite-recorded-prompt.mjs`` (this checkout's system prompt and tool list for the same call), so the
same records are measured under the current prompt. Without it, collect sources use their exact ids.

    prompt_cache_map.py --tokenizer TOK --out OUT.json [--collect DIR --max-groups N] [--held DIR] [--turns FILE] [--rewrite]
"""
from __future__ import annotations

import argparse
import array
import collections
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(HERE)]

BLOCK = 16   # prefix hashes every BLOCK tokens; the exact position is then found against the matching earlier call


def jsonl(path):
    if str(path).endswith('.zst'):
        import zstandard
        with open(path, 'rb') as raw:
            yield from (json.loads(line) for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw),
                                                                     encoding='utf-8') if line.strip())
    else:
        with open(path) as handle:
            yield from (json.loads(line) for line in handle if line.strip())


# ---- sources: (episode, invocation, messages, tools, exact ids or None) per call, in record order -------------------
def collect_calls(directory, max_groups):
    groups = sorted(glob.glob(str(Path(directory) / 'groups' / 'g*' / 'rollouts.jsonl')))
    for path in groups[:max_groups] if max_groups else groups:
        for row in jsonl(path):
            episode = f"{Path(path).parent.name}/{row.get('sample')}"
            for turn in row.get('turns') or []:
                items = (turn.get('sampling') or {}).get('prompt') or []
                ids = [t for item in items for t in (item.get('token_ids') or [-1])] if items else None
                yield episode, turn.get('invocation_id'), turn['messages'], turn.get('tools') or [], ids


def held_calls(directory):
    for path in sorted(glob.glob(str(Path(directory) / 'row-*' / 'sample-*' / 'results.jsonl'))):
        for index, row in enumerate(jsonl(path)):
            episode = f"{Path(path).parent.parent.name}/{Path(path).parent.name}/{index}"
            for call in row.get('trajectory') or []:
                if call.get('context') is not None:
                    yield episode, call.get('invocation_id'), call['context'], call.get('tools_offered') or [], None


def turns_calls(path):
    for index, episode in enumerate(jsonl(path)):
        turns = episode.get('turns')
        if not turns and episode.get('trajectory_ir'):
            from live_teacher_sources import observed_training_turns
            turns = observed_training_turns(episode['trajectory_ir'])
        for turn in turns or []:
            yield str(episode.get('key', index)), turn.get('invocation_id'), turn['messages'], turn.get('tools') or [], None


# ---- rendering ---------------------------------------------------------------------------------------------------
class Renderer:
    def __init__(self, tokenizer_dir):
        from sft_sources import TemplateRenderer
        self.renderer = TemplateRenderer(tokenizer_dir)
        tok = self.renderer.tokenizer
        self.im_start = tok.convert_tokens_to_ids('<|im_start|>')
        self.tools_open = array.array('i', tok('<tools>', add_special_tokens=False)['input_ids'])
        self.tok = tok

    def ids(self, messages, tools):
        from natlang_neuralese.serve.targets import _items
        prompt = self.renderer.render_request(messages, tools)
        items = _items(self.renderer, prompt.segments, prompt.blocks, prompt.escape_nonce,
                       prompt.block_stop_gradients, prompt.block_source_occurrences)
        return [value if kind == 'tok' else -1 for kind, value in items]

    def regions(self, ids):
        """(tools start, opening start, history start) token positions."""
        starts = [i for i, t in enumerate(ids) if t == self.im_start][:3]
        opening = starts[1] if len(starts) > 1 else len(ids)
        history = starts[2] if len(starts) > 2 else len(ids)
        n = len(self.tools_open)
        tools = next((i for i in range(opening - n) if ids[i:i + n] == self.tools_open), opening)
        return tools, opening, history

    def tool_at(self, ids, tools_start, opening, position):
        """The name of the tool whose line holds ``position`` (a position inside the tool block)."""
        before = self.tok.decode(list(ids[tools_start:position]))
        block = self.tok.decode(list(ids[tools_start:opening]))
        line = block[:len(before)].rsplit('\n', 1)[-1] + block[len(before):].split('\n', 1)[0]
        marker = '"name": "'
        at = line.find(marker)
        return line[at + len(marker):].split('"', 1)[0] if at >= 0 else 'tools-frame'


def rewrite(calls, repo):
    """Pass (messages, tools) of every call through this checkout's prompt (rewrite-recorded-prompt.mjs), batched."""
    script = Path(repo) / 'ts-host' / 'scripts' / 'rewrite-recorded-prompt.mjs'
    batch = []

    def flush():
        if not batch:
            return []
        payload = ''.join(json.dumps({'messages': m, 'tools': t}) + '\n' for (_, _, m, t, _) in batch)
        out = subprocess.run(['node', str(script)], input=payload, capture_output=True, text=True, check=True).stdout
        rows = [json.loads(line) for line in out.splitlines() if line.strip()]
        assert len(rows) == len(batch), (len(rows), len(batch))
        done = [(e, i, r['messages'], r['tools'], None) for (e, i, _, _, _), r in zip(batch, rows)]
        batch.clear()
        return done
    for call in calls:
        batch.append(call)
        if len(batch) >= 200:
            yield from flush()
    yield from flush()


# ---- measurement ---------------------------------------------------------------------------------------------------
def block_hashes(ids):
    raw, out, h = ids.tobytes(), [], hashlib.blake2b(digest_size=12)
    for start in range(0, len(ids) - BLOCK + 1, BLOCK):
        h.update(raw[4 * start:4 * (start + BLOCK)])
        out.append(h.copy().digest())
    return out


class PrefixIndex:
    """Earlier calls' prompts, indexed by their block-prefix hashes; ``longest(ids)`` -> (shared tokens, call index)."""

    def __init__(self):
        self.first = {}
        self.calls = []

    def longest(self, ids, hashes):
        blocks, owner = 0, None
        for k, h in enumerate(hashes):
            if h not in self.first:
                break
            blocks, owner = k + 1, self.first[h]
        if owner is None:    # under one block shared: any earlier call tells the exact count within it
            if not self.calls:
                return 0, None
            owner = self.calls[-1]
        other = self.calls_ids[owner]
        shared = blocks * BLOCK
        limit = min(len(other), len(ids))
        while shared < limit and other[shared] == ids[shared]:
            shared += 1
        return shared, owner

    def add(self, index, hashes):
        for h in hashes:
            self.first.setdefault(h, index)
        self.calls.append(index)


def measure(name, calls, renderer):
    ids_of = []
    window = PrefixIndex()
    window.calls_ids = ids_of
    episodes = {}
    seen_invocations = set()
    rows = []
    systems, toolblocks, toolblocks_head = collections.Counter(), collections.Counter(), collections.Counter()
    for episode, invocation, messages, tools, exact in calls:
        ids = array.array('i', exact if exact is not None else renderer.ids(messages, tools))
        index = len(ids_of)
        ids_of.append(ids)
        hashes = block_hashes(ids)
        tools_start, opening, history = renderer.regions(ids)
        systems[hashlib.sha1(ids[:tools_start].tobytes()).hexdigest()[:12]] += 1
        toolblocks[hashlib.sha1(ids[tools_start:opening].tobytes()).hexdigest()[:12]] += 1
        names = [((t.get('function') or t).get('name')) for t in tools]
        toolblocks_head[','.join(n for n in names if n != 'return_result')] += 1
        mine = episodes.setdefault(episode, PrefixIndex())
        mine.calls_ids = ids_of
        if invocation is None:   # the call's opening message names it
            invocation = hashlib.sha1(json.dumps([m.get('content') for m in messages if m.get('role') == 'user'][:1],
                                                 default=str).encode()).hexdigest()
        first_turn = (episode, invocation) not in seen_invocations
        seen_invocations.add((episode, invocation))
        result = {'tokens': len(ids), 'first_turn': first_turn, 'system_tokens': tools_start,
                  'tool_tokens': opening - tools_start, 'opening_tokens': history - opening}
        for scope, index_ in (('episode', mine), ('window', window)):
            shared, owner = index_.longest(ids, hashes)
            region = ('end' if shared >= len(ids) else 'system' if shared < tools_start else 'tools' if shared < opening
                      else 'opening' if shared < history else 'history')
            result[scope] = {'shared': shared, 'region': region}
            if region == 'tools':
                result[scope]['tool'] = renderer.tool_at(ids, tools_start, opening, shared)
        rows.append(result)
        mine.add(index, hashes)
        window.add(index, hashes)
    return summarize(name, rows, systems, toolblocks, toolblocks_head)


def summarize(name, rows, systems, toolblocks, toolblocks_head):
    def part(selected):
        out = {'calls': len(selected), 'prompt_tokens': sum(r['tokens'] for r in selected)}
        for scope in ('episode', 'window'):
            shared = sum(r[scope]['shared'] for r in selected)
            out[scope] = {
                'shared_tokens': shared,
                'recomputed_tokens': out['prompt_tokens'] - shared,
                'hit_rate': round(shared / out['prompt_tokens'], 4) if out['prompt_tokens'] else None,
                'mean_shared': round(shared / len(selected), 1) if selected else None,
                'first_difference': dict(collections.Counter(r[scope]['region'] for r in selected).most_common()),
                'first_differing_tool': dict(collections.Counter(r[scope]['tool'] for r in selected
                                                                 if r[scope]['region'] == 'tools').most_common())}
        return out
    first = [r for r in rows if r['first_turn']]
    n = len(rows) or 1
    return {'source': name, 'all_calls': part(rows), 'invocation_openings': part(first),
            'mean_system_tokens': round(sum(r['system_tokens'] for r in rows) / n, 1),
            'mean_tool_tokens': round(sum(r['tool_tokens'] for r in rows) / n, 1),
            'mean_opening_tokens': round(sum(r['opening_tokens'] for r in rows) / n, 1),
            'distinct_system_texts': len(systems), 'distinct_tool_blocks': len(toolblocks),
            'distinct_tool_lists_without_return_result': dict(toolblocks_head.most_common(12))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--collect', action='append', default=[])
    parser.add_argument('--max-groups', type=int, default=40)
    parser.add_argument('--held', action='append', default=[])
    parser.add_argument('--turns', action='append', default=[])
    parser.add_argument('--rewrite', action='store_true', help='measure the records under this checkout\'s prompt')
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    renderer = Renderer(args.tokenizer)
    sources = ([(f'collect:{d}', lambda d=d: collect_calls(d, args.max_groups)) for d in args.collect] +
               [(f'held:{d}', lambda d=d: held_calls(d)) for d in args.held] +
               [(f'turns:{f}', lambda f=f: turns_calls(f)) for f in args.turns])
    report = {'schema': 'natlang.prompt-cache-map/1', 'label': args.label, 'block': BLOCK, 'rewrite': args.rewrite,
              'tokenizer': renderer.renderer.identity, 'sources': []}
    from natlang_neuralese.common.code_provenance import code_provenance
    report['code'] = code_provenance()
    for name, calls in sources:
        stream = calls()
        if args.rewrite:
            stream = rewrite(stream, REPO)
        report['sources'].append(measure(name, stream, renderer))
        print(json.dumps({'source': name, 'calls': report['sources'][-1]['all_calls']['calls']}), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)   # after each source: a later failure keeps these
        Path(args.out).write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
