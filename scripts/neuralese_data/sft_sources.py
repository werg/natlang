#!/usr/bin/env python3
"""SFT data set from verified trajectories: GRPO successes and teacher successes on the GRPO pool, replayed through the
current runtime and oracle, capped, deduplicated, held-excluded and rendered with the LFM3 chat template
(plans/neuralese/SFT_SOURCES_20261117.md).

Stages (each reads the previous stage's output under WORK; read-only on every source):

  extract  every `task_success` episode of the GRPO rollout archives (`--rl RUN_DIR:ARCHIVE_DIR`, the run's live
           group directories included) and of teacher-generation pools (`--teacher POOL_DIR`) -> WORK/episodes.jsonl.zst
           (`--out-name`: a growing teacher pool is re-extracted alone as WORK/episodes-teacher.jsonl.zst before a build;
           the build refuses manifest successes that no extraction covers):
           the task row (natlang.program/2, from the group's cases.jsonl), the semantic turns ({messages, tools, target})
           and, for student episodes, the exact actor prompt/sample token ids of each call (renderer check only).
  replay   each episode's task runs again under this checkout's runtime (`task_rollouts.prepare`, `run_bonsai_queue.py`,
           `task_rollouts.score`, the teacher_gen path) against a replay server that answers every model request with
           the recorded reply of the same call (matched by its visible messages); WORK/replay/<key>/rollouts.jsonl is the
           current runtime's record of the same decisions and its oracle verdict. A call the recording has no reply for
           ends the episode (the actions no longer lead where they led).
  build    the replayed successes -> per-call rows: held-v3/v2 closure excluded, per-task cap, near-duplicate removal,
           tasks weighted toward low base pass rates, per-source shares, one epoch; rendered with the chat template
           (prompt zero-weight, the reply the target) into OUT/rows.jsonl.zst + OUT/chains.pkl (`exp0_sft.py train
           --chains`) + OUT/build.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(HERE), str(REPO / 'scripts/rl_tasks')]

from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402

SUCCESS = 'task_success'
OPS = ('data_nvme', 'lfm3-rl-dgx-20261029', 'ops')


def zstd_lines(path):
    """Lines of a .zst or plain JSONL file, streamed."""
    import io
    import zstandard
    if str(path).endswith('.zst'):
        with open(path, 'rb') as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
                yield from io.TextIOWrapper(reader, encoding='utf-8')
    else:
        with open(path, encoding='utf-8') as handle:
            yield from handle


class ZstdWriter:
    """JSONL writer to path.zst through a temp file renamed on close."""

    def __init__(self, path, level=6):
        import zstandard
        self.path = Path(path)
        self.partial = self.path.with_suffix(self.path.suffix + '.partial')
        self.raw = self.partial.open('wb')
        self.writer = zstandard.ZstdCompressor(level=level).stream_writer(self.raw)
        self.rows = 0

    def write(self, row):
        self.writer.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
        self.rows += 1

    def close(self):
        self.writer.close()   # ends the frame and closes the file
        os.replace(self.partial, self.path)


# ---- extract ---------------------------------------------------------------------------------------------------------
def exact_tokens(turn):
    """(prompt ids, sampled ids) of a student call from its sampling capture, or None (teacher call, native imports)."""
    sampling = turn.get('sampling') or {}
    prompt = sampling.get('prompt') or []
    if not prompt or any(part.get('kind') != 'tokens' for part in prompt):
        return None
    ids = [i for part in prompt for i in part['token_ids']]
    return {'prompt_ids': ids, 'sample_ids': [s['token_id'] for s in sampling.get('samples') or []]}


def semantic_turn(turn, keep_exact):
    out = {k: turn.get(k) for k in ('messages', 'tools', 'target', 'invocation_id')}
    out['completion_tokens'] = (turn.get('usage') or {}).get('completion_tokens')
    if keep_exact:
        out['exact'] = exact_tokens(turn)
    return out


def group_cases(directory):
    rows = {}
    path = Path(directory) / 'cases.jsonl'
    if path.exists():
        for line in path.open():
            if line.strip():
                row = json.loads(line)
                rows[row['id']] = row
    return rows


def rl_rollout_files(run_dir, archive_dir):
    """(group name, rollouts file, group dir) of every group of one GRPO run: archived (zst) or still live."""
    seen = set()
    for path in sorted(Path(archive_dir).glob('g*.rollouts.jsonl.zst')):
        name = path.name.split('.')[0]
        seen.add(name)
        yield name, path, Path(run_dir) / 'groups' / name
    for directory in sorted(Path(run_dir, 'groups').glob('g*')):
        if directory.name not in seen and (directory / 'rollouts.jsonl').exists() and (directory / 'score.json').exists():
            yield directory.name, directory / 'rollouts.jsonl', directory


def extract(args):
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    out = ZstdWriter(work / f'{args.out_name}.jsonl.zst')
    census = collections.OrderedDict()
    group_rates = collections.defaultdict(lambda: [0, 0])     # task -> [successes, scored] over all student groups
    for spec in args.rl:
        run_dir, archive_dir = spec.split(':')
        label = 'rl:' + '/'.join(Path(run_dir).parts[-2:])
        c = census.setdefault(label, collections.Counter())
        for name, path, group_dir in rl_rollout_files(run_dir, archive_dir):
            cases = group_cases(group_dir)
            c['groups'] += 1
            if not path.exists():   # a live group the run archived since the listing
                path = Path(archive_dir) / f'{name}.rollouts.jsonl.zst'
            for line in zstd_lines(path):
                if not line.strip():
                    continue
                row = json.loads(line)
                c['episodes'] += 1
                if row.get('unscored') or row.get('reward') is None:
                    c['unscored'] += 1
                    continue
                group_rates[row['task_id']][1] += 1
                if row.get('outcome') != SUCCESS or float(row['reward'].get('quality') or 0) < 1:
                    continue
                group_rates[row['task_id']][0] += 1
                task = cases.get(row['task_id'])
                if task is None:
                    c['success_without_task_row'] += 1
                    continue
                c['successes'] += 1
                out.write({'key': f"{label}:{name}:{row['sample']}", 'source': 'rl-own', 'run': label, 'group': name,
                           'sample': row['sample'], 'seed': row.get('seed'), 'task_id': row['task_id'],
                           'family': row.get('family'), 'policy': (row.get('policy') or {}).get('checkpoint_sha256'),
                           'task': task, 'turns': [semantic_turn(t, True) for t in row['turns']],
                           'execution_graph_events': len(row.get('execution_graph') or [])})
    for pool in args.teacher:
        label = 'teacher:' + '/'.join(Path(pool).parts[-2:])
        c = census.setdefault(label, collections.Counter())
        for line in Path(pool, 'manifest.jsonl').open():
            if not line.strip():
                continue
            record = json.loads(line)
            c['tasks'] += 1
            directory = Path(record['rollouts']).parent
            cases = group_cases(directory)
            for rline in Path(record['rollouts']).open():
                if not rline.strip():
                    continue
                row = json.loads(rline)
                c['episodes'] += 1
                if row.get('outcome') != SUCCESS or float((row.get('reward') or {}).get('quality') or 0) < 1:
                    continue
                task = cases.get(row['task_id'])
                if task is None:
                    c['success_without_task_row'] += 1
                    continue
                c['successes'] += 1
                out.write({'key': f"{label}:{directory.name}:{row['sample']}", 'source': 'teacher', 'run': label,
                           'group': directory.name, 'sample': row['sample'], 'seed': row.get('seed'),
                           'task_id': row['task_id'], 'family': row.get('family'), 'teacher': record.get('teacher'),
                           'base_rank': record.get('base_rank'), 'task': task,
                           'turns': [semantic_turn(t, False) for t in row['turns']],
                           'execution_graph_events': len(row.get('execution_graph') or [])})
    out.close()
    if args.rl:
        rates = {k: {'successes': s, 'scored': n} for k, (s, n) in sorted(group_rates.items())}
        (work / 'rl-group-rates.json').write_text(json.dumps(rates, indent=0) + '\n')
    summary = {k: dict(v) for k, v in census.items()}
    (work / f'{args.out_name}-extract.json').write_text(json.dumps({'sources': summary, 'episodes': out.rows,
                                                    'rl': args.rl, 'teacher': args.teacher}, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


def episode_files(work):
    """Every extraction of WORK: `episodes.jsonl.zst` and `episodes-*.jsonl.zst` (e.g. a teacher pool re-extracted at
    build time as `--out-name episodes-teacher`)."""
    return sorted(Path(work).glob('episodes*.jsonl.zst'))


def all_episodes(work):
    """The extracted episodes of WORK, each key once (a later extraction of the same episode is the same record)."""
    seen = set()
    for path in episode_files(work):
        for line in zstd_lines(path):
            if line.strip():
                episode = json.loads(line)
                if episode['key'] not in seen:
                    seen.add(episode['key'])
                    yield episode


def teacher_coverage(pools, extracted):
    """(successes in the pools' manifests now, those not extracted yet): the build refuses a stale teacher snapshot."""
    total, missing = 0, []
    for pool in pools:
        label = 'teacher:' + '/'.join(Path(pool).parts[-2:])
        for line in Path(pool, 'manifest.jsonl').open():
            if not line.strip():
                continue
            record = json.loads(line)
            directory = Path(record['rollouts']).parent.name
            for episode in record.get('episodes') or ():
                if episode.get('outcome') == SUCCESS:
                    total += 1
                    key = f"{label}:{directory}:{episode['sample']}"
                    if key not in extracted:
                        missing.append(key)
    return total, missing


# ---- rendering -------------------------------------------------------------------------------------------------------
class TemplateRenderer:
    """The serving engine's own request rendering (`serve.engine.Engine.render_request`, `serve.targets.target_items`)
    on a tokenizer alone: no model is loaded. `render(messages, tools, target)` -> (prompt ids, target ids), the target
    being the assistant reply after the generation prompt through its end token. `check` compares it with the actor's
    recorded prompt and sampled ids (the RL captures)."""

    def __init__(self, tokenizer_dir):
        from transformers import AutoTokenizer
        from natlang_neuralese.serve.chat import bind_history_reasoning, tool_call_format
        from natlang_neuralese.serve.engine import Engine
        # the checkpoint's config.json minus auto_map (no remote code: only the tokenizer is loaded)
        config = json.loads(Path(tokenizer_dir, 'model_type.json').read_text())
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=False)
        self.tokenizer = bind_history_reasoning(tokenizer, model_type=config['model_type'])
        self.tool_call_format = tool_call_format(config['model_type'])
        self.prompt_rendering = 'full'
        self.identity = {'tokenizer_sha256': sha256_file(Path(tokenizer_dir, 'tokenizer.json')),
                         'chat_template_sha256': sha256_file(Path(tokenizer_dir, 'chat_template.jinja')),
                         'model_type': config['model_type'],
                         'history_reasoning': getattr(self.tokenizer, 'natlang_history_reasoning', None)}
        self._engine = Engine

    # the engine's methods, bound to this tokenizer-only object
    def _template(self, messages, tools):
        return self._engine._template(self, messages, tools)

    def render_request(self, messages, tools=None, template=None, *, response_tools=None):
        return self._engine.render_request(self, messages, tools, template, response_tools=response_tools)

    def _tokens(self, text):
        return self._engine._tokens(self, text)

    def _template_tokens(self, text, escape_nonce=''):
        return self._engine._template_tokens(self, text, escape_nonce)

    @property
    def specials(self):
        return self._engine.specials.fget(self)

    def think_context_end(self, target_ids):
        """For a reply with no reasoning: the number of leading target tokens that are the template's empty think block
        (through `</think>` and the whitespace after it), to be zero-weight context; the action after it is the target."""
        close = self.tokenizer.convert_tokens_to_ids('</think>')
        ids = list(target_ids)
        if close not in ids:
            return 0
        end = ids.index(close) + 1
        while end < len(ids) and not self.tokenizer.decode([ids[end]]).strip():
            end += 1
        return end

    def render(self, messages, tools, target):
        from natlang_neuralese.serve.targets import target_items
        target = {k: v for k, v in target.items() if v is not None}
        before, rest = target_items(self, list(messages), tools, target)
        if any(kind != 'tok' for kind, _ in before + rest):
            raise ValueError('native-state blocks in a text-only rendering')
        return [i for _, i in before], [i for _, i in rest]


def reasoning_free_context(renderer, source, target_message, target_ids):
    """Leading target tokens that are context, not loss: a teacher's or October reply without reasoning renders the
    template's empty think block, and training it would teach the student to close its reasoning at once (no verified
    outcome selects that). The student's own replies keep every token as sampled."""
    if source == 'rl-own' or (target_message.get('reasoning_content') or '').strip():
        return 0
    return renderer.think_context_end(target_ids)


def check_render(args):
    """Render recorded student calls and compare with the actor's exact prompt and sampled ids."""
    renderer = TemplateRenderer(args.tokenizer)
    counts = collections.Counter()
    examples = []
    for episode in all_episodes(args.work):
        if episode['source'] != 'rl-own':
            continue
        for turn in episode['turns']:
            exact = turn.get('exact')
            if not exact:
                counts['no_exact'] += 1
                continue
            try:
                prompt, target = renderer.render(turn['messages'], turn['tools'], turn['target'])
            except Exception as error:
                counts['render_error'] += 1
                examples.append({'key': episode['key'], 'error': repr(error)[:300]})
                continue
            counts['turns'] += 1
            same_prompt = prompt == exact['prompt_ids']
            counts['prompt_equal'] += same_prompt
            # a later call's history holds the earlier replies as sampled, which may tokenize differently from the text
            counts['prompt_text_equal'] += (renderer.tokenizer.decode(prompt)
                                            == renderer.tokenizer.decode(exact['prompt_ids']))
            sample = exact['sample_ids']
            counts['target_equal'] += target == sample
            # a capture may end at a stop without the end token, or the reply may tokenize differently from the sample
            counts['target_text_equal'] += (renderer.tokenizer.decode(target).rstrip('<|im_end|>').rstrip()
                                            == renderer.tokenizer.decode(sample).rstrip('<|im_end|>').rstrip())
            if not same_prompt and len(examples) < 20:
                first = next((i for i, (a, b) in enumerate(zip(prompt, exact['prompt_ids'])) if a != b),
                             min(len(prompt), len(exact['prompt_ids'])))
                examples.append({'key': episode['key'], 'len': [len(prompt), len(exact['prompt_ids'])], 'first': first,
                                 'rendered': renderer.tokenizer.decode(prompt[max(0, first - 20):first + 40]),
                                 'recorded': renderer.tokenizer.decode(exact['prompt_ids'][max(0, first - 20):first + 40])})
            if counts['turns'] >= args.turns:
                break
        if counts['turns'] >= args.turns:
            break
    report = {'counts': dict(counts), 'identity': renderer.identity, 'examples': examples}
    Path(args.work, 'render-check.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(report['counts']))


# ---- replay ----------------------------------------------------------------------------------------------------------
def visible_key(messages):
    """What a call saw, minus the runtime's own wording: every non-system message's role, text, tool calls (name and
    arguments) and tool output. Tool-call ids and reasoning are left out (the runtime echoes them as it received them)."""
    out = []
    for m in messages:
        if m.get('role') == 'system':
            continue
        calls = [((c.get('function') or {}).get('name'), (c.get('function') or {}).get('arguments'))
                 for c in m.get('tool_calls') or []]
        content = m.get('content')
        if isinstance(content, list):   # content parts: their text in order
            content = ''.join(part.get('text', '') if isinstance(part, dict) else str(part) for part in content)
        out.append((m.get('role'), content or '', calls))
    return json.dumps(out, ensure_ascii=False)


def opening_key(messages):
    """The call's invocation: its first non-system message (the task or nl() request)."""
    for m in messages:
        if m.get('role') != 'system':
            return json.dumps([m.get('role'), m.get('content')], ensure_ascii=False)
    return ''


class ReplayEpisode:
    """The recorded calls of one episode, served once each: an exact visible-history match first; otherwise the next
    unserved call of the same invocation with as many visible messages (a tool output that differs from the recording,
    counted as 'drifted'); otherwise no reply (the episode ends)."""

    def __init__(self, turns):
        import threading
        self.lock = threading.Lock()
        self.turns = turns
        self.keys = [visible_key(t['messages']) for t in turns]
        self.openings = [opening_key(t['messages']) for t in turns]
        self.lengths = [sum(m.get('role') != 'system' for m in t['messages']) for t in turns]
        self.served = [False] * len(turns)
        self.log = []

    def answer(self, body):
        messages = body.get('messages') or []
        key, opening = visible_key(messages), opening_key(messages)
        length = sum(m.get('role') != 'system' for m in messages)
        with self.lock:
            match, kind = None, 'missing'
            for i, k in enumerate(self.keys):
                if not self.served[i] and k == key:
                    match, kind = i, 'exact'
                    break
            if match is None:
                for i in range(len(self.turns)):
                    if not self.served[i] and self.openings[i] == opening and self.lengths[i] == length:
                        match, kind = i, 'drifted'
                        break
            entry = {'kind': kind, 'turn': match, 'messages': length}
            if match is not None:
                self.served[match] = True
                recorded = self.turns[match]
                system = [m.get('content') for m in messages if m.get('role') == 'system']
                recorded_system = [m.get('content') for m in recorded['messages'] if m.get('role') == 'system']
                entry['system_same'] = system == recorded_system
                entry['tools_same'] = (json.dumps(body.get('tools') or [], sort_keys=True)
                                       == json.dumps(recorded.get('tools') or [], sort_keys=True))
                if kind == 'drifted':
                    recorded_visible = json.loads(self.keys[match])
                    now_visible = json.loads(key)
                    first = next((j for j, (a, b) in enumerate(zip(recorded_visible, now_visible)) if a != b), None)
                    entry['first_difference'] = first
                    if first is not None:
                        entry['recorded'] = str(recorded_visible[first])[:600]
                        entry['now'] = str(now_visible[first])[:600]
            self.log.append(entry)
        if match is None:
            return None
        target = {k: v for k, v in self.turns[match]['target'].items() if k != 'role' and v is not None}
        message = {'role': 'assistant', 'content': target.pop('content', '') or '', **target}
        tokens = self.turns[match].get('completion_tokens') or 0
        return {'id': f'replay-{match}', 'object': 'chat.completion', 'model': body.get('model', 'replay'),
                'choices': [{'index': 0, 'message': message,
                             'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}],
                'usage': {'prompt_tokens': 0, 'completion_tokens': tokens, 'total_tokens': tokens}}


class ReplayServer:
    """OpenAI-compatible chat endpoint on 127.0.0.1: POST /ep/<episode>/v1/chat/completions answers from that episode's
    recording (non-streaming JSON; the collector accepts it for a streamed request)."""

    def __init__(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.episodes = {}
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self._send(200, {'object': 'list', 'data': [{'id': 'replay', 'object': 'model'}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
                parts = self.path.strip('/').split('/')
                episode = server.episodes.get(parts[1]) if len(parts) > 2 and parts[0] == 'ep' else None
                reply = episode.answer(body) if episode is not None else None
                if reply is None:
                    self._send(500, {'error': {'message': 'replay: the recording has no reply for this request',
                                               'type': 'replay_missing'}})
                else:
                    self._send(200, reply)

            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


def safe_name(key):
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def has_native_state(episode):
    return any(isinstance(m.get('content'), list) and any(isinstance(part, dict) and part.get('type') == 'native_state'
                                                           for part in m['content'])
               for turn in episode['turns'] for m in turn['messages'])


def replay_episode(args, server, episode, env):
    """Run one recorded episode's task under this checkout's runtime against its recording; returns its replay record."""
    import shutil
    import subprocess
    import time
    from types import SimpleNamespace
    from natlang_neuralese.common.jsonio import read_jsonl
    from natlang_neuralese.rl import task_rollouts
    name = safe_name(episode['key'])
    directory = Path(args.work) / 'replay' / name
    done = directory / 'replay.json'
    if done.exists():
        return json.loads(done.read_text())
    if has_native_state(episode):   # its imported states came from the actor's store: no text endpoint can replay them
        return {'key': episode['key'], 'name': name, 'source': episode['source'], 'task_id': episode['task_id'],
                'outcome': 'skipped_native_state', 'exact': False, 'recorded_turns': len(episode['turns'])}
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    (directory / 'cases.jsonl').write_text(json.dumps(episode['task']) + '\n')
    task_rollouts.prepare(SimpleNamespace(
        cases=directory / 'cases.jsonl', out=directory, rows=1, offset=0, group=2, temperature=1.0,
        case_seconds=args.case_seconds, max_model_requests=args.max_model_requests, max_turns=128,
        seed_base=0, root_mode='student', match_seeds_from=None, policy_checkpoint_sha256=None))
    entry = list(read_jsonl(directory / 'queue.jsonl'))[0]
    entry.update(seed=episode.get('seed') or entry['seed'], prompt_rendering='full', request_retries=0,
                 context_tokens=131072 if episode['source'] == 'rl-own' else 65536)
    (directory / 'queue.jsonl').write_text(json.dumps(entry, sort_keys=True) + '\n')
    queue_file = directory / f"{entry['key']}.queue.jsonl"
    queue_file.write_text(json.dumps(entry, sort_keys=True) + '\n')
    (directory / 'chat-config.json').write_text('{}\n')
    replay = ReplayEpisode(episode['turns'])
    server.episodes[name] = replay
    url = f'http://127.0.0.1:{server.port}/ep/{name}/v1/chat/completions'
    command = [sys.executable, str(Path(args.runtime_repo) / 'scripts/run_bonsai_queue.py'), str(queue_file),
               str(directory / f"{entry['key']}.journal.jsonl"), '--runtime', str(Path(args.runtime_repo) / 'ts-host'),
               '--model-id', 'replay', '--model-concurrency', '1', '--server', f'http://127.0.0.1:{server.port}',
               '--chat-completions-url', url, '--chat-request-config', str(directory / 'chat-config.json')]
    started = time.time()
    with (directory / 'collector.log').open('w') as handle:
        result = subprocess.run(command, cwd=args.runtime_repo, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                timeout=args.case_seconds + 300)
    del server.episodes[name]
    with (directory / 'journal.jsonl').open('w') as merged:
        merged.write(Path(directory / f"{entry['key']}.journal.jsonl").read_text()
                     if (directory / f"{entry['key']}.journal.jsonl").exists() else '')
    outcome, turns = None, 0
    try:
        task_rollouts.score(directory, directory / 'rollouts.jsonl', neuralese_bonus=0., efficiency_weight=0.)
        rows = list(read_jsonl(directory / 'rollouts.jsonl'))
        if rows:
            outcome, turns = rows[0].get('outcome'), len(rows[0].get('turns') or [])
    except Exception as error:   # a scoring failure is this episode's verdict, never the whole replay's
        outcome = 'score_error: ' + repr(error)[:300]
    kinds = collections.Counter(e['kind'] for e in replay.log)
    record = {'key': episode['key'], 'name': name, 'source': episode['source'], 'task_id': episode['task_id'],
              'outcome': outcome, 'returncode': result.returncode, 'seconds': round(time.time() - started, 1),
              'recorded_turns': len(episode['turns']), 'replayed_turns': turns,
              'served': dict(kinds), 'unserved': replay.served.count(False),
              'system_changed': sum(1 for e in replay.log if e.get('system_same') is False),
              'tools_changed': sum(1 for e in replay.log if e.get('tools_same') is False),
              'log': replay.log[:64]}
    record['exact'] = (outcome == SUCCESS and record['unserved'] == 0 and kinds == {'exact': len(episode['turns'])}
                       and turns == len(episode['turns']))
    for path in directory.iterdir():   # keep the verdict, the replayed record and the logs; drop the job scratch
        if path.is_dir():
            shutil.rmtree(path)
    done.write_text(json.dumps(record) + '\n')
    return record


def replay(args):
    import concurrent.futures
    import threading
    work = Path(args.work)
    args.runtime_repo = str(Path(args.runtime_repo or REPO).resolve())
    from natlang_neuralese.common.code_provenance import code_provenance
    from natlang_neuralese.rl.held_eval import EPISODE_MODEL_CALLS, series
    args.max_model_requests = args.max_model_requests or EPISODE_MODEL_CALLS
    env = {**os.environ, 'NATLANG_REPO': args.runtime_repo,
           'PYTHONPATH': f'{args.runtime_repo}/training/neuralese:{args.runtime_repo}',
           'NATLANG_MODEL_FACING': series()['model_facing'], 'NATLANG_COLLECTOR_SHARED': '0'}
    server = ReplayServer()
    episodes = all_episodes(work)
    if args.limit:
        import itertools
        episodes = itertools.islice(episodes, args.limit)
    lock = threading.Lock()
    counts = collections.Counter()
    (work / 'replay').mkdir(exist_ok=True)
    with (work / 'replay-launches.jsonl').open('a') as handle:
        handle.write(json.dumps({'argv': sys.argv, 'runtime_repo': args.runtime_repo, 'code': code_provenance(),
                                 'model_facing': env['NATLANG_MODEL_FACING']}) + '\n')
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool, (work / 'replay.jsonl').open('a') as results:
        pending = set()

        def finished(future):
            try:
                record = future.result()
            except Exception as error:
                record = {'outcome': 'replay_error: ' + repr(error)[:300]}
            with lock:
                counts[str(record.get('outcome'))[:40]] += 1
                results.write(json.dumps({k: v for k, v in record.items() if k != 'log'}) + '\n')
                results.flush()
                if sum(counts.values()) % 50 == 0:
                    print(json.dumps(dict(counts)), flush=True)
        for episode in episodes:
            while len(pending) >= args.workers * 2:
                done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            future = pool.submit(replay_episode, args, server, episode, env)
            future.add_done_callback(finished)
            pending.add(future)
        concurrent.futures.wait(pending)
    print(json.dumps(dict(counts)), flush=True)


# ---- build -----------------------------------------------------------------------------------------------------------
WORD = None


def shingles(text, n=5):
    import re
    global WORD
    WORD = WORD or re.compile(r'\w+')
    words = [w.lower() for w in WORD.findall(text)]
    words = ['0' if w.isdigit() else w for w in words]
    return {' '.join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))} if words else set()


def minhash(items, perms=64):
    """64-permutation MinHash of a shingle set (the DATA_MIX_DIVERSITY.md method)."""
    import numpy as np
    if not items:
        return None
    hashes = np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little') for s in items],
                      dtype=np.uint64)
    rng = np.random.RandomState(20261117)
    a = rng.randint(1, 2**31 - 1, size=perms).astype(np.uint64)
    b = rng.randint(0, 2**31 - 1, size=perms).astype(np.uint64)
    return ((hashes[None, :] * a[:, None] + b[:, None]) % np.uint64(2**61 - 1)).min(axis=1)


class NearDuplicates:
    """LSH (16 bands of 4) over MinHash signatures; `seen(sig)` is true when an admitted item has estimated Jaccard >=
    threshold, else the item is admitted."""

    def __init__(self, threshold=0.8, bands=16, rows=4):
        self.threshold, self.bands, self.rows = threshold, bands, rows
        self.buckets = collections.defaultdict(list)
        self.signatures = []

    def seen(self, signature):
        if signature is None:
            return False
        candidates = set()
        for band in range(self.bands):
            key = (band, signature[band * self.rows:(band + 1) * self.rows].tobytes())
            candidates.update(self.buckets.get(key, ()))
        for index in candidates:
            if (self.signatures[index] == signature).mean() >= self.threshold:
                return True
        index = len(self.signatures)
        self.signatures.append(signature)
        for band in range(self.bands):
            self.buckets[(band, signature[band * self.rows:(band + 1) * self.rows].tobytes())].append(index)
        return False


def action_text(turns):
    """An episode's own outputs as text: replies, tool calls (name and arguments) and reasoning."""
    parts = []
    for turn in turns:
        target = turn['target']
        parts.append(target.get('reasoning_content') or '')
        parts.append(target.get('content') or '')
        for call in target.get('tool_calls') or []:
            parts.append(f"{call['function']['name']} {call['function']['arguments']}")
    return '\n'.join(parts)


def held_closure(manifest_path):
    """(ids, groups, source ids) of the held set and every earlier held set it excludes: task ids, split groups, the
    held rows' source groups and source ids (the separation grpo.load_tasks applies, for rows outside the pool too)."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    ids, groups, sources = {t['id'] for t in manifest['tasks']}, set(manifest['excluded_split_groups']), set()
    cases = manifest_path.parent / manifest['cases']['path']
    if sha256_file(cases) != manifest['cases']['sha256']:
        raise ValueError(f'held cases changed: {cases}')
    for line in cases.open():
        if line.strip():
            row = json.loads(line)
            ids.add(row['id'])
            groups.update(row.get('source_groups') or ())
            groups.add((row.get('curriculum') or {}).get('split_group') or row['id'])
            sources.update(row.get('source_ids') or ())
    for earlier in manifest.get('also_excludes') or ():
        if sha256_file(earlier['path']) != earlier['sha256']:
            raise ValueError('an excluded earlier held manifest changed: ' + earlier['path'])
        more = held_closure(earlier['path'])
        ids |= more[0]
        groups |= more[1]
        sources |= more[2]
    groups.discard(None)
    return ids, groups, sources


CHARS_PER_TOKEN = 3.3   # estimate for selecting October rows before rendering (calibrated in build.json)
DEFAULT_OCTOBER_GROUPS = ('luna-curricula=3', 'qwen-s72=2', 'fallow-v13r2=2', 'qwen-campaign=1', 'luna-early-cohorts=1',
                          'luna-v19-decision-children=1', 'luna-clinc-children=1', 'luna-ep-campaigns=1',
                          'student-projection=1', 'acq-swe-rebench-openhands=2', 'acq-d32-openresearcher=2',
                          'acq-d13-cooperbench=1')


def held_hit(task_id, source_ids, groups, held):
    held_ids, held_groups, held_sources = held
    return task_id in held_ids or bool(set(groups) & held_groups) or bool(set(source_ids) & held_sources)


def water_fill(sizes, weights, budget):
    """Per-group budgets: shares of `budget` by weight; a group with less than its share gives the rest to the others."""
    budgets, open_groups, left = {}, {g for g in sizes if weights.get(g, 0) > 0}, budget
    while open_groups:
        total = sum(weights[g] for g in open_groups)
        small = {g for g in open_groups if sizes[g] <= left * weights[g] / total}
        if not small:
            for g in open_groups:
                budgets[g] = left * weights[g] / total
            break
        for g in small:
            budgets[g] = sizes[g]
            left -= sizes[g]
        open_groups -= small
    return budgets


def core_candidates(args, work, funnel, held):
    """Replay-verified successes (exact replay, current oracle success), held closure removed: metadata and signatures
    only (the turns are read again when rendering)."""
    from natlang_neuralese.rl.held_eval import exclusion_ids
    replays = {}
    for line in (work / 'replay.jsonl').open():
        if line.strip():
            record = json.loads(line)
            if 'key' in record:
                replays[record['key']] = record
    candidates, tasks = [], {}
    extracted = set()
    for episode in all_episodes(work):
        extracted.add(episode['key'])
        source = episode['source']
        f = funnel[source]
        f['successes'] += 1
        record = replays.get(episode['key'])
        if record is None:
            f['not_replayed'] += 1
            continue
        if record['outcome'] == 'skipped_native_state':
            f['native_state'] += 1
            continue
        if record['outcome'] != SUCCESS:
            f['replay_failed'] += 1
            continue
        if not record.get('exact'):
            f['replay_inexact'] += 1
            continue
        f['replayed_exact'] += 1
        task = episode['task']
        tasks[episode['task_id']] = task
        candidates.append({'key': episode['key'], 'source': source, 'run': episode.get('run'),
                           'task_id': episode['task_id'], 'family': episode.get('family') or task.get('family'),
                           'domain': (task.get('curriculum') or {}).get('domain'),
                           'teacher': (episode.get('teacher') or {}).get('model_id'), 'policy': episode.get('policy'),
                           'name': record['name'], 'signature': minhash(shingles(action_text(episode['turns'])))})
    total, missing = teacher_coverage(args.teacher_pool, extracted)
    funnel['teacher']['manifest_successes_now'] = total
    funnel['teacher']['manifest_successes_not_extracted'] = len(missing)
    if missing and not args.allow_stale_teacher:
        raise SystemExit(f'{len(missing)} of {total} teacher successes in the pool manifests are not extracted yet '
                         f'(e.g. {missing[0]}): run extract --teacher POOL --out-name episodes-teacher and replay first')
    excluded = exclusion_ids(args.held_manifest, list(tasks.values()))
    kept = []
    for c in candidates:
        task = tasks[c['task_id']]
        groups = list(task.get('source_groups') or ()) + [(task.get('curriculum') or {}).get('split_group') or task['id']]
        if c['task_id'] in excluded or held_hit(c['task_id'], task.get('source_ids') or (), groups, held):
            funnel[c['source']]['held_excluded'] += 1
            continue
        kept.append(c)
    return kept, tasks


def october_candidates(args, funnel, held, weights):
    """One pass over the October candidate rows (sft-sources-20261117/october, per-call rows rebuilt for the current
    runtime): rows whose eval actions do not compile (`ts-host/scripts/check-eval-actions.mjs`), non-teacher rows and
    held rows are dropped; the rest are grouped into trajectories with a size estimate and an action signature (the
    MinHash of a union is the element-wise minimum of its parts' MinHashes)."""
    import numpy as np
    root = Path(args.october)
    trajectories = {}
    for group in sorted(weights):
        f = funnel['october:' + group]
        failed = set()
        check = root / 'eval-check' / f'{group}.jsonl'
        if check.exists():
            for line in check.open():
                row = json.loads(line)
                if row['failed']:
                    failed.add(row['id'])
        elif not group.startswith('acq-'):
            raise ValueError(f'no eval check for natlang group {group}: run ts-host/scripts/check-eval-actions.mjs')
        for line in zstd_lines(root / f'{group}.jsonl.zst'):
            row = json.loads(line)
            f['rows'] += 1
            provenance = row.get('provenance') or {}
            if row['id'] in failed:
                f['eval_does_not_compile'] += 1
                continue
            if group == 'student-projection' and provenance.get('collection_role') != 'teacher':
                f['not_teacher'] += 1
                continue
            if held_hit(row.get('task_id'), row.get('source_ids') or (), row.get('source_groups') or (), held):
                f['held_excluded'] += 1
                continue
            key = f"october:{group}:{provenance.get('trajectory') or row.get('task_id')}"
            size = (len(json.dumps(row['messages'], ensure_ascii=False)) + len(json.dumps(row['tools'], ensure_ascii=False))
                    + len(json.dumps(row['target'], ensure_ascii=False))) / CHARS_PER_TOKEN
            signature = minhash(shingles(action_text([row])))
            t = trajectories.get(key)
            if t is None:
                t = trajectories[key] = {'key': key, 'source': 'october:' + group, 'group': group, 'run': group,
                                         'task_id': row.get('task_id'), 'family': row.get('family'),
                                         'domain': row.get('domain'), 'teacher': provenance.get('model') or provenance.get('teacher'),
                                         'policy': None, 'rows': 0, 'size': 0., 'signature': signature}
            elif signature is not None:
                t['signature'] = signature if t['signature'] is None else np.minimum(t['signature'], signature)
            t['rows'] += 1
            t['size'] += size
            f['rows_kept'] += 1
    return list(trajectories.values())


def round_robin(members, rng, order):
    """A task's episodes taking turns by producer (source and teacher model), the producers in `order`: a cap keeps
    different producers' solutions before a second one of any producer (no single teacher fills a task)."""
    producers = collections.defaultdict(list)
    for c in members:
        producers[(order(c), c['source'], c.get('teacher') or '')].append(c)
    queues = [producers[k] for k in sorted(producers, key=lambda k: (k[0], rng.random()))]
    for queue in queues:
        rng.shuffle(queue)
    out = []
    while any(queues):
        for queue in queues:
            if queue:
                out.append(queue.pop())
    return out


def capped(candidates, rng, funnel, cap_of, family_share, order):
    """Per-task caps (`cap_of(task_id)`, producers taking turns), then a per-family cap of `family_share` of the rest."""
    by_task = collections.defaultdict(list)
    for c in candidates:
        by_task[c['task_id']].append(c)
    selected = []
    for task_id, members in sorted(by_task.items(), key=lambda item: str(item[0])):
        members = round_robin(members, rng, order)
        cap = cap_of(task_id)
        selected += members[:cap]
        for c in members[cap:]:
            funnel[c['source']]['over_task_cap'] += 1
    by_family = collections.defaultdict(list)
    for c in selected:
        by_family[c['family']].append(c)
    limit = max(1, int(family_share * len(selected)))
    final = []
    for family, members in sorted(by_family.items(), key=lambda item: str(item[0])):
        rng.shuffle(members)
        final += members[:limit]
        for c in members[limit:]:
            funnel[c['source']]['over_family_cap'] += 1
    return final


# Program constructs the held set tests heavily (held-v3: 30% Neuralese results, 30% folder reducers, 18% iterateOn,
# 11% decide, 75% several functions) while dataset wrappers were 53% of v2's pool tasks and 9% of held-v3's.
HELD_CONSTRUCTS = ('neuralese_result', 'neuralese_argument', 'decide_call', 'folder_reducer', 'iterate',
                   'readout_decision', 'call_depth_ge_3', 'multi_function')


def task_constructs(task):
    """(held constructs the task's program declares, whether it is a one-function dataset wrapper)."""
    from measure_repetition import program_features
    try:
        features = program_features(task)
    except (KeyError, TypeError, AttributeError):
        return (), False
    return tuple(k for k in HELD_CONSTRUCTS if features.get(k)), bool(features.get('dataset_wrapper'))


def wrapper_capped(selected, rng, funnel, is_wrapper, share):
    """Keep dataset-wrapper episodes to at most `share` of the selection."""
    wrappers = [c for c in selected if is_wrapper(c)]
    rest = [c for c in selected if not is_wrapper(c)]
    limit = int(share / (1 - share) * len(rest)) if share < 1 else len(wrappers)
    rng.shuffle(wrappers)
    for c in wrappers[limit:]:
        funnel[c['source']]['over_wrapper_share'] += 1
    return rest + wrappers[:limit]


def build(args):
    import random
    import numpy as np
    import pickle
    from natlang_neuralese.common.code_provenance import code_provenance
    from exp0_sft import MAX_LEN, chains_of
    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    funnel = collections.defaultdict(collections.Counter)
    held = held_closure(args.held_manifest)
    # base pass rates: the exp1 measurement of the start policy, else the run groups' own (near-start) outcomes
    rates = {k: (v['successes'], v['scored']) for k, v in json.loads(Path(args.base_rates).read_text()).items()
             if v.get('scored')}
    rates.update({k: (v['successes'], v['scored']) for k, v in json.loads((work / 'rl-group-rates.json').read_text()).items()
                  if k not in rates and v['scored']})

    def base_rate(task_id):
        s, n = rates.get(task_id, (None, 0))
        return s / n if n else None

    # 1. verified successes of the GRPO pool (own and teacher): near-duplicates out, task caps by base pass rate
    core, tasks = core_candidates(args, work, funnel, held)
    rng.shuffle(core)
    core.sort(key=lambda c: 0 if c['source'] == 'teacher' else 1)   # stable: teacher first, else shuffled
    dedup = NearDuplicates(args.dup_threshold)
    unique = []
    for c in core:
        if dedup.seen(c['signature']):
            funnel[c['source']]['near_duplicate'] += 1
        else:
            unique.append(c)

    constructs = {task_id: task_constructs(task) for task_id, task in tasks.items()}

    def core_cap(task_id):
        top = args.rich_max_per_task if constructs.get(task_id, ((), False))[0] else args.max_per_task
        p = base_rate(task_id)
        return top if p is None else max(1, round(1 + (top - 1) * (1 - p)))
    core = capped(unique, rng, funnel, core_cap, args.max_family_share, lambda c: 0 if c['source'] == 'teacher' else 1)
    core = wrapper_capped(core, rng, funnel, lambda c: constructs.get(c['task_id'], ((), False))[1], args.wrapper_share)

    renderer = TemplateRenderer(args.tokenizer)
    rows_out = ZstdWriter(out / 'rows.jsonl.zst')
    sft_episodes = []
    stats = collections.defaultdict(collections.Counter)
    families, domains = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    producers = collections.defaultdict(collections.Counter)

    def admit(c, turns):
        """Render one trajectory's calls; write its rows; keep its token rows for the chains."""
        source, rows, written = c['source'], [], []
        for index, turn in enumerate(turns):
            try:
                prompt, target = renderer.render(turn['messages'], turn['tools'], turn['target'])
            except Exception as error:
                stats[source]['render_error_calls'] += 1
                print(json.dumps({'event': 'render error', 'key': c['key'], 'call': index,
                                  'error': repr(error)[:300]}), flush=True)
                if c['source'] in ('rl-own', 'teacher'):
                    return False   # a verified episode is admitted whole or not at all
                continue
            row_id = turn.get('id') or hashlib.sha256(f"{c['key']}#{index}".encode()).hexdigest()[:24]
            row = {'id': row_id, 'invocation': turn.get('invocation_id') or '', 'index': index,
                   'prompt': np.asarray(prompt, np.int32), 'sample': np.asarray(target, np.int32)}
            context = reasoning_free_context(renderer, source, turn['target'], target)
            if context:
                row['weight'] = (np.arange(len(target)) >= context).astype(np.float32)
            rows.append(row)
            written.append({'id': row_id, 'episode': c['key'], 'source': source, 'run': c['run'],
                            'task_id': c['task_id'], 'family': c['family'], 'domain': c['domain'],
                            'teacher': c['teacher'], 'policy': c['policy'], 'call': index,
                            'invocation': turn.get('invocation_id'), 'messages': turn['messages'],
                            'tools': turn['tools'], 'target': turn['target'], 'prompt_tokens': len(prompt),
                            'target_tokens': len(target), 'loss_from': context,
                            'too_long': len(prompt) + len(target) > MAX_LEN})
        if not rows:
            return False
        for row in written:
            rows_out.write(row)
        funnel[source]['admitted'] += 1
        s = stats[source]
        s['calls'] += len(rows)
        s['target_tokens'] += sum(len(r['sample']) for r in rows)
        s['weighted_target_tokens'] += int(sum(float(r['weight'].sum()) if 'weight' in r else len(r['sample'])
                                               for r in rows))
        s['call_tokens'] += sum(len(r['prompt']) + len(r['sample']) for r in rows)
        s['too_long_calls'] += sum(len(r['prompt']) + len(r['sample']) > MAX_LEN for r in rows)
        families[source][c['family']] += 1
        domains[source][c['domain']] += 1
        producer = f"{source}:{c.get('teacher') or c.get('run') or ''}"
        producers[producer]['episodes'] += 1
        producers[producer]['target_tokens'] += sum(len(r['sample']) for r in rows)
        producers[producer]['weighted_target_tokens'] += int(sum(float(r['weight'].sum()) if 'weight' in r
                                                                 else len(r['sample']) for r in rows))
        producers[producer]['think_context_calls'] += sum('weight' in r for r in rows)
        producers[producer]['reasoning_free_calls'] += sum(not (t['target'].get('reasoning_content') or '').strip()
                                                           for t in turns)
        episode = {'task_id': c['task_id'], 'round': c['key'], 'rows': rows}
        s['forward_tokens'] += sum(len(chain['ids']) for chain in chains_of([episode]))
        sft_episodes.append(episode)
        return True

    for c in core:
        replayed = [json.loads(l) for l in (work / 'replay' / c['name'] / 'rollouts.jsonl').open() if l.strip()][0]
        if not admit(c, [{k: t.get(k) for k in ('messages', 'tools', 'target', 'invocation_id')} for t in replayed['turns']]):
            funnel[c['source']]['render_error'] += 1
    core_forward = sum(stats[s]['forward_tokens'] for s in ('rl-own', 'teacher'))

    # 2. October trajectories (no replayable task): compiled actions, held out, deduplicated against everything admitted,
    #    one trajectory per task by default, family caps, then a forward-token budget split by group weights
    october_budget, october_report = 0, None
    if args.october and args.october_share > 0:
        weights = dict((g, float(w)) for g, w in (spec.split('=') for spec in (args.october_group or DEFAULT_OCTOBER_GROUPS)))
        candidates = october_candidates(args, funnel, held, weights)
        rng.shuffle(candidates)
        unique = []
        for c in candidates:
            if dedup.seen(c['signature']):
                funnel[c['source']]['near_duplicate'] += 1
            else:
                unique.append(c)
        selected = capped(unique, rng, funnel, lambda task_id: args.october_max_per_task, args.october_family_share,
                          lambda c: 0)
        october_budget = core_forward * args.october_share / (1 - args.october_share)
        sizes = collections.Counter()
        for c in selected:
            sizes[c['group']] += c['size']
        rng.shuffle(selected)
        root = Path(args.october)
        scale = collections.defaultdict(lambda: 1.)        # actual forward tokens per estimated token, per group
        actual, estimated_done, admitted_keys = collections.Counter(), collections.Counter(), set()
        passes = []
        for attempt in range(args.october_passes):
            # budgets on the current size estimates (sizes still open, scaled), less what earlier passes used
            open_sizes = collections.Counter()
            for c in selected:
                if c['key'] not in admitted_keys:
                    open_sizes[c['group']] += c['size'] * scale[c['group']]
            budgets = water_fill({g: open_sizes[g] + actual[g] for g in weights if open_sizes[g] + actual[g] > 0},
                                 weights, october_budget)
            chosen, planned = {}, collections.Counter()
            for c in selected:
                g = c['group']
                if c['key'] in admitted_keys:
                    continue
                estimate = c['size'] * scale[g]
                if actual[g] + planned[g] + estimate <= budgets.get(g, 0):
                    planned[g] += estimate
                    chosen[c['key']] = c
            if not chosen:
                break
            # render the chosen trajectories, one group at a time (rows of a trajectory kept in file order)
            for group in sorted({c['group'] for c in chosen.values()}):
                pending = collections.defaultdict(list)
                for line in zstd_lines(root / f'{group}.jsonl.zst'):
                    row = json.loads(line)
                    key = f"october:{group}:{(row.get('provenance') or {}).get('trajectory') or row.get('task_id')}"
                    if key in chosen:
                        pending[key].append(row)
                for key, rows in pending.items():
                    c = chosen[key]
                    turns = [{'id': r['id'], 'messages': r['messages'], 'tools': r['tools'], 'target': r['target'],
                              'invocation_id': (r.get('provenance') or {}).get('invocation') or key} for r in rows]
                    c['run'] = group
                    before = stats[c['source']]['forward_tokens']
                    admitted_keys.add(key)
                    if not admit(c, turns):
                        funnel[c['source']]['render_error'] += 1
                        continue
                    actual[group] += stats[c['source']]['forward_tokens'] - before
                    estimated_done[group] += c['size']
                del pending
            for g in estimated_done:
                scale[g] = actual[g] / estimated_done[g] if estimated_done[g] else 1.
            passes.append({'chosen': len(chosen), 'actual_forward': dict(actual), 'budgets': {g: round(v) for g, v in budgets.items()}})
            if sum(actual.values()) >= 0.95 * october_budget:
                break
        for c in selected:
            if c['key'] not in admitted_keys:
                funnel[c['source']]['over_budget'] += 1
        october_report = {'weights': weights, 'budget_forward_tokens': round(october_budget),
                          'estimated_group_tokens': {g: round(v) for g, v in sizes.items()},
                          'actual_per_estimated_token': {g: round(v, 3) for g, v in scale.items()},
                          'passes': passes, 'chars_per_token_estimate': CHARS_PER_TOKEN}

    rows_out.close()
    chains = chains_of(sft_episodes)
    rng.shuffle(chains)
    with (out / 'chains.pkl.partial').open('wb') as handle:
        pickle.dump({'chains': chains, 'schema': 'natlang.sft-chains/1'}, handle, protocol=4)
    os.replace(out / 'chains.pkl.partial', out / 'chains.pkl')
    rated = [base_rate(e['task_id']) for e in sft_episodes if e['round'].split(':')[0] in ('rl', 'teacher')]
    rated = [p for p in rated if p is not None]
    forward = int(sum(len(c['ids']) for c in chains))
    report = {
        'schema': 'natlang.sft-sources-build/1', 'code': code_provenance(), 'argv': sys.argv,
        'inputs': {'episodes': {str(p): sha256_file(p) for p in episode_files(work)}, 'teacher_pools': args.teacher_pool,
                   'replay': str(work / 'replay.jsonl'), 'replay_sha256': sha256_file(work / 'replay.jsonl'),
                   'october': args.october, 'held_manifest': args.held_manifest,
                   'held_manifest_sha256': sha256_file(args.held_manifest), 'base_rates': args.base_rates,
                   'renderer': renderer.identity},
        'policy': {'max_per_task': args.max_per_task, 'rich_max_per_task': args.rich_max_per_task,
                   'held_constructs': HELD_CONSTRUCTS, 'wrapper_share': args.wrapper_share,
                   'cap_rule': 'round(1 + (cap - 1) * (1 - base pass rate)); cap = rich_max_per_task for a task '
                               'declaring a held construct, else max_per_task',
                   'max_family_share': args.max_family_share, 'dup_threshold': args.dup_threshold, 'seed': args.seed,
                   'october_share': args.october_share, 'october_max_per_task': args.october_max_per_task,
                   'october_family_share': args.october_family_share,
                   'loss': 'per-call rows: prompt (system, task, history, tool output) zero weight; the assistant reply '
                           '(reasoning, text, tool calls, end token) weight 1; a teacher reply without reasoning: the '
                           'empty think block (through </think>) zero weight, the action after it weight 1 (row weight)'},
        'october': october_report,
        'funnel': {k: dict(v) for k, v in funnel.items()},
        'stats': {k: dict(v) for k, v in stats.items()},
        'episodes': len(sft_episodes), 'tasks': len({e['task_id'] for e in sft_episodes}),
        'chains': len(chains), 'forward_tokens': forward,
        'target_tokens': int(sum(len(r['sample']) for e in sft_episodes for r in e['rows'])),
        'epoch_hours_at_5500_tokens_per_second': round(forward / 5500 / 3600, 2),
        'producers': {k: {**dict(v), 'weighted_target_token_share': round(v['weighted_target_tokens'] / max(1, sum(
            p['weighted_target_tokens'] for p in producers.values())), 4)} for k, v in sorted(producers.items())},
        'families': {k: len(v) for k, v in families.items()},
        'family_counts': {k: dict(v.most_common()) for k, v in families.items()},
        'domain_counts': {k: dict(v.most_common()) for k, v in domains.items()},
        'core_base_rate': {'measured_episodes': len(rated),
                           'mean': round(float(np.mean(rated)), 3) if rated else None},
    }
    (out / 'build.json').write_text(json.dumps(report, indent=1) + '\n')
    (out / 'cases.jsonl').write_text(''.join(json.dumps(tasks[t]) + '\n' for t in sorted(tasks)
                                             if any(e['task_id'] == t for e in sft_episodes)))
    print(json.dumps({k: report[k] for k in ('funnel', 'stats', 'episodes', 'tasks', 'chains', 'forward_tokens',
                                             'target_tokens')}, indent=1))


def neuralese_parts(messages):
    """Number of native_state parts (imported Neuralese state) in a call's messages."""
    return sum(isinstance(m.get('content'), list) and sum(isinstance(part, dict) and part.get('type') == 'native_state'
                                                          for part in m['content']) for m in messages)


def surface_key(turn):
    """(system prompt, tool schemas) of a call: what the runtime surface renders around the task."""
    system = ''.join(m['content'] if isinstance(m['content'], str) else json.dumps(m['content'])
                     for m in turn['messages'] if m.get('role') == 'system')
    return (hashlib.sha256(system.encode()).hexdigest()[:16],
            hashlib.sha256(json.dumps(turn.get('tools'), sort_keys=True).encode()).hexdigest()[:16])


def neuralese_text(args):
    """Text rows of the verified successes the build skips for their Neuralese state (`has_native_state`): every call
    whose messages hold no imported native state, i.e. the Neuralese producer invocations (the prompted calls whose
    returned text becomes the state) and the consumer's calls before its first import. A call after an import needs the
    imported state as context (no text rendering is exact) and is counted, not written. No replay is possible for these
    episodes (their states lived only in the actor's store), so validity is: oracle success at collection, the held-v3/v2
    closure, and each call's system prompt and tool schemas equal to ones the current runtime rendered in an exact replay
    (calls with an unseen surface are dropped). Writes OUT/chains.pkl (`exp0_sft.py train --chains`, the build's format),
    OUT/rows.jsonl.zst and OUT/build.json."""
    import random
    import numpy as np
    import pickle
    from natlang_neuralese.common.code_provenance import code_provenance
    from natlang_neuralese.common.jsonio import read_jsonl
    from natlang_neuralese.rl.held_eval import exclusion_ids
    from exp0_sft import MAX_LEN, chains_of
    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    held = held_closure(args.held_manifest)
    funnel = collections.Counter()
    # the current runtime's surfaces: every call of every exact replay
    surfaces = set()
    for line in (work / 'replay.jsonl').open():
        record = json.loads(line) if line.strip() else {}
        if record.get('exact'):
            path = work / 'replay' / record['name'] / 'rollouts.jsonl'
            for row in read_jsonl(path):
                surfaces.update(surface_key(t) for t in row.get('turns') or ())
    episodes = [e for e in all_episodes(work) if has_native_state(e)]
    funnel['native_state_episodes'] = len(episodes)
    excluded = exclusion_ids(args.held_manifest, [e['task'] for e in episodes])
    kept = []
    for e in episodes:
        task = e['task']
        groups = list(task.get('source_groups') or ()) + [(task.get('curriculum') or {}).get('split_group') or task['id']]
        if e['task_id'] in excluded or held_hit(e['task_id'], task.get('source_ids') or (), groups, held):
            funnel['held_excluded'] += 1
            continue
        kept.append(e)
    rng.shuffle(kept)
    per_task = collections.Counter()
    renderer = TemplateRenderer(args.tokenizer)
    rows_out = ZstdWriter(out / 'rows.jsonl.zst')
    sft_episodes, tasks = [], {}
    stats = collections.defaultdict(collections.Counter)
    for e in kept:
        if per_task[e['task_id']] >= args.max_per_task:
            funnel['over_task_cap'] += 1
            continue
        generator = 'bgkit' if e['task_id'].startswith('ext:bgkit:') else 'other'
        root = e['turns'][0].get('invocation_id')
        rows, written = [], []
        for index, turn in enumerate(e['turns']):
            role = 'root' if turn.get('invocation_id') == root else 'child'
            s = stats[f'{generator}:{role}']
            if neuralese_parts(turn['messages']):
                s['calls_after_import'] += 1
                s['target_tokens_after_import'] += turn.get('completion_tokens') or 0
                continue
            if surface_key(turn) not in surfaces:
                s['calls_unseen_surface'] += 1
                continue
            try:
                prompt, target = renderer.render(turn['messages'], turn['tools'], turn['target'])
            except Exception as error:
                s['render_error'] += 1
                print(json.dumps({'event': 'render error', 'key': e['key'], 'call': index, 'error': repr(error)[:300]}),
                      flush=True)
                continue
            row_id = hashlib.sha256(f"{e['key']}#{index}".encode()).hexdigest()[:24]
            rows.append({'id': row_id, 'invocation': turn.get('invocation_id') or '', 'index': index,
                         'prompt': np.asarray(prompt, np.int32), 'sample': np.asarray(target, np.int32)})
            written.append({'id': row_id, 'episode': e['key'], 'source': 'rl-own-neuralese', 'run': e.get('run'),
                            'task_id': e['task_id'], 'family': e.get('family') or e['task'].get('family'),
                            'generator': generator, 'role': role, 'policy': e.get('policy'), 'call': index,
                            'invocation': turn.get('invocation_id'), 'messages': turn['messages'],
                            'tools': turn['tools'], 'target': turn['target'], 'prompt_tokens': len(prompt),
                            'target_tokens': len(target), 'loss_from': 0,
                            'too_long': len(prompt) + len(target) > MAX_LEN})
            s['calls'] += 1
            s['target_tokens'] += len(target)
            s['call_tokens'] += len(prompt) + len(target)
            s['too_long_calls'] += len(prompt) + len(target) > MAX_LEN
        if not rows:
            funnel['no_text_calls'] += 1
            continue
        per_task[e['task_id']] += 1
        funnel['admitted'] += 1
        funnel[f'admitted_{generator}'] += 1
        tasks[e['task_id']] = e['task']
        for row in written:
            rows_out.write(row)
        episode = {'task_id': e['task_id'], 'round': f"rl-neuralese:{e['key']}", 'rows': rows}
        stats[generator]['forward_tokens'] += sum(len(c['ids']) for c in chains_of([episode]))
        stats[generator]['episodes'] += 1
        sft_episodes.append(episode)
    rows_out.close()
    chains = chains_of(sft_episodes)
    rng.shuffle(chains)
    with (out / 'chains.pkl.partial').open('wb') as handle:
        pickle.dump({'chains': chains, 'schema': 'natlang.sft-chains/1'}, handle, protocol=4)
    os.replace(out / 'chains.pkl.partial', out / 'chains.pkl')
    report = {
        'schema': 'natlang.sft-sources-neuralese-text/1', 'code': code_provenance(), 'argv': sys.argv,
        'inputs': {'episodes': {str(p): sha256_file(p) for p in episode_files(work)},
                   'replay_sha256': sha256_file(work / 'replay.jsonl'), 'held_manifest': args.held_manifest,
                   'held_manifest_sha256': sha256_file(args.held_manifest), 'renderer': renderer.identity},
        'policy': {'max_per_task': args.max_per_task, 'seed': args.seed, 'current_surfaces': len(surfaces),
                   'calls': 'no native_state part in the messages (producer invocations; consumer calls before the '
                            'first import); system prompt and tools equal to an exact replay\'s',
                   'loss': 'per-call rows: prompt zero weight; the own sampled reply weight 1'},
        'funnel': dict(funnel), 'stats': {k: dict(v) for k, v in sorted(stats.items())},
        'episodes': len(sft_episodes), 'tasks': len(tasks), 'chains': len(chains),
        'calls': int(sum(len(e['rows']) for e in sft_episodes)),
        'forward_tokens': int(sum(len(c['ids']) for c in chains)),
        'target_tokens': int(sum(len(r['sample']) for e in sft_episodes for r in e['rows'])),
    }
    (out / 'build.json').write_text(json.dumps(report, indent=1) + '\n')
    (out / 'cases.jsonl').write_text(''.join(json.dumps(tasks[t]) + '\n' for t in sorted(tasks)))
    print(json.dumps({k: report[k] for k in ('funnel', 'stats', 'episodes', 'tasks', 'chains', 'calls',
                                             'forward_tokens', 'target_tokens')}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--work', required=True)
    e.add_argument('--rl', action='append', default=[], metavar='RUN_DIR:ARCHIVE_DIR')
    e.add_argument('--teacher', action='append', default=[], metavar='POOL_DIR')
    e.add_argument('--out-name', default='episodes', help='WORK/NAME.jsonl.zst (build reads every episodes*.jsonl.zst)')
    r = sub.add_parser('replay')
    r.add_argument('--work', required=True)
    r.add_argument('--runtime-repo', help='checkout (a pin) whose runtime, collector and oracle replay (default: this one)')
    r.add_argument('--workers', type=int, default=6)
    r.add_argument('--limit', type=int, default=0)
    r.add_argument('--case-seconds', type=int, default=900)
    r.add_argument('--max-model-requests', type=int, default=None)
    c = sub.add_parser('check-render')
    c.add_argument('--work', required=True)
    c.add_argument('--tokenizer', required=True)
    c.add_argument('--turns', type=int, default=400)
    b = sub.add_parser('build')
    b.add_argument('--work', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--tokenizer', required=True)
    b.add_argument('--held-manifest', default=resolve_str(*OPS, 'held-v3', 'manifest.json'))
    b.add_argument('--base-rates', default=resolve_str(*OPS, 'exp1-rft', 'base-pass-rates.json'))
    b.add_argument('--max-per-task', type=int, default=4,
                   help='episodes of a task the start policy never solves; a task it always solves keeps 1')
    b.add_argument('--rich-max-per-task', type=int, default=8,
                   help='the per-task cap for tasks whose program declares a held construct (HELD_CONSTRUCTS)')
    b.add_argument('--wrapper-share', type=float, default=0.15,
                   help='dataset-wrapper episodes kept to at most this share of the GRPO-pool selection (1 = no cap)')
    b.add_argument('--max-family-share', type=float, default=0.05)
    b.add_argument('--dup-threshold', type=float, default=0.8)
    b.add_argument('--seed', type=int, default=20261117)
    b.add_argument('--teacher-pool', action='append', default=None,
                   help='teacher pools whose current manifests the extraction must cover (default: teacher-gen pool-v1)')
    b.add_argument('--allow-stale-teacher', action='store_true')
    b.add_argument('--october', help='directory of October candidate rows (GROUP.jsonl.zst, eval-check/GROUP.jsonl)')
    b.add_argument('--october-group', action='append', metavar='GROUP=WEIGHT',
                   help='October groups taken and their budget weights (default: %s)' % ', '.join(DEFAULT_OCTOBER_GROUPS))
    b.add_argument('--october-share', type=float, default=0.4,
                   help='October share of the forward tokens (prompt + reply of every call)')
    b.add_argument('--october-max-per-task', type=int, default=1)
    b.add_argument('--october-passes', type=int, default=4,
                   help='selection passes: each renders, measures actual forward tokens and refills toward the budget')
    b.add_argument('--october-family-share', type=float, default=0.05)
    n = sub.add_parser('neuralese-text', help='text rows (producers, calls before an import) of Neuralese successes')
    n.add_argument('--work', required=True)
    n.add_argument('--out', required=True)
    n.add_argument('--tokenizer', required=True)
    n.add_argument('--held-manifest', default=resolve_str(*OPS, 'held-v3', 'manifest.json'))
    n.add_argument('--max-per-task', type=int, default=4)
    n.add_argument('--seed', type=int, default=20261117)
    args = p.parse_args()
    if args.command == 'build' and args.teacher_pool is None:
        args.teacher_pool = [resolve_str(*OPS, 'teacher-gen-20261115', 'pool-v1')]
    {'extract': extract, 'replay': replay, 'check-render': check_render, 'build': build,
     'neuralese-text': neuralese_text}[args.command](args)


if __name__ == '__main__':
    main()
