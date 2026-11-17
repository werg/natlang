"""The memetic improvement workspace: the crisp service an "improve program P" episode acts through
(plans/neuralese/RESTORE_MEMETIC_JOLT_HINDSIGHT_20261117.md section 3; first built 2026-11-03,
MEMETIC_META_LEARNING_REVIEW_20261103.md).

One episode (``natlang.memetic-episode/1`` spec, written by ``rl.memetic_tasks``, copied per GRPO group by
``grpo.py``) gives the supervisor program (``applications/memetic-improver``) a target program P, P's support
invocations with the current policy's outcomes on them, and typed operations. The supervisor's decisions (what failed,
what to change, whether a change helped, when to stop) are its own natural-language turns; this module is mechanism:

- evidence: support attempts (outcome class, failed checks, returned value, errors, steps), never gold, never query
  worlds;
- typed changes: guidance (program, one function or an input class), skills in a function's folder, instruction
  find/replace patches, signature edits, tool withholding, reverts; each is checked before it gets an id (exact find,
  frontmatter parse, natlang loader check);
- evaluation: the program with a change set runs on the support worlds by a nested collection on the episode's own
  actor (``task_rollouts.prepare`` + ``scripts/run_bonsai_queue.py`` + ``task_rollouts.score``, as the GRPO collector
  runs episodes), paired seeds, cached by change-set digest and shared by the group's samples;
- commit: the supervisor's final change set. ``score`` (host only, never offered to the model) evaluates the committed
  set on the private query worlds and returns the graded outcome ``rl.task_rollouts`` reads.

Guidance reaches the evaluated program through the runtime's program-guidance channel: its text is written as a native
state on the actor (``/v1/neuralese/native-states/write-text``) and bound as ``natlang.memetic-program-guidance/1``.

Served as a JSON-lines world process (``ts-host/src/teacher/world-bridge.ts`` kind ``memetic``):
``python -m natlang_neuralese.rl.memetic_workspace serve``.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from ..common.hashing import canonical_json_sha256_hex
from ..common.jsonio import write_json_durable

SPEC_SCHEMA = 'natlang.memetic-episode/1'
OUTCOME_SCHEMA = 'natlang.memetic-outcome/1'
CHANGE_KINDS = ('guidance', 'skill', 'instructions', 'signature', 'withhold', 'revert')
# Tools a function's frontmatter may withhold from its calls (ts-host runtime/loader.ts `withhold`).
WITHHOLDABLE = ('edit_code', 'diff_code', 'list_files', 'search_files', 'read_file', 'write_file', 'edit_file',
                'diff_files', 'bash', 'python', 'delegate', 'editor')
SCOPE_KINDS = ('program', 'function', 'inputs')
MAX_TEXT = 4000
REPO = Path(__file__).resolve().parents[4]
STATE_ID = re.compile(r'nzs1_[a-z2-7]{52}\Z')
IDENTITY = ('checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture')

DECLARATION = '''declare const workspace: {
  /** The failed support attempts of the program: id, outcome, what failed, the returned value, errors. */
  failures(): Promise<Attempt[]>;
  /** All support attempts, passed and failed. */
  attempts(): Promise<Attempt[]>;
  /** One attempt step by step: the call, tool, code, reasoning and output of each step, clipped. */
  attempt(id: string): Promise<AttemptStep[]>;
  /** The program's file paths, or the text of one file. */
  source(path?: string): Promise<string[] | string>;
  /** Check one change and keep it under a new id; problems explain a change that cannot apply. */
  propose(change: Change): Promise<{ id: string } | { problems: string[] }>;
  /** Run the program with exactly these changes on its support invocations. */
  evaluate(ids: string[]): Promise<Evaluation>;
  /** The support success rate with each listed change left out, one at a time. */
  ablate(ids: string[]): Promise<{ id: string; without: number }[]>;
  /** Commit the final change set (possibly empty) with one sentence on what it fixes. Call once, at the end. */
  commit(ids: string[], reason: string): Promise<{ committed: string[] }>;
  /** Record a defect of the harness or of the task's check (it is not a program change and earns nothing). */
  reportDefect(text: string): Promise<{ recorded: true }>;
  /** Evaluations left. */
  budget(): Promise<{ evaluations: number }>;
  /** How the unchanged program did on its support invocations. */
  baseline(): Promise<{ attempts: number; success: number }>;
};
type Attempt = { id: string; world: string; outcome: string; failedChecks: string[]; returned: string; errors: string[] };
type AttemptStep = { call: string; turn: number; tool: string; status: string; code?: string; arguments?: string; reasoning?: string; output: string };
type Scope = { kind: "program" } | { kind: "function"; name: string } | { kind: "inputs"; when: string };
type Change =
  | { kind: "guidance"; scope: Scope; text: string }
  | { kind: "skill"; function: string; name: string; description: string; body: string; action: "create" | "revise" | "retire" }
  | { kind: "instructions"; file: string; find: string; replace: string }
  | { kind: "signature"; file: string; field: "args" | "returns"; name?: string; type: string }
  | { kind: "withhold"; file: string; tools: string[] }
  | { kind: "revert"; id: string };
type Evaluation = { ids: string[]; success: number; baseline: number; attempts: Attempt[]; evaluationsLeft: number };'''

METHODS = ('failures', 'attempts', 'attempt', 'source', 'propose', 'evaluate', 'ablate', 'commit', 'reportDefect',
           'budget', 'baseline')


def _clip(text, limit):
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + f' <<cut off: {len(text) - limit} more characters>>'


@contextlib.contextmanager
def locked(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.lock', 'a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# ---- evidence --------------------------------------------------------------------------------------------------------
def attempt_digest(result, *, attempt_id, world):
    """Crisp evidence of one support attempt: what happened, never the expected answer."""
    outcome = (result or {}).get('outcome') or {}
    checks = outcome.get('checks') or {}
    failed = sorted({key for key, value in checks.items() if value is False} |
                    {str(reason) for reason in outcome.get('rejection_reasons') or []})
    errors = [_clip(str(item.get('message', item)), 240) for item in outcome.get('scope_failures') or []][:3]
    errors += [_clip(str(d), 240) for a in outcome.get('action_ledger') or [] if isinstance(a, dict)
               for d in (a.get('diagnostics') or [])][-3:]
    return {'id': attempt_id, 'world': world, 'outcome': str(outcome.get('status') or 'unknown'),
            'failedChecks': failed, 'returned': _clip(outcome.get('value'), 400), 'errors': errors}


def attempt_steps(result, limit=60):
    """One attempt step by step, every call of it (root first): the shared transcript reading of
    ``rl/hindsight.py call_steps`` over the policy's own turns (``task_rollouts.rollout_turns``)."""
    from .hindsight import trajectory_value
    from .task_rollouts import rollout_turns
    turns = rollout_turns(result)
    if not turns:
        return []
    root = trajectory_value({'turns': turns})
    steps = []
    for call in [root, *root['children']]:
        for step in call['steps']:
            item = {'call': call['call'] if call is not root else 'root', 'turn': step['turn'], 'tool': step['tool'],
                    'status': step['status'], 'output': _clip(step.get('output') or '', 500)}
            if step.get('code'):
                item['code'] = _clip(step['code'], 700)
            if step.get('arguments'):
                item['arguments'] = _clip(step['arguments'], 300)
            if step.get('reasoning'):
                item['reasoning'] = _clip(step['reasoning'], 300)
            steps.append(item)
    return steps[:limit]


_OUTLINE_SKIPS = {'read_code', 'compact_history', 'transcript.search', 'reply'}
_ERROR_OUTPUT = re.compile(r'\b\w*Error\b|\binvalid-|is not defined|no-such-|\berror:', re.IGNORECASE)


def mechanical_guidance(successes, limit=1500):
    """The default crisp improvement (the anchor's second arm): the action outline of a verified successful support
    attempt, stated as an example for other invocations; the attempt that needed the fewest actions, without looking
    around and without errored actions."""
    outlines = []
    for steps in successes:
        kept = [step for step in steps if step['call'] == 'root' and step['tool'] not in _OUTLINE_SKIPS
                and step['status'] != 'error' and not _ERROR_OUTPUT.search(step.get('output') or '')]
        if kept:
            outlines.append(kept if len(kept) <= 12 else kept[:11] + kept[-1:])
    if not outlines:
        return None
    lines = []
    for step in min(outlines, key=len):
        what = step.get('code') or step.get('arguments') or ''
        lines.append(f"{len(lines) + 1}. {step['tool']}" + (f": {what}" if what else ''))
    text = 'A verified successful run of this program on another invocation took these actions:\n' + '\n'.join(lines)
    return text[:limit]


# ---- changes ---------------------------------------------------------------------------------------------------------
def _placeholder(text):
    """A stand-in such as "..." or "TODO" rather than written text."""
    words = re.findall(r"[^\W\d_]{2,}", str(text).lower())
    if len(words) < 2 and set(words) <= {'tbd', 'todo', 'tba', 'body', 'text', 'skill', 'guidance', 'placeholder'}:
        return True
    return len(words) < 12 and re.search(r"\b(to be (filled|added|written|determined|defined|completed)|fill(ed)? in"
                                          r"( later)?|placeholder|tbd|todo|details later)\b", ' '.join(words)) is not None


def frontmatter(text):
    """(meta dict, body) of a .nl file; raises on a malformed header."""
    import yaml
    match = re.match(r'---\n(.*?)\n---\n?(.*)\Z', text, re.S)
    if not match:
        raise ValueError('a .nl file starts with a --- frontmatter block')
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        raise ValueError('frontmatter is a mapping')
    return meta, match.group(2)


def render_nl(meta, body):
    import yaml
    return '---\n' + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).rstrip('\n') + '\n---\n' + body


def compose_guidance(changes):
    """Scoped guidance entries ``[{function: None | name, text}]``: one program-wide text, and one text per scoped
    function (the program-wide part first, then the function's own). Input-class guidance becomes a skill."""
    program = [change['text'].strip() for change in changes if change['scope']['kind'] == 'program']
    functions = {}
    for change in changes:
        if change['scope']['kind'] == 'function':
            functions.setdefault(change['scope']['name'], []).append(change['text'].strip())
    entries = [{'function': None, 'text': '\n\n'.join(program)}] if program else []
    entries += [{'function': name, 'text': '\n\n'.join(program + parts)} for name, parts in sorted(functions.items())]
    return entries


def input_skills(changes, root):
    """Input-class guidance as skills of the root function: the listing shows ``when`` as the description and the
    called model loads the text only for inputs it fits (progressive disclosure)."""
    skills = []
    for change in changes:
        if change['scope']['kind'] == 'inputs':
            when = change['scope']['when']
            name = 'for-' + (re.sub(r'[^a-z0-9]+', '-', when.lower()).strip('-')[:32].strip('-') or 'inputs')
            skills.append({'kind': 'skill', 'function': Path(root).stem, 'action': 'create',
                           'name': name + '-' + hashlib.sha256(when.encode()).hexdigest()[:6],
                           'description': 'Use for inputs where ' + when.strip(), 'body': change['text'].strip()})
    return skills


def skill_folder(files, function):
    """The companion folder of the function that reads the skill: ``f.nl``'s skills live in ``f/skills/``."""
    matches = [path for path in files if path.endswith('.nl') and Path(path).stem == function]
    if len(matches) != 1:
        raise ValueError(f'skill: no unique function named {function!r} in the program')
    return matches[0][:-3] + '/'


def apply_changes(case, changes):
    """The task row with the file-level changes applied (instructions, signatures, withholding, skills). Guidance
    travels separately (program-guidance binding); inputs and checks are never touched."""
    case = copy.deepcopy(case)
    files = case['semantics']['files']
    for change in changes:
        kind = change['kind']
        if kind == 'instructions':
            meta, body = frontmatter(files[change['file']])
            if body.count(change['find']) != 1:
                raise ValueError(f"instructions patch: the text to replace must occur exactly once in {change['file']}")
            files[change['file']] = render_nl(meta, body.replace(change['find'], change['replace'], 1))
        elif kind == 'signature':
            meta, body = frontmatter(files[change['file']])
            if change['field'] == 'returns':
                meta['returns'] = change['type']
            else:
                if not change.get('name'):
                    raise ValueError('a signature args change names the argument')
                meta['args'] = {**(meta.get('args') or {}), change['name']: change['type']}
            files[change['file']] = render_nl(meta, body)
        elif kind == 'withhold':
            meta, body = frontmatter(files[change['file']])
            meta['withhold'] = sorted(set(meta.get('withhold') or []) | set(change['tools']))
            files[change['file']] = render_nl(meta, body)
        elif kind == 'skill':
            path = f"{skill_folder(files, change['function'])}skills/{change['name']}/SKILL.md"
            if change['action'] == 'retire':
                files.pop(path, None)
            else:
                files[path] = ('---\nname: ' + change['name'] + '\ndescription: ' + json.dumps(change['description'])
                               + '\n---\n' + change['body'].strip() + '\n')
    return case


def check_program(case, root=REPO):
    """Load the changed program with the natlang loader (the compiler's own check). Returns problems."""
    files = case['semantics']['files']
    script = ("import('" + (root / 'ts-host/dist/index.js').as_uri() + "').then(m=>{const files=JSON.parse("
              "process.argv[1]);const problems=[];for(const p of Object.keys(files).filter(p=>p.endsWith('.nl'))){"
              "try{m.loadVirtualNatlang(files,p)}catch(e){problems.push(p+': '+String(e.message||e).split('\\n')[0])}}"
              "process.stdout.write(JSON.stringify(problems))})")
    try:
        output = subprocess.run([os.environ.get('NATLANG_NODE', 'node'), '--input-type=module', '-e', script,
                                 json.dumps(files)], capture_output=True, text=True, timeout=120, check=True).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return [f'the natlang loader could not check the program: {str(error)[:200]}']
    return json.loads(output or '[]')


def leaked_answers(candidate, query_cases):
    """A candidate whose texts contain a query world's expected answer verbatim is refused."""
    blob = '\n'.join([entry['text'] for entry in candidate.get('guidance') or []] +
                     [json.dumps(change, ensure_ascii=False) for change in candidate['files']])
    for case in query_cases:
        expected = (case.get('semantics') or {}).get('expected')
        values = [expected] if isinstance(expected, str) else [
            v for v in (expected.values() if isinstance(expected, dict) else []) if isinstance(v, str)]
        if any(len(value) >= 6 and value in blob for value in values):
            return True
    return False


def memetic_reward(query_success, baselines, *, evaluations, budget, cost_weight, committed):
    """(quality, anchor, gain, cost) of one episode. quality: the committed set's query success minus a bounded cost for
    the evaluation budget it used (at most ``cost_weight``) and that weight again when it never committed. anchor: the
    best trivial improvement's query success (no change; the mechanical default); the learner centers on
    max(group mean, anchor) (grpo.py ``episode_advantages``), so a candidate that does not beat the trivial thing never
    earns a positive advantage."""
    available = [item['success'] for item in baselines.values() if item and item.get('success') is not None]
    anchor = max(available) if available else 0.
    cost = cost_weight * min(1., evaluations / max(1, budget['evaluations']) / 2) + (0. if committed else cost_weight)
    quality = max(0., min(1., query_success - cost))
    return quality, anchor, query_success - anchor, cost


# ---- actor: native guidance and nested collections -------------------------------------------------------------------
def request_json(server, path, value=None, timeout=900):
    request = urllib.request.Request(server.rstrip('/') + path,
                                     data=None if value is None else json.dumps(value).encode(),
                                     headers={'content-type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def native_identity(info):
    """The native-state identity the actor writes and imports states under (an in-place GRPO update keeps the start
    checkpoint's, ``sglang_actor.rebind_in_place``)."""
    state = info.get('native_state') or {}
    if state.get('enabled') is not True or any(not isinstance(state.get(key), str) or not state[key] for key in IDENTITY):
        raise ValueError('memetic guidance needs an actor serving native state')
    return {key: state[key] for key in IDENTITY}


def guidance_binding(server, out, program_id, text, function=None):
    """``natlang.memetic-program-guidance/1`` for ``text``: written as a native state by the actor's producer, its bytes
    archived under ``out/states`` (the runtime imports the exact stored state)."""
    identity = native_identity(request_json(server, '/v1/neuralese/info'))
    written = request_json(server, '/v1/neuralese/native-states/write-text', {'text': text, 'type': 'Neuralese<ProgramGuidance>'})
    state_id = written.get('id')
    if not isinstance(state_id, str) or not STATE_ID.fullmatch(state_id):
        raise ValueError('the actor returned no exact native state id for the guidance')
    path = Path(out) / 'states' / f'{state_id}.safetensors'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(server.rstrip('/') + '/v1/neuralese/native-states/' + state_id, timeout=900) as response:
            raw = response.read()
        partial = path.with_suffix('.partial')
        partial.write_bytes(raw)
        partial.replace(path)
    raw = path.read_bytes()
    binding = {'schema': 'natlang.memetic-program-guidance/1', 'program_id': program_id, 'student': identity,
               'text': text, 'native_state_id': state_id,
               'native_state_artifact': {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                                         'bytes': len(raw)},
               'program_guidance': state_id}
    if function is not None:
        binding['function'] = function
    return binding


def collection_evaluator(spec):
    """``evaluate(cases, guidance, *, name, samples)`` for this episode: a nested collection of the changed program on
    the episode's actor, the GRPO collector's own episode command (``scripts/run_bonsai_queue.py`` per queue entry with
    the run's sampling controls), scored by ``task_rollouts.score``. Results are cached by name under a lock, so the
    samples of one group share every identical evaluation."""
    from ..common.jsonio import read_jsonl
    from . import task_rollouts
    settings, endpoint = spec['eval'], spec['endpoint']
    runtime = spec.get('runtime') or str(REPO / 'ts-host')

    def run_entry(entry, directory):
        workers = directory / 'workers'
        workers.mkdir(parents=True, exist_ok=True)
        queue_file, journal = workers / f"{entry['key']}.queue.jsonl", workers / f"{entry['key']}.journal.jsonl"
        queue_file.write_text(json.dumps(entry, sort_keys=True) + '\n')
        command = [sys.executable, str(REPO / 'scripts/run_bonsai_queue.py'), str(queue_file), str(journal),
                   '--runtime', runtime, '--server', endpoint, '--chat-completions-url', endpoint + '/v1/chat/completions',
                   '--model-id', 'natlang-rl', '--model-concurrency', '1', '--neuralese-server']
        if spec.get('sampling_config'):
            command += ['--chat-request-config', spec['sampling_config']]
        with (workers / f"{entry['key']}.log").open('a') as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False)

    def evaluate(cases, guidance, *, name, samples=None):
        out = Path(spec['out']) / 'evals' / name
        with locked(out):
            if (out / 'summary.json').exists():
                return json.loads((out / 'summary.json').read_text())
            out.mkdir(parents=True, exist_ok=True)
            (out / 'cases.jsonl').write_text(''.join(json.dumps(case, ensure_ascii=False) + '\n' for case in cases))
            collection = out / 'collection'
            group = max(2, samples or settings['samples'])
            task_rollouts.prepare(SimpleNamespace(
                cases=out / 'cases.jsonl', out=collection, rows=len(cases), offset=0, group=group, temperature=1.,
                case_seconds=settings['case_seconds'], max_model_requests=settings['max_model_requests'],
                max_turns=settings['max_turns'], seed_base=settings['seed_base'], root_mode='student',
                match_seeds_from=None, policy_checkpoint_sha256=None))
            entries = list(read_jsonl(collection / 'queue.jsonl'))
            binding = None
            if guidance:
                bindings = [guidance_binding(endpoint, out, spec['program']['id'], entry['text'], entry['function'])
                            for entry in guidance]
                binding = bindings[0] if len(bindings) == 1 and 'function' not in bindings[0] else bindings
                write_json_durable(out / 'program-guidance.json', binding)
            for entry in entries:
                entry.update(prompt_rendering='full', context_tokens=settings.get('context_tokens', 131072),
                             rl_endpoint=endpoint)
                if binding is not None:
                    entry['program_guidance_file'] = str(out / 'program-guidance.json')
            (collection / 'queue.jsonl').write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in entries))
            pending, lock = list(entries), threading.Lock()

            def worker():
                while True:
                    with lock:
                        if not pending:
                            return
                        entry = pending.pop(0)
                    run_entry(entry, collection)
            threads = [threading.Thread(target=worker) for _ in range(max(1, min(settings['workers'], len(entries))))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            with (collection / 'journal.jsonl').open('w') as merged:
                for path in sorted((collection / 'workers').glob('*.journal.jsonl')):
                    merged.write(path.read_text())
            task_rollouts.score(collection, None, neuralese_bonus=0., efficiency_weight=0.)
            results = []
            for reward in read_jsonl(collection / 'rewards.jsonl'):
                entry = next(e for e in entries if e['group'] == reward['group'] and e['sample'] == reward['sample'])
                path = Path(entry['output'])
                result = (list(read_jsonl(path)) or [None])[0] if path.exists() else None
                if result is not None:
                    result.setdefault('outcome', {})['status'] = reward['outcome']
                results.append((cases[reward['group']]['id'], result, reward['reward']))
            scored = [reward for _, _, reward in results if reward is not None]
            summary = {'success': sum(scored) / len(scored) if scored else 0., 'scored': len(scored),
                       'rollouts': len(results), 'results': [(world, result) for world, result, _ in results],
                       'dir': str(out), 'samples': group}
            write_json_durable(out / 'summary.json', summary)
            return summary
    return evaluate


# ---- preparation -------------------------------------------------------------------------------------------------------
def prepare(spec, evaluate):
    """Evidence and baselines of one episode group under the current policy, computed once by whichever sample loads
    first (the others wait on the lock and read it): the unchanged program on the support worlds (admission: success
    within [min_success, max_success], resolved), then on the query worlds with no change and with the mechanical
    default guidance (the anchor's two arms). Unscored support runs are no measurement."""
    path = Path(spec['out']) / 'prepared.json'
    with locked(path):
        if path.exists():
            return json.loads(path.read_text())
        settings = spec['settings']
        support = evaluate(spec['support'], None, name='baseline-support')
        attempts = [{'digest': attempt_digest(result, attempt_id=f'a{i + 1}', world=world), 'steps': attempt_steps(result)}
                    for i, (world, result) in enumerate(support['results'])]
        resolved = support['scored'] / max(1, support['rollouts'])
        admitted = resolved >= 0.5 and settings['min_success'] <= support['success'] <= settings['max_success']
        prepared = {'schema': 'natlang.memetic-prepared/1', 'support_success': support['success'],
                    'resolved_fraction': round(resolved, 3), 'attempts': attempts, 'admitted': admitted,
                    'baselines': {'none': None, 'default': None}}
        if admitted:
            none = evaluate(spec['query'], None, name='baseline-query-none', samples=settings['query_samples'])
            default_text = mechanical_guidance([item['steps'] for item in attempts
                                                if item['digest']['outcome'] == 'task_success'])
            default = (evaluate(spec['query'], [{'function': None, 'text': default_text}], name='baseline-query-default',
                                samples=settings['query_samples']) if default_text else None)
            prepared['baselines'] = {'none': {'success': none['success'], 'dir': none['dir']},
                                     'default': None if default is None else {
                                         'success': default['success'], 'text': default_text, 'dir': default['dir']}}
        write_json_durable(path, prepared)
        return prepared


# ---- the workspace -----------------------------------------------------------------------------------------------------
def run_directory(run):
    """A filesystem-safe, collision-free directory name for one rollout's run id."""
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', str(run))[:80]
    return f"{safe}-{hashlib.sha256(str(run).encode()).hexdigest()[:12]}"


class Workspace:
    """One rollout's workspace. ``evaluator(cases, guidance, *, name, samples=None)`` returns ``{'success', 'scored',
    'rollouts', 'results': [(world, result)], 'dir'}``; injected so tests run without a model."""

    def __init__(self, spec, evaluator, *, check=check_program, run=None, prepare_fn=prepare):
        if spec.get('schema') != SPEC_SCHEMA:
            raise ValueError(f'the memetic workspace needs a {SPEC_SCHEMA} spec')
        self.spec, self.evaluate_cases, self.check = spec, evaluator, check
        self.out = Path(spec['out']) / 'runs' / run_directory(run) if run is not None else Path(spec['out'])
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / 'workspace.json'
        self.state = (json.loads(self.state_path.read_text()) if self.state_path.exists() else
                      {'changes': {}, 'order': [], 'evaluations': [], 'committed': None, 'defects': []})
        self.program = spec['program']
        self._prepare, self._prepared = prepare_fn, None

    @property
    def prepared(self):
        if self._prepared is None:
            self._prepared = self._prepare(self.spec, self.evaluate_cases)
            if not self._prepared['admitted']:
                raise ValueError('the program has no measurable improvement headroom under this policy (support success '
                                 f"{self._prepared['support_success']}, resolved {self._prepared['resolved_fraction']})")
        return self._prepared

    @property
    def attempts(self):
        if not hasattr(self, '_attempts'):
            self._attempts = {item['digest']['id']: item for item in self.prepared['attempts']}
        return self._attempts

    def _save(self):
        write_json_durable(self.state_path, self.state)

    def failures(self):
        return [item['digest'] for item in self.attempts.values() if item['digest']['outcome'] != 'task_success']

    def all_attempts(self):
        return [item['digest'] for item in self.attempts.values()]

    def attempt(self, attempt_id):
        if attempt_id not in self.attempts:
            raise ValueError(f'no support attempt {attempt_id!r}; workspace.attempts() lists them')
        return self.attempts[attempt_id]['steps']

    def source(self, path=None):
        files = self.program['files']
        if path is None:
            return sorted(files)
        if path not in files:
            raise ValueError(f'no file {path!r}; workspace.source() lists the program files')
        return files[path]

    def problems(self, change):
        if not isinstance(change, dict) or change.get('kind') not in CHANGE_KINDS:
            return ['kind is guidance, skill, instructions, signature, withhold or revert']
        kind = change['kind']
        if any(len(value) > MAX_TEXT for value in change.values() if isinstance(value, str)):
            return [f'keep each text under {MAX_TEXT} characters']
        functions = {Path(path).stem for path in self.program['files'] if path.endswith('.nl')}
        if kind == 'guidance':
            scope = change.get('scope') or {}
            if scope.get('kind') not in SCOPE_KINDS:
                return ['scope is {kind: "program"}, {kind: "function", name} or {kind: "inputs", when}']
            if scope['kind'] == 'function' and scope.get('name') not in functions:
                return [f"scope.name is one of the program's functions: {', '.join(sorted(functions))}"]
            if scope['kind'] == 'inputs' and not str(scope.get('when') or '').strip():
                return ['an inputs scope says which inputs it is for in when']
            if not str(change.get('text') or '').strip():
                return ['guidance needs text']
            if _placeholder(change['text']):
                return ['text is the guidance itself, which the function reads as written; write it out in full']
            return []
        if kind == 'revert':
            return [] if change.get('id') in self.state['changes'] else ['revert names the id of an earlier proposed change']
        if kind == 'skill':
            if change.get('action') not in ('create', 'revise', 'retire'):
                return ['action is create, revise or retire']
            if not re.fullmatch(r'[a-z][a-z0-9-]{1,40}', str(change.get('name') or '')):
                return ['name is lowercase words joined by hyphens']
            try:
                skill_folder(self.program['files'], change.get('function'))
            except ValueError as error:
                return [str(error)]
            if change['action'] != 'retire' and not (str(change.get('description') or '').strip()
                                                     and str(change.get('body') or '').strip()):
                return ['a skill needs a description (when to use it) and a body (what to do)']
            if change['action'] != 'retire' and _placeholder(change['body']):
                return ['body is the skill text the function reads as written; write out its steps in full']
        if kind == 'withhold' and (not isinstance(change.get('tools'), list) or not change['tools']
                                   or any(tool not in WITHHOLDABLE for tool in change['tools'])):
            return ['tools lists tools to withhold, from: ' + ', '.join(WITHHOLDABLE)]
        if kind in ('instructions', 'signature', 'withhold') and change.get('file') not in self.program['files']:
            return [f"file is one of: {', '.join(sorted(p for p in self.program['files'] if p.endswith('.nl')))}"]
        if kind == 'signature' and change.get('field') not in ('args', 'returns'):
            return ['field is args or returns']
        try:
            changed = apply_changes({'semantics': {'files': self.program['files']}}, [change])
        except Exception as error:  # noqa: BLE001 - reported to the supervisor as its own mistake
            return [str(error)[:300]]
        return self.check(changed)

    def propose(self, change):
        problems = self.problems(change)
        if problems:
            return {'problems': problems}
        change_id = 'c' + str(len(self.state['order']) + 1)
        self.state['changes'][change_id] = change
        self.state['order'].append(change_id)
        self._save()
        return {'id': change_id}

    def resolve(self, ids):
        """The effective changes of a change set (reverts drop their target), in proposal order."""
        if not isinstance(ids, list) or any(i not in self.state['changes'] for i in ids):
            raise ValueError('ids are change ids returned by propose')
        chosen = [i for i in self.state['order'] if i in set(ids)]
        dropped = {self.state['changes'][i]['id'] for i in chosen if self.state['changes'][i]['kind'] == 'revert'}
        return [(i, self.state['changes'][i]) for i in chosen
                if i not in dropped and self.state['changes'][i]['kind'] != 'revert']

    def candidate(self, ids):
        changes = self.resolve(ids)
        guidance = [change for _, change in changes if change['kind'] == 'guidance']
        files = [change for _, change in changes if change['kind'] in ('instructions', 'signature', 'withhold', 'skill')]
        files += input_skills(guidance, self.program['root'])
        return {'ids': [i for i, _ in changes], 'files': files,
                'guidance': compose_guidance([c for c in guidance if c['scope']['kind'] != 'inputs']) or None}

    def _run(self, cases, candidate, name, samples=None):
        changed = [apply_changes(case, candidate['files']) for case in cases]
        return self.evaluate_cases(changed, candidate['guidance'], name=name, samples=samples)

    def evaluate(self, ids):
        baseline = self.prepared['support_success']
        budget = self.spec['budget']['evaluations'] - len(self.state['evaluations'])
        candidate = self.candidate(ids)
        key = canonical_json_sha256_hex(candidate)
        name = 'support-' + key[:16]
        earlier = next((item for item in self.state['evaluations'] if item['key'] == key), None)
        if earlier is None and budget <= 0:
            raise ValueError('the evaluation budget is spent; commit the best change set')
        report = self._run(self.spec['support'], candidate, name)
        attempts = [attempt_digest(result, attempt_id=f'{name}:{index}', world=world)
                    for index, (world, result) in enumerate(report['results'])]
        if earlier is None:
            self.state['evaluations'].append({'ids': candidate['ids'], 'key': key, 'success': report['success'],
                                              'dir': report.get('dir')})
            budget -= 1
        for (world, result), digest in zip(report['results'], attempts):
            self.attempts.setdefault(digest['id'], {'digest': digest, 'steps': attempt_steps(result)})
        self._save()
        return {'ids': candidate['ids'], 'success': report['success'], 'baseline': baseline, 'attempts': attempts,
                'evaluationsLeft': budget}

    def ablate(self, ids):
        return [{'id': change_id, 'without': self.evaluate([i for i in ids if i != change_id])['success']}
                for change_id in ids]

    def commit(self, ids, reason):
        if self.state['committed'] is not None:
            raise ValueError('the change set is already committed')
        self.resolve(ids)
        self.state['committed'] = {'ids': list(ids), 'reason': str(reason)[:1000]}
        self._save()
        return {'committed': list(ids)}

    def report_defect(self, text):
        self.state['defects'].append(_clip(str(text), 2000))
        self._save()
        return {'recorded': True}

    def budget(self):
        return {'evaluations': self.spec['budget']['evaluations'] - len(self.state['evaluations'])}

    def baseline(self):
        return {'attempts': len(self.prepared['attempts']), 'success': self.prepared['support_success']}

    def score(self):
        """Host only: the committed change set on the private query worlds (the baselines' worlds and seeds: paired),
        as the world score the collector records (``outcome.world``); ``task_rollouts.memetic_outcome`` reads it."""
        cached = self.out / 'score.json'
        if cached.exists():
            return json.loads(cached.read_text())
        prepared, committed = self.prepared, self.state['committed']
        candidate = self.candidate(committed['ids'] if committed else [])
        leak = leaked_answers(candidate, self.spec['query'])
        if leak:
            query = {'success': 0.0, 'refused': 'candidate text reproduces a query answer verbatim'}
        elif candidate['ids']:
            query = self._run(self.spec['query'], candidate, 'query-' + canonical_json_sha256_hex(candidate)[:16],
                              self.spec['settings']['query_samples'])
        else:
            query = {'success': prepared['baselines']['none']['success']}
        quality, anchor, gain, cost = memetic_reward(
            query['success'], prepared['baselines'], evaluations=len(self.state['evaluations']),
            budget=self.spec['budget'], cost_weight=self.spec['settings'].get('cost_weight', 0.05),
            committed=committed is not None)
        result = {'score': 100, 'done': True, 'moves': len(self.state['evaluations']),
                  'memetic': {'schema': OUTCOME_SCHEMA, 'quality': quality, 'anchor': anchor, 'gain': gain, 'cost': cost,
                              'query_success': query['success'], 'committed': committed, 'candidate': candidate,
                              'baselines': prepared['baselines'], 'defects': self.state['defects'],
                              'evaluations': self.state['evaluations'], 'leak_refused': bool(leak),
                              'query_dir': query.get('dir'), 'score_path': str(cached)}}
        write_json_durable(cached, result)
        return result


def serve(stdin=sys.stdin, stdout=None):
    """JSON-lines bridge (ts-host/src/teacher/world-bridge.ts): {id, op, ...args} -> {id, result | error}. Protocol
    lines go to a private copy of the original stdout; everything else printed while handling (nested collections)
    goes to stderr."""
    if stdout is None:
        stdout = os.fdopen(os.dup(1), 'w', buffering=1)
        os.dup2(2, 1)
    sys.stdout = sys.stderr
    workspace = None
    handlers = {
        'failures': lambda w, a: w.failures(), 'attempts': lambda w, a: w.all_attempts(),
        'attempt': lambda w, a: w.attempt(*(a.get('args') or [None])[:1]),
        'source': lambda w, a: w.source(*(a.get('args') or [])[:1]),
        'propose': lambda w, a: w.propose(*(a.get('args') or [None])[:1]),
        'evaluate': lambda w, a: w.evaluate(*(a.get('args') or [[]])[:1]),
        'ablate': lambda w, a: w.ablate(*(a.get('args') or [[]])[:1]),
        'commit': lambda w, a: w.commit(*((a.get('args') or []) + [[], ''])[:2]),
        'reportDefect': lambda w, a: w.report_defect(*(a.get('args') or [''])[:1]),
        'budget': lambda w, a: w.budget(), 'baseline': lambda w, a: w.baseline(), 'score': lambda w, a: w.score(),
    }
    for line in stdin:
        request = json.loads(line)
        try:
            if request['op'] == 'load':
                spec = json.loads(Path(request['task']).read_text())
                workspace = Workspace(spec, collection_evaluator(spec), run=request.get('run'))
                workspace.prepared   # the policy's evidence and baselines, before the supervisor's first turn
                result = {'service': 'workspace', 'declaration': DECLARATION, 'methods': list(METHODS)}
            elif workspace is None:
                raise ValueError('load the episode first')
            else:
                result = handlers[request['op']](workspace, request)
            stdout.write(json.dumps({'id': request['id'], 'result': result}, default=str) + '\n')
        except Exception as error:  # noqa: BLE001 - every failure goes back to the caller as the tool's error
            stdout.write(json.dumps({'id': request.get('id'), 'error': str(error)[:1000]}) + '\n')
        stdout.flush()


if __name__ == '__main__':
    if sys.argv[1:] != ['serve']:
        raise SystemExit('usage: python -m natlang_neuralese.rl.memetic_workspace serve')
    serve()
