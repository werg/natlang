"""The memetic improvement workspace: the crisp service an "improve program P" episode acts through.

One episode (``natlang.memetic-episode/1`` spec, written by ``rl.memetic_stage``) gives the supervisor program
(``applications/memetic-improver``) a target program P, P's support invocations with the parent's outcomes on them,
and typed operations. The supervisor's decisions (what failed, what to change, whether a change helped, when to stop)
are its own natural-language turns; this module is mechanism only:

- evidence: support attempts of the parent (outcome class, failed checks, returned value, errors, steps), never gold,
  never query/development/confirmation worlds;
- typed changes: guidance (program, one function, or an input class), skills in a function's folder, instruction
  find/replace patches, signature (args/returns) edits, reverts; each is checked before it gets an id (exact find,
  frontmatter parse, natlang loader check of the changed program);
- evaluation: the program with a change set runs on the support worlds through the production collector on the
  episode's own actor, paired seeds, cached by change-set digest; ablation drops one change at a time;
- adaptation: ``fit`` calls the LoRA-bank coefficient-fit service when one is configured (NATLANG_BANK_SERVICE),
  else reports that it is unavailable (an explicit capability error, never a silent no-op);
- commit: the supervisor's final change set. ``score`` (host only, never offered to the model) evaluates the committed
  set on the private query worlds and returns the episode's graded outcome for ``rl.task_rollouts.score``.

Served as a JSON-lines world process (``ts-host/src/teacher/world-bridge.ts`` kind ``memetic``):
``python -m natlang_neuralese.rl.memetic_workspace serve``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex
from ..common.jsonio import write_json_durable

SPEC_SCHEMA = 'natlang.memetic-episode/1'
CHANGE_KINDS = ('guidance', 'skill', 'instructions', 'signature', 'revert', 'adapter')
SCOPE_KINDS = ('program', 'function', 'inputs')
MAX_TEXT = 4000
REPO = Path(__file__).resolve().parents[4]

DECLARATION = '''declare const workspace: {
  /** The failed support attempts of the program: id, outcome, what failed, the returned value, errors. */
  failures(): Promise<Attempt[]>;
  /** All support attempts, passed and failed. */
  attempts(): Promise<Attempt[]>;
  /** One attempt step by step: tool, code, reasoning and output of each step, clipped. */
  attempt(id: string): Promise<AttemptStep[]>;
  /** The program's file paths, or the text of one file. */
  source(path?: string): Promise<string[] | string>;
  /** Check one change and keep it under a new id; problems explain a change that cannot apply. */
  propose(change: Change): Promise<{ id: string } | { problems: string[] }>;
  /** Run the program with exactly these changes on its support invocations. */
  evaluate(ids: string[]): Promise<Evaluation>;
  /** The support success rate with each listed change left out, one at a time. */
  ablate(ids: string[]): Promise<{ id: string; without: number }[]>;
  /** Fit adapter weights (LoRA bank mixture coefficients) on the program's verified support runs; returns a change id. */
  fit(spec: FitSpec): Promise<{ id: string; supportLoss: { before: number; after: number } } | { problems: string[] }>;
  /** Commit the final change set (possibly empty) with one sentence on what it fixes. Call once, at the end. */
  commit(ids: string[], reason: string): Promise<{ committed: string[] }>;
  /** Record a defect of the harness or of the task's check (it is not a program change and earns nothing). */
  reportDefect(text: string): Promise<{ recorded: true }>;
  /** Evaluations and fits left. */
  budget(): Promise<{ evaluations: number; fits: number }>;
};
type Attempt = { id: string; world: string; outcome: string; failedChecks: string[]; returned: string; errors: string[] };
type AttemptStep = { turn: number; tool: string; code?: string; arguments?: string; reasoning?: string; output: string };
type Scope = { kind: "program" } | { kind: "function"; name: string } | { kind: "inputs"; when: string };
type Change =
  | { kind: "guidance"; scope: Scope; text: string }
  | { kind: "skill"; function: string; name: string; description: string; body: string; action: "create" | "revise" | "retire" }
  | { kind: "instructions"; file: string; find: string; replace: string }
  | { kind: "signature"; file: string; field: "args" | "returns"; name?: string; type: string }
  | { kind: "revert"; id: string };
type FitSpec = { worlds?: string[]; components?: number; steps?: number; scope?: "all" | "attention" | "convolution"; init?: "learned" | "zero" };
type Evaluation = { ids: string[]; success: number; baseline: number; attempts: Attempt[]; evaluationsLeft: number };'''

METHODS = ('failures', 'attempts', 'attempt', 'source', 'propose', 'evaluate', 'ablate', 'fit', 'commit',
           'reportDefect', 'budget')


def _clip(text, limit):
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + f' <<cut off: {len(text) - limit} more characters>>'


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


def attempt_digest(rollout_result, *, attempt_id, world):
    """Crisp evidence of one support attempt: what happened, never the expected answer."""
    outcome = (rollout_result or {}).get('outcome') or {}
    checks = outcome.get('checks') or {}
    failed = sorted({key for key, value in checks.items() if value is False} |
                    {str(reason) for reason in outcome.get('rejection_reasons') or []})
    errors = [_clip(str(item.get('message', item)), 240) for item in outcome.get('scope_failures') or []][:3]
    errors += [_clip(str(d), 240) for a in outcome.get('action_ledger') or [] if isinstance(a, dict)
               for d in (a.get('diagnostics') or [])][-3:]
    return {'id': attempt_id, 'world': world, 'outcome': str(outcome.get('status') or 'unknown'),
            'failedChecks': failed, 'returned': _clip(outcome.get('value'), 400), 'errors': errors}


def attempt_steps(rollout_result, limit=40):
    """One attempt step by step from its recorded trajectory: each action, the model's brief reasoning and the
    tool output that came back (the next turn's newest tool message)."""
    turns = (rollout_result or {}).get('trajectory') or []
    steps = []
    for index, turn in enumerate(turns):
        response = turn.get('model_response') or {}
        if isinstance(response, str):
            try:
                import ast
                response = ast.literal_eval(response)
            except (ValueError, SyntaxError):
                response = {}
        following = turns[index + 1].get('context') if index + 1 < len(turns) else None
        tool_out = next((m.get('content') for m in reversed(following or []) if isinstance(m, dict)
                         and m.get('role') == 'tool'), '')
        reasoning = (turn.get('assistant') or {}).get('reasoning') if isinstance(turn.get('assistant'), dict) else None
        for call in response.get('calls') or [[None, {}]]:
            name = call[0] if isinstance(call, (list, tuple)) else (call or {}).get('name')
            arguments = call[1] if isinstance(call, (list, tuple)) and len(call) > 1 else {}
            steps.append({'turn': index + 1, 'tool': str(name or 'reply'),
                          **({'code': _clip(arguments.get('code'), 700)} if isinstance(arguments, dict)
                             and arguments.get('code') else {}),
                          **({'arguments': _clip({k: v for k, v in arguments.items() if k != 'code'}, 300)}
                             if isinstance(arguments, dict) and set(arguments) - {'code'} else {}),
                          **({'reasoning': _clip(reasoning, 300)} if reasoning else {}),
                          'output': _clip(tool_out or '', 500)})
    return steps[:limit]


def compose_guidance(changes):
    """One program-guidance text from the guidance changes, scope by scope (the stored state's exact source text)."""
    parts = []
    for change in changes:
        scope = change['scope']
        head = ('For every call of this program:' if scope['kind'] == 'program' else
                f"For calls of {scope['name']}:" if scope['kind'] == 'function' else
                f"For inputs where {scope['when']}:")
        parts.append(head + '\n' + change['text'].strip())
    return '\n\n'.join(parts)


def apply_changes(case, changes):
    """The task row with the file-level changes applied (instructions, signatures, skills). Guidance and adapters
    travel separately (program-guidance binding, adapter binding); this function never touches inputs or checks."""
    case = copy.deepcopy(case)
    files = case['semantics']['files']
    for change in changes:
        kind = change['kind']
        if kind == 'instructions':
            text = files[change['file']]
            meta, body = frontmatter(text)
            if body.count(change['find']) != 1:
                raise ValueError(f"instructions patch: the text to replace must occur exactly once in {change['file']}")
            files[change['file']] = render_nl(meta, body.replace(change['find'], change['replace'], 1))
        elif kind == 'signature':
            meta, body = frontmatter(files[change['file']])
            if change['field'] == 'returns':
                meta['returns'] = change['type']
            else:
                args = dict(meta.get('args') or {})
                if not change.get('name'):
                    raise ValueError('a signature args change names the argument')
                args[change['name']] = change['type']
                meta['args'] = args
            files[change['file']] = render_nl(meta, body)
        elif kind == 'skill':
            folder = skill_folder(files, change['function'])
            path = f"{folder}skills/{change['name']}/SKILL.md"
            if change['action'] == 'retire':
                files.pop(path, None)
            else:
                files[path] = ('---\nname: ' + change['name'] + '\ndescription: ' + json.dumps(change['description'])
                               + '\n---\n' + change['body'].strip() + '\n')
    return case


def skill_folder(files, function):
    """The folder of the function that reads the skill: the root folder for a root function, its callable folder
    (the directory of its .nl file) for a child. A program's own ``skills/`` folder is bound to its functions."""
    matches = [path for path in files if path.endswith('.nl') and Path(path).stem == function]
    if len(matches) != 1:
        raise ValueError(f'skill: no unique function named {function!r} in the program')
    parent = str(Path(matches[0]).parent)
    return '' if parent in ('', '.') else parent + '/'


def check_program(case, root=REPO):
    """Load the changed program with the natlang loader (the compiler's own check). Returns problems."""
    import subprocess
    files = case['semantics']['files']
    script = ("import('" + (root / 'ts-host/dist/index.js').as_uri() + "').then(m=>{const files=JSON.parse("
              "process.argv[1]);const problems=[];for(const p of Object.keys(files).filter(p=>p.endsWith('.nl'))){"
              "try{m.loadVirtualNatlang(files,p)}catch(e){problems.push(p+': '+String(e.message||e).split('\\n')[0])}}"
              "process.stdout.write(JSON.stringify(problems))})")
    try:
        output = subprocess.run(['node', '--input-type=module', '-e', script, json.dumps(files)], capture_output=True,
                                text=True, timeout=120, check=True).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return [f'the natlang loader could not check the program: {str(error)[:200]}']
    return json.loads(output or '[]')


class Workspace:
    """One episode's workspace state. ``evaluator(cases, guidance_text, adapter, *, name)`` returns
    ``{'success': float, 'results': [(world, result)], 'dir': str}``; injected so tests run without a model."""

    def __init__(self, spec, evaluator, *, fitter=None, check=check_program):
        if spec.get('schema') != SPEC_SCHEMA:
            raise ValueError('memetic workspace needs a natlang.memetic-episode/1 spec')
        self.spec = spec
        self.evaluate_cases = evaluator
        self.fitter = fitter
        self.check = check
        self.out = Path(spec['out'])
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / 'workspace.json'
        self.state = (json.loads(self.state_path.read_text()) if self.state_path.exists() else
                      {'changes': {}, 'order': [], 'evaluations': [], 'fits': 0, 'committed': None, 'defects': []})
        self.program = spec['program']
        self.attempts = {item['digest']['id']: item for item in spec['evidence']['attempts']}

    def _save(self):
        write_json_durable(self.state_path, self.state)

    # --- evidence -------------------------------------------------------------------------------------------------
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

    # --- changes --------------------------------------------------------------------------------------------------
    def problems(self, change):
        if not isinstance(change, dict) or change.get('kind') not in CHANGE_KINDS or change['kind'] == 'adapter':
            return ['kind is guidance, skill, instructions, signature or revert']
        kind = change['kind']
        texts = [value for key, value in change.items() if isinstance(value, str)]
        if any(len(text) > MAX_TEXT for text in texts):
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
            return []
        if kind == 'revert':
            if change.get('id') not in self.state['changes']:
                return ['revert names the id of an earlier proposed change']
            return []
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
        if kind in ('instructions', 'signature') and change.get('file') not in self.program['files']:
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
            raise ValueError('ids are change ids returned by propose or fit')
        chosen = [i for i in self.state['order'] if i in set(ids)]
        dropped = {self.state['changes'][i]['id'] for i in chosen if self.state['changes'][i]['kind'] == 'revert'}
        return [(i, self.state['changes'][i]) for i in chosen if i not in dropped
                and self.state['changes'][i]['kind'] != 'revert']

    def candidate(self, ids):
        changes = self.resolve(ids)
        guidance = [change for _, change in changes if change['kind'] == 'guidance']
        adapters = [change for _, change in changes if change['kind'] == 'adapter']
        if len(adapters) > 1:
            raise ValueError('one adapter fit per change set')
        files = [change for _, change in changes if change['kind'] in ('instructions', 'signature', 'skill')]
        return {'ids': [i for i, _ in changes], 'files': files, 'guidance': compose_guidance(guidance) or None,
                'adapter': adapters[0]['adapter'] if adapters else None}

    def _run(self, cases, candidate, name):
        changed = [apply_changes(case, candidate['files']) for case in cases]
        return self.evaluate_cases(changed, candidate['guidance'], candidate['adapter'], name=name)

    def evaluate(self, ids):
        budget = self.spec['budget']['evaluations'] - len(self.state['evaluations'])
        if budget <= 0:
            raise ValueError('the evaluation budget is spent; commit the best change set')
        candidate = self.candidate(ids)
        key = canonical_json_sha256_hex(candidate)
        name = 'support-' + key[:16]
        report = self._run(self.spec['support'], candidate, name)
        attempts = [attempt_digest(result, attempt_id=f'{name}:{index}', world=world)
                    for index, (world, result) in enumerate(report['results'])]
        self.state['evaluations'].append({'ids': candidate['ids'], 'key': key, 'success': report['success'],
                                          'dir': report.get('dir')})
        self._save()
        return {'ids': candidate['ids'], 'success': report['success'], 'baseline': self.spec['evidence']['success'],
                'attempts': attempts, 'evaluationsLeft': budget - 1}

    def ablate(self, ids):
        rows = []
        for change_id in ids:
            rest = [i for i in ids if i != change_id]
            rows.append({'id': change_id, 'without': self.evaluate(rest)['success']})
        return rows

    def fit(self, spec):
        if self.state['fits'] >= self.spec['budget']['fits']:
            return {'problems': ['the fit budget is spent']}
        if self.fitter is None:
            return {'problems': ['adapter fits are unavailable in this run (no bank coefficient-fit service is '
                                 'configured); use guidance, skills or patches']}
        result = self.fitter(self.spec, spec or {})
        self.state['fits'] += 1
        change_id = 'c' + str(len(self.state['order']) + 1)
        self.state['changes'][change_id] = {'kind': 'adapter', 'adapter': result['adapter'], 'spec': spec or {}}
        self.state['order'].append(change_id)
        self._save()
        return {'id': change_id, 'supportLoss': result['support_loss']}

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
        return {'evaluations': self.spec['budget']['evaluations'] - len(self.state['evaluations']),
                'fits': self.spec['budget']['fits'] - self.state['fits']}

    # --- host-only outcome ----------------------------------------------------------------------------------------
    def score(self):
        """Evaluate the committed change set on the private query worlds; the graded episode outcome."""
        cached = self.out / 'score.json'
        if cached.exists():
            return json.loads(cached.read_text())
        committed = self.state['committed']
        candidate = self.candidate(committed['ids'] if committed else [])
        leak = leaked_answers(candidate, self.spec['query'])
        if leak:
            query = {'success': 0.0, 'refused': 'candidate text reproduces a query answer verbatim'}
        elif candidate['ids']:
            query = self._run(self.spec['query'], candidate, 'query-' + canonical_json_sha256_hex(candidate)[:16])
        else:
            query = {'success': self.spec['baselines']['none']['success']}
        quality, anchor, gain, cost = memetic_reward(query['success'], self.spec['baselines'],
            evaluations=len(self.state['evaluations']), fits=self.state['fits'], budget=self.spec['budget'],
            cost_weight=self.spec.get('cost_weight', 0.05), committed=committed is not None)
        result = {'score': 100, 'done': True, 'moves': len(self.state['evaluations']),
                  'memetic': {'schema': 'natlang.memetic-outcome/1', 'quality': quality, 'anchor': anchor,
                              'gain': gain, 'cost': cost, 'query_success': query['success'],
                              'committed': committed, 'candidate': candidate,
                              'baselines': self.spec['baselines'], 'defects': self.state['defects'],
                              'evaluations': self.state['evaluations'], 'leak_refused': bool(leak),
                              'query_dir': query.get('dir')}}
        write_json_durable(cached, result)
        return result


def leaked_answers(candidate, query_cases):
    """Defensive guard: a candidate whose texts contain a query world's expected answer verbatim is refused."""
    texts = [candidate.get('guidance') or ''] + [json.dumps(change, ensure_ascii=False) for change in candidate['files']]
    blob = '\n'.join(texts)
    for case in query_cases:
        expected = (case.get('semantics') or {}).get('expected')
        for value in ([expected] if isinstance(expected, str) else
                      [v for v in (expected.values() if isinstance(expected, dict) else []) if isinstance(v, str)]):
            if len(value) >= 6 and value in blob:
                return True
    return False


def memetic_reward(query_success, baselines, *, evaluations, fits, budget, cost_weight, committed):
    """(quality, anchor, gain, cost) of one episode.

    quality: the committed candidate's query success rate, minus a bounded cost for the budget it used (at most
    ``cost_weight``) and minus that weight again when the episode never committed. anchor: the best trivial
    improvement's query success (no change, and the default crisp improvement); the learner centers on
    max(group mean, anchor), so a candidate that does not beat doing the trivial thing never earns a positive
    advantage (rl.native_replay.group_episode_advantages)."""
    available = [item['success'] for item in baselines.values() if item is not None and item.get('success') is not None]
    anchor = max(available) if available else 0.
    used = evaluations / max(1, budget['evaluations']) + (fits / budget['fits'] if budget.get('fits') else 0.)
    cost = cost_weight * min(1., used / 2) + (0. if committed else cost_weight)
    quality = max(0., min(1., query_success - cost))
    return quality, anchor, query_success - anchor, cost


# --- world process --------------------------------------------------------------------------------------------------

def production_evaluator(spec, endpoint):
    """Evaluate a changed program on a slice through the production collector, on the episode's actor."""
    from ..eval.task_packet import read_jsonl
    from .memetic_program import write_state, archive_state, program_binding, actor_identity
    from .production import collect
    identity = actor_identity(endpoint)
    settings = spec['eval']

    def evaluate(cases, guidance, adapter, *, name):
        if adapter is not None:
            raise ValueError('adapter evaluation needs an adapter-capable actor; this episode has none')
        out = Path(spec['out']) / 'evals' / name
        if (out / 'summary.json').exists():
            return json.loads((out / 'summary.json').read_text())
        out.mkdir(parents=True, exist_ok=True)
        packet = out / 'cases.jsonl'
        packet.write_text(''.join(json.dumps(case, ensure_ascii=False) + '\n' for case in cases))
        binding = None
        if guidance:
            written = write_state(endpoint, guidance, identity)
            binding = program_binding(spec['program']['id'], identity, guidance, written,
                                      archive_state(endpoint, out, written))
        report = collect(cases=packet, out=out / 'collection', endpoints=[endpoint],
                         policy_id=identity['checkpoint_sha256'], rows=len(cases), group=settings['samples'],
                         workers=settings['workers'], root_mode='student', seed_base=settings['seed_base'],
                         case_seconds=settings['case_seconds'], max_model_requests=settings['max_model_requests'],
                         max_turns=settings['max_turns'], program_guidance=binding, prompt_rendering='full')
        results = []
        rewards = read_jsonl(out / 'collection' / 'rewards.jsonl')
        for reward in rewards:
            path = out / 'collection' / f"row-{reward['group']:03d}" / f"sample-{reward['sample']:02d}" / 'results.jsonl'
            result = (read_jsonl(path) or [None])[0] if path.exists() else None
            if result is not None:
                result.setdefault('outcome', {})['status'] = reward['outcome']
            results.append((cases[reward['group']]['id'], result))
        scored = [r for r in rewards if r['reward'] is not None]
        summary = {'success': (sum(r['reward'] for r in scored) / len(scored)) if scored else 0.,
                   'scored': len(scored), 'results': results, 'dir': str(out),
                   **({'guidance_state': binding['native_state_id']} if binding else {}),
                   'report': {key: report.get(key) for key in ('successes', 'scored', 'rollouts')}}
        write_json_durable(out / 'summary.json', summary)
        return summary
    return evaluate


def serve(stdin=sys.stdin, stdout=None):
    """JSON-lines bridge (ts-host/src/teacher/world-bridge.ts): {id, op, ...args} -> {id, result | error}.
    Protocol lines go to the original stdout; anything else printed while handling (collector progress) goes to
    stderr, so it cannot corrupt the channel."""
    stdout = stdout or sys.stdout
    sys.stdout = sys.stderr
    workspace = None
    handlers = {
        'failures': lambda w, a: w.failures(), 'attempts': lambda w, a: w.all_attempts(),
        'attempt': lambda w, a: w.attempt(*(a.get('args') or [None])[:1]),
        'source': lambda w, a: w.source(*(a.get('args') or [])[:1]),
        'propose': lambda w, a: w.propose(*(a.get('args') or [None])[:1]),
        'evaluate': lambda w, a: w.evaluate(*(a.get('args') or [[]])[:1]),
        'ablate': lambda w, a: w.ablate(*(a.get('args') or [[]])[:1]),
        'fit': lambda w, a: w.fit(*(a.get('args') or [{}])[:1]),
        'commit': lambda w, a: w.commit(*((a.get('args') or []) + [[], ''])[:2]),
        'reportDefect': lambda w, a: w.report_defect(*(a.get('args') or [''])[:1]),
        'budget': lambda w, a: w.budget(), 'score': lambda w, a: w.score(),
    }
    for line in stdin:
        request = json.loads(line)
        try:
            if request['op'] == 'load':
                spec = json.loads(Path(request['task']).read_text())
                endpoint = os.environ.get('NATLANG_MEMETIC_ENDPOINT') or spec.get('endpoint')
                if not endpoint:
                    raise ValueError('the memetic workspace needs the episode actor endpoint (NATLANG_MEMETIC_ENDPOINT)')
                workspace = Workspace(spec, production_evaluator(spec, endpoint))
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
