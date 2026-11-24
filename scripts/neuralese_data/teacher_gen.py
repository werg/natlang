#!/usr/bin/env python3
"""Teacher episodes on the GRPO task pool, for distillation (SFT) into the LFM3 actor.

Any OpenAI-compatible chat endpoint is the teacher (a local llama-server, a hosted free tier, a paid API): `--endpoint`
and `--model-id` (plus `--api-key-env` for hosted ones). A natlang provider profile is one too: a teacher spec with
`"provider"` (e.g. `{"model_id": "gpt-6-luna", "provider": "openai-codex", "reasoning_effort": "low"}`) runs the collector's
Pi provider transport with natlang's stored credentials (`natlang auth`), the transport natlang programs use. Episodes run through the GRPO path itself: `task_rollouts.prepare`
(student root, full prompt rendering, the shared `EPISODE_MODEL_CALLS` budget), `run_bonsai_queue.py` (the frozen
collector, without the RL capture: the teacher's tokens are not the student's) and `task_rollouts.score` (the task's exact
oracle). Each scored episode's `natlang.rollout/1` record (turns = {messages, tools, target}) is the SFT material; the
successes are the distillation set, failures keep their scores.

Tasks are the GRPO pools minus the held closure (`grpo.load_tasks`: held-v3 `exclusion_ids` incl. held-v2, shared source
groups, `ops/pool-errata`, the missing ext_codebase repos), ordered once into `WORK/tasks.jsonl`: tasks in the base's pass band
first, then tasks the base fails, then unmeasured ones, then those it solves, each tier round-robin over families.
A gap campaign (`--covered-by` earlier WORKs) keeps only the tasks without a teacher success there, ordered by GAP_TIERS
(expected value of a teacher attempt): tasks the student rarely or never solves that no teacher failed (`--rare-rate`, over
the base rates, `--student-rates`, `--student-groups` and `--frontier-rates`, newest measurement first), new task supply
(`--priority-pool`), tasks in the student's pass band (a teacher success adds a solved variant), unmeasured tasks no teacher
tried, then tasks teachers failed (fewest failures first; the student-rare ones last: 1-3% teacher yield on 2026-11-24).
Tasks the student solves at least the band's upper rate are left out. Without `--limit` the campaign is a standing queue:
every eligible task, until the driver asks it to drain (WORK/DRAIN) for a newer frontier's re-prioritised queue.
Several teachers share one WORK: each claims a task before running it (WORK/claims), so faster teachers take more.

  pause:   touch WORK/PAUSE   (running episodes finish, no new ones start; remove the file to continue);
           --pause-below-available-gb N does the same while host MemAvailable is below N GB
  drain:   touch WORK/DRAIN   (running episodes finish, no new ones start, the process exits: a newer queue replaces it)
  stop:    systemctl --user stop <unit>   (killed episodes are redone on the next launch)
  resume:  launch the same command again (finished tasks are skipped)
  routing: each teacher takes the first task of the ordered queue it may take; a task goes first to the teacher with
           the best yield on its family (--route-ratio, over the --covered-by campaigns' and this campaign's verdicts);
           a teacher far below --route-floor on a family leaves it alone; a weak teacher (measured yield on fresh,
           teacher_untried tasks below --weak-yield) never takes the student-rare tiers, which the strong teachers take
           first; a task a teacher fails here goes on to another teacher (--attempts teachers per task,
           episodes/tNNNNNN-aK); a teacher spec's "min_failed" reserves a (stronger, scarcer) teacher for tasks that many
           teachers failed
  provider failures (an episode ending before its first model turn, or with exhausted provider retries: 429s, quota windows,
  refused or expired credentials) are no verdict: deferred.jsonl, retried twice; a teacher whose provider keeps failing
  backs off (--provider-cooldown, doubling per consecutive failure up to an hour); a refused account (credit balance,
  quota spent: PERMANENT_PROVIDER_FAILURES) retires its teacher for this launch
"""
import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(HERE)]

from natlang_neuralese.common import data_registry  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402

BASE_RATES = resolve_str('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'exp1-rft', 'base-pass-rates.json')


BASE_TIERS = ['band', 'base_fails', 'unmeasured', 'base_solves']
# --covered-by (gap campaigns): tasks without a teacher success in the earlier campaigns, by the expected value of a teacher
# attempt (value of a success x teacher yield, measured 2026-11-24 over the aei1 campaigns): the student rarely solves it
# and no teacher failed it (Luna/Sol 42%), new task supply (--priority-pool), the student's pass band, unmeasured and
# untried (Luna 48%, Sol 57%), teachers failed it (Luna 31%, Sol 40% on the older pools), the student rarely solves it and
# teachers failed it (Luna/Sol 3% after three failures: last)
GAP_TIERS = ['student_rare', 'priority_pool', 'student_band', 'teacher_untried', 'teacher_failed', 'student_rare_failed']
# tiers a weak teacher (--weak-yield) leaves to the strong ones: a failure there costs one of the task's few attempts and
# demotes it for every later campaign (Nemotron 0/99 on the aei1 student-rare tasks)
HARD_TIERS = ('student_rare', 'student_rare_failed')
# the tier whose verdicts measure a teacher's strength: fresh tasks (unmeasured, no teacher failed them)
FRESH_TIER = 'teacher_untried'
# family routing: a teacher's family rate is shrunk towards its overall rate with this many episodes of weight; a teacher
# with no record at all starts at ROUTE_DEFAULT
ROUTE_PRIOR, ROUTE_DEFAULT = 5, 0.3


def base_rank(rates, band):
    """Tiers by the base's outcome: in-band first, then base failures, then unmeasured, then base successes."""
    low, high = band

    def rank(row):
        rate = rates.get(row['id'])
        if not rate or not rate.get('scored'):
            return 2
        p = rate['successes'] / rate['scored']
        return 0 if low <= p <= high else 1 if p < low else 3
    return rank


def voided(work):
    """Rollouts paths of WORK/voided.jsonl: verdicts a harness defect produced (e.g. the teacher transport without a
    Neuralese writer before 7c6d00345). They count as no verdict: not a teacher's failure, not a try of the task."""
    path = Path(work, 'voided.jsonl')
    if not path.exists():
        return set()
    return {json.loads(line)['rollouts'] for line in path.open() if line.strip()}


def verdicts(work):
    """The manifest records of a teacher_gen WORK that are teacher verdicts (voided ones left out)."""
    path = Path(work, 'manifest.jsonl')
    if not path.exists():
        return
    void = voided(work)
    for line in path.open():
        if line.strip():
            record = json.loads(line)
            if record.get('rollouts') not in void:
                yield record


def teacher_coverage(works):
    """task id -> {'success': bool, 'failed_by': teacher model ids with a scored non-success} over earlier campaigns'
    manifests (deferred, no-verdict records never enter a manifest; voided ones are left out)."""
    coverage = {}
    for work in works:
        for record in verdicts(work):
            entry = coverage.setdefault(record['task_id'], {'success': False, 'failed_by': set()})
            for episode in record['episodes']:
                if episode['outcome'] == 'task_success' and (episode.get('reward') or 0) >= 1:
                    entry['success'] = True
                else:
                    entry['failed_by'].add(record['teacher']['model_id'])
    return coverage


def family_stats(works):
    """(family, teacher model id) -> [successes, scored episodes] over teacher_gen manifests (routing's evidence)."""
    stats = {}
    for work in works:
        for record in verdicts(work):
            add_stats(stats, record)
    return stats


def add_stats(stats, record):
    entry = stats.setdefault((record.get('family') or '?', record['teacher']['model_id']), [0, 0])
    for episode in record['episodes']:
        entry[0] += episode['outcome'] == 'task_success' and (episode.get('reward') or 0) >= 1
        entry[1] += 1


def rates_file(path):
    """task id -> (successes, scored) of a rates file: ``expert_iteration select``'s ``tasks.rates.json`` (every
    collection's outcomes, newest first) or a round's ``frontier.json`` (``{"tasks": {...}}``)."""
    data = json.loads(Path(path).read_text())
    data = data.get('tasks', data) if isinstance(data.get('tasks'), dict) else data
    return {k: (v['successes'], v['scored']) for k, v in data.items() if isinstance(v, dict) and v.get('scored')}


def student_rates(base_rates, group_globs, rate_files=(), newest_files=()):
    """task id -> (successes, scored), the newest measurement of each task winning: the base rates, then each of
    ``rate_files`` (``rates_file``), then the student samples of the given GRPO group directories together
    (`rewards.jsonl`; JOLT+ teacher attempts left out), then each of ``newest_files`` (the newest frontier)."""
    import glob
    rates = {k: (v['successes'], v['scored']) for k, v in base_rates.items() if v.get('scored')}
    for path in rate_files:
        rates.update(rates_file(path))
    groups = {}
    for pattern in group_globs:
        for path in sorted(glob.glob(str(Path(pattern) / 'rewards.jsonl'))):
            for line in Path(path).open():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('reward') is None or row.get('jolt_role', 'student') != 'student':
                    continue
                entry = groups.setdefault(row['task_id'], [0, 0])
                entry[0] += row['reward'] == 1.0
                entry[1] += 1
    rates.update({k: tuple(v) for k, v in groups.items()})
    for path in newest_files:
        rates.update(rates_file(path))
    return rates


def gap_rank(coverage, rates, priority_ids, rare, solved=0.9):
    """GAP_TIERS rank of a row, or None (left out): a teacher success already exists, or the student solves it at least
    ``solved`` of the time (unless it is priority-pool supply)."""
    def rank(row):
        entry = coverage.get(row['id'])
        if entry and entry['success']:
            return None
        failed = bool(entry and entry['failed_by'])
        rate = rates.get(row['id'])
        p = rate[0] / rate[1] if rate and rate[1] else None
        if p is not None and p <= rare:
            return 5 if failed else 0
        if row['id'] in priority_ids:
            return 1
        if p is not None and p >= solved:
            return None
        if failed:
            return 4
        return 2 if p is not None else 3
    return rank


def order_tasks(rows, rank, seed, sub=None):
    """Rows ordered by tier (`rank`; None leaves a row out), within a tier by ``sub`` (e.g. how many teachers failed it;
    fewest first), each (tier, sub) round-robin over a shuffled family order: no family repeats within it before every
    family had its turn."""
    sub = sub or (lambda row: 0)
    rng = random.Random(seed)
    rows = [row for row in rows if rank(row) is not None]
    families = {}
    for row in rows:
        families.setdefault(row.get('family') or '?', []).append(row)
    names = sorted(families)
    rng.shuffle(names)
    keyed = []
    for order, name in enumerate(names):
        members = families[name]
        rng.shuffle(members)
        seen = {}
        for row in members:   # the k-th task of its tier in this family comes in the tier's k-th family round
            tier = (rank(row), sub(row))
            seen[tier] = seen.get(tier, -1) + 1
            keyed.append(((*tier, seen[tier], order), row))
    keyed.sort(key=lambda item: item[0])
    return [row for _, row in keyed]


GEMINI_HOST_CALL_FIELDS = {'extra_content': {'google': {'thought_signature': 'skip_thought_signature_validator'}}}


# Gemini's OpenAI-compatible endpoints: AI Studio (API key) and Vertex AI (the GCP project's billing and credits)
GEMINI_HOSTS = ('generativelanguage.googleapis.com', 'aiplatform.googleapis.com')


def default_chat_config(url, model_id=None):
    """Request controls when none are given: top-p; for Gemini's OpenAI endpoints also no `seed` field (it rejects it),
    Google's documented bypass signature on the function calls the runtime wrote itself (it rejects unsigned ones), and
    its thought summaries (`x_natlang_include_thoughts`: the turn's reasoning, not sent back: the signatures carry the
    state). Vertex also serves open models (`publisher/model-maas`): those take the plain controls; DeepSeek's thinking
    is off unless asked for (`chat_template_kwargs.thinking`)."""
    vertex_open = 'aiplatform.googleapis.com' in url and model_id is not None and not model_id.startswith('google/')
    gemini = any(host in url for host in GEMINI_HOSTS) and not vertex_open
    if vertex_open:
        return {'top_p': 0.95, **({'chat_template_kwargs': {'thinking': True}} if model_id.startswith('deepseek-ai/') else {})}
    # NIM serves several models behind a parser that reads `reasoning` and `reasoning_content` as one field and rejects
    # the pair the runtime sends by default (Mistral Large 4, MiniMax M3.1: "duplicate field"); one alias suits all of them
    nim = 'integrate.api.nvidia.com' in url
    return {'top_p': 0.95, **({'x_natlang_omit_seed': True, 'x_natlang_host_call_fields': GEMINI_HOST_CALL_FIELDS,
                               'x_natlang_include_thoughts': True, 'x_natlang_history_reasoning': 'none'} if gemini else {}),
            **({'x_natlang_history_reasoning': 'reasoning_content'} if nim else {})}


def mem_available_gb():
    with open('/proc/meminfo') as handle:
        for line in handle:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2**20
    return float('inf')


def completions_url(endpoint):
    """A bare server (`http://host:port`) serves `/v1/chat/completions`; an endpoint with a path is the API's base
    (`.../v1`, Gemini's `/v1beta/openai`, Vertex's `.../endpoints/openapi`, Z.ai's `/api/paas/v4`): `/chat/completions`."""
    from urllib.parse import urlparse
    endpoint = endpoint.rstrip('/')
    return endpoint + ('/chat/completions' if urlparse(endpoint).path.strip('/') else '/v1/chat/completions')


def chat_config_path(work, model_id):
    """Each teacher sharing WORK keeps its own request controls."""
    return Path(work) / f"chat-config.{re.sub(r'[^A-Za-z0-9._-]+', '_', model_id)}.json"


# the collector's message when a provider kept failing a request (429, 5xx, transport) through every retry
PROVIDER_FAILURE = 'provider request retry exhausted'
# a provider refusing the account itself (credits or quota spent): every further request fails the same way, so the
# teacher is retired for this launch instead of deferring task after task (an hour-long cooldown still wakes 8 slots)
PERMANENT_PROVIDER_FAILURES = ('credit balance is too low', 'insufficient_quota', 'exceeded your current quota',
                               'Insufficient balance')   # the last: Z.ai code 1113, OpenCode Zen 402
# decide() collection needs the actor's exact native capture, which a hosted teacher endpoint does not give
UNSUPPORTED_DECIDE = 'decision collection needs its exact native capture'


# HTTP teacher endpoints by host -> the provider that served them (manifest records name a provider for every teacher)
ENDPOINT_PROVIDERS = {'integrate.api.nvidia.com': 'nvidia-nim', 'openrouter.ai': 'openrouter',
                      'generativelanguage.googleapis.com': 'google-gemini', 'api.z.ai': 'zai',
                      'api.anthropic.com': 'anthropic', 'api.openai.com': 'openai',
                      'aiplatform.googleapis.com': 'google-vertex', 'api.groq.com': 'groq', 'opencode.ai': 'opencode-zen',
                      'ai-gateway.kilo.ai': 'kilo', 'api.llm7.io': 'llm7'}

# What a teacher's recorded think block is (owner 2026-11-24: teacher reasoning stays in SFT data as zero-weight
# context; a summary is never presented as a raw trace): `raw` the model's own reasoning tokens (open-weight reasoning
# models' `reasoning_content` on NIM, OpenRouter, Groq, Z.ai, Vertex's open models, a local server), `summary` a
# provider-written summary of hidden reasoning (Codex reasoning summaries, Claude's summarized thinking, Gemini's thought
# summaries under x_natlang_include_thoughts), `hidden` none returned (Gemini without it). A spec's "reasoning" overrides the provider's default.
REASONING_KINDS = ('raw', 'summary', 'hidden')
PROVIDER_REASONING = {'openai-codex': 'summary', 'anthropic': 'summary', 'nvidia-nim': 'raw', 'openrouter': 'raw',
                      'groq': 'raw', 'zai': 'raw', 'kilo': 'raw', 'llm7': 'raw', 'google-gemini': 'summary',
                      'google-vertex': 'summary'}


def reasoning_kind(teacher):
    """The teacher's recorded reasoning kind: its spec's, else its provider's, else raw for a local server."""
    if getattr(teacher, 'reasoning', None):
        return teacher.reasoning
    provider = teacher.provider or endpoint_provider(teacher.endpoint)
    if provider == 'google-vertex' and not teacher.model_id.startswith('google/'):
        return 'raw'   # an open model served on Vertex (DeepSeek, GLM, gpt-oss): its own reasoning tokens
    if provider in ('google-gemini', 'google-vertex') and 'gemma' in teacher.model_id:
        return 'raw'   # Gemma (open weights) writes its own thinking as <thought>...</thought>
    config = getattr(teacher, 'chat_config', None) or {}
    if provider in ('google-gemini', 'google-vertex') and not config.get('x_natlang_include_thoughts'):
        return 'hidden'
    return PROVIDER_REASONING.get(provider, 'raw' if str(provider).startswith('local') else 'unknown')


def endpoint_provider(endpoint):
    """The provider of an OpenAI-compatible teacher endpoint: a known host's name, `local:<port>` for a self-hosted
    server on this machine, else the host itself."""
    from urllib.parse import urlparse
    url = urlparse(endpoint)
    host = (url.hostname or '').lower()
    if host in ('127.0.0.1', 'localhost', '0.0.0.0', '::1'):
        return f'local:{url.port}' if url.port else 'local'
    return ENDPOINT_PROVIDERS.get(host, host or None)


def teacher_identity(teacher):
    """What a manifest record names as its teacher: the model, the provider that served it, and its transport (a natlang
    Pi provider with its reasoning effort, or an HTTP endpoint)."""
    if teacher.provider:
        return {'model_id': teacher.model_id, 'provider': teacher.provider, 'reasoning_effort': teacher.reasoning_effort,
                'reasoning': reasoning_kind(teacher)}
    return {'model_id': teacher.model_id, 'provider': endpoint_provider(teacher.endpoint), 'endpoint': teacher.endpoint,
            'reasoning': reasoning_kind(teacher)}


def episode_env(teacher, env):
    """The environment of one episode's collector: a teacher with `api_key_command` (a short-lived bearer token, e.g.
    Vertex AI's `gcloud auth application-default print-access-token`) gets a freshly minted key in `api_key_env`, so a
    campaign outlives the token's hour."""
    command = getattr(teacher, 'api_key_command', None)
    if not command:
        return env
    token = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True).stdout.strip()
    if not token:
        raise RuntimeError(f'api_key_command of {teacher.model_id} printed no key')
    return {**env, teacher.api_key_env: token}


def queue_command(teacher, work, queue_file, journal):
    """The frozen collector's queue runner for one episode of `teacher`."""
    command = [sys.executable, str(REPO / 'scripts/run_bonsai_queue.py'), str(queue_file), str(journal),
               '--runtime', str(REPO / 'ts-host'), '--model-id', teacher.model_id, '--model-concurrency', '1']
    if teacher.provider:   # Pi provider transport: credentials from natlang's store, request controls are the provider's
        return command + ['--provider', teacher.provider, '--reasoning-effort', teacher.reasoning_effort]
    command += ['--server', teacher.endpoint, '--chat-completions-url', teacher.chat_completions_url,
                '--chat-request-config', str(chat_config_path(work, teacher.model_id))]
    return command + (['--api-key-env', teacher.api_key_env] if teacher.api_key_env else [])


def make_teacher(spec):
    """A teacher from its --teacher spec: an OpenAI-compatible endpoint, or a natlang Pi provider (`provider`)."""
    if not spec.get('model_id') or bool(spec.get('endpoint')) == bool(spec.get('provider')):
        raise ValueError('each teacher needs model_id and exactly one of endpoint or provider')
    if spec.get('reasoning') is not None and spec['reasoning'] not in REASONING_KINDS:
        raise ValueError(f'reasoning must be one of {REASONING_KINDS}')
    if spec.get('api_key_command') is not None and not (isinstance(spec['api_key_command'], list) and spec['api_key_command']
                                                        and spec.get('api_key_env')):
        raise ValueError('api_key_command is an argv list and needs api_key_env')
    if spec.get('provider'):
        extra = set(spec) & {'chat_completions_url', 'api_key_env', 'chat_config', 'api_key_command'}
        if extra:
            raise ValueError(f'a provider teacher takes no HTTP controls: {sorted(extra)}')
        return SimpleNamespace(model_id=spec['model_id'], provider=spec['provider'], endpoint=None,
                               reasoning_effort=spec.get('reasoning_effort', 'low'), slots=int(spec.get('slots', 2)),
                               reasoning=spec.get('reasoning'),
                               min_failed=int(spec.get('min_failed', 0)), cool_until=0., failures=0, enabled=True)
    url = spec.get('chat_completions_url') or completions_url(spec['endpoint'])
    return SimpleNamespace(model_id=spec['model_id'], provider=None, endpoint=spec['endpoint'].rstrip('/'),
                           chat_completions_url=url, api_key_env=spec.get('api_key_env'), slots=int(spec.get('slots', 2)),
                           chat_config=default_chat_config(url, spec['model_id']) if spec.get('chat_config') is None else spec['chat_config'],
                           reasoning=spec.get('reasoning'), api_key_command=spec.get('api_key_command'),
                           min_failed=int(spec.get('min_failed', 0)), cool_until=0., failures=0, enabled=True)


class Generator:
    def __init__(self, args, teachers):
        self.args, self.work = args, Path(args.work)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.env = {**os.environ, 'NATLANG_REPO': str(REPO), 'PYTHONPATH': f'{REPO}/training/neuralese:{REPO}'}
        from natlang_neuralese.rl.held_eval import series
        self.env['NATLANG_MODEL_FACING'] = series()['model_facing']
        self.tasks = [json.loads(line) for line in (self.work / 'tasks.jsonl').open()]
        self.manifest = self.work / 'manifest.jsonl'
        self.teachers = teachers
        # routing evidence: the earlier campaigns' verdicts plus this one's (updated as verdicts come in)
        self.stats = family_stats(args.covered_by)
        self.overall = {}         # teacher model id -> [successes, scored] over every family
        for (_, model), (wins, scored) in self.stats.items():
            entry = self.overall.setdefault(model, [0, 0])
            entry[0], entry[1] = entry[0] + wins, entry[1] + scored
        # a teacher's strength: its yield on fresh tasks (FRESH_TIER verdicts), for the weak-teacher routing
        self.fresh = {}           # teacher model id -> [successes, scored] on FRESH_TIER tasks
        for work in args.covered_by:
            for record in verdicts(work):
                self._note_fresh(record)
        self.tried = {}           # index -> teacher model ids with a verdict on it in this campaign
        self.solved = set()
        self.recorded = {}        # index -> manifest records (voided included): the next attempt's episode directory
        if self.manifest.exists():
            for line in self.manifest.open():
                if line.strip():
                    index = json.loads(line)['index']
                    self.recorded[index] = self.recorded.get(index, 0) + 1
            for record in verdicts(self.work):
                self._note_verdict(record)
        if args.covered_by:   # failed_by as the earlier campaigns' verdicts stand now (voided ones dropped since)
            coverage = teacher_coverage(args.covered_by)
            for task in self.tasks:
                if 'failed_by' in task:
                    failed = (coverage.get(task['row']['id']) or {}).get('failed_by') or set()
                    task['failed_by'] = sorted(set(task['failed_by']) & failed)
            for record in verdicts(self.work):   # this campaign's own failures stay counted
                if record['index'] < len(self.tasks) and record['index'] not in self.solved:
                    task = self.tasks[record['index']]
                    task['failed_by'] = sorted({*task.get('failed_by', ()), record['teacher']['model_id']})
        self.done = {index for index in self.tried if self._finished(index)}
        (self.work / 'claims').mkdir(exist_ok=True)
        self.retry = {}           # index -> time: tasks whose episodes the provider failed (no verdict), retried then
        self.attempts = {}
        self.running = 0
        self.active = set()       # indices this process runs now

    def _note_fresh(self, record):
        if record.get('base_rank') == FRESH_TIER:
            entry = self.fresh.setdefault(record['teacher']['model_id'], [0, 0])
            for episode in record['episodes']:
                entry[0] += episode['outcome'] == 'task_success' and (episode.get('reward') or 0) >= 1
                entry[1] += 1

    def strength(self, teacher):
        """(posterior yield on fresh tasks, scored episodes): ROUTE_PRIOR episodes of ROUTE_DEFAULT weight."""
        s, n = self.fresh.get(teacher.model_id, (0, 0))
        return (s + ROUTE_DEFAULT * ROUTE_PRIOR) / (n + ROUTE_PRIOR), n

    def weak(self, teacher):
        """A teacher whose measured fresh-task yield is below --weak-yield: it leaves HARD_TIERS to the others."""
        return self.strength(teacher)[0] < getattr(self.args, 'weak_yield', 0)

    def _note_verdict(self, record):
        index, model = record['index'], record['teacher']['model_id']
        add_stats(self.stats, record)
        self._note_fresh(record)
        entry = self.overall.setdefault(model, [0, 0])
        for episode in record['episodes']:
            entry[0] += episode['outcome'] == 'task_success' and (episode.get('reward') or 0) >= 1
            entry[1] += 1
        self.tried.setdefault(index, set()).add(model)
        if any(e['outcome'] == 'task_success' and (e.get('reward') or 0) >= 1 for e in record['episodes']):
            self.solved.add(index)
        if index < len(self.tasks):
            task = self.tasks[index]
            if index not in self.solved and model not in (task.get('failed_by') or ()):
                task['failed_by'] = sorted({*(task.get('failed_by') or ()), model})

    def _finished(self, index):
        """No further teacher of this campaign takes the task: solved, --attempts teachers tried it, or no enabled
        teacher that has not tried it would take it."""
        tried = self.tried.get(index, set())
        if index in self.solved or len(tried) >= self.args.attempts:
            return True
        return not any(t.enabled and t.model_id not in tried and self._eligible(index, t) for t in self.teachers)

    def _estimate(self, family, teacher):
        """(posterior success rate, scored episodes) of `teacher` on `family`: its family record shrunk towards its overall
        rate (ROUTE_PRIOR episodes of weight; teachers without any record start at ROUTE_DEFAULT)."""
        model = teacher.model_id
        overall = self.overall.get(model, (0, 0))
        prior = (overall[0] + ROUTE_DEFAULT * ROUTE_PRIOR) / (overall[1] + ROUTE_PRIOR)
        s, n = self.stats.get((family, model), (0, 0))
        return (s + prior * ROUTE_PRIOR) / (n + ROUTE_PRIOR), n

    def _eligible(self, index, teacher):
        """Whether `teacher` may take the task at all: not tried by it here, enough teachers failed it for its
        `min_failed`, and its family yield is not hopeless (below --route-floor over at least --route-min-scored)."""
        task = self.tasks[index]
        if not teacher.enabled or teacher.model_id in self.tried.get(index, ()):
            return False
        if len(task.get('failed_by') or ()) < teacher.min_failed:
            return False
        rate, n = self._estimate(task['row'].get('family') or '?', teacher)
        return not (n >= self.args.route_min_scored and rate < self.args.route_floor)

    def _passes(self, index, teacher):
        """`teacher` leaves the task to another eligible teacher first: it failed the task (an earlier campaign) and
        another did not, or another's family yield is at least --route-ratio times its own."""
        failed = set(self.tasks[index].get('failed_by') or ())
        others = [t for t in self.teachers if t is not teacher and self._eligible(index, t)]
        if teacher.model_id in failed and any(t.model_id not in failed for t in others):
            return True
        if not self.args.route_ratio:
            return False
        family = self.tasks[index]['row'].get('family') or '?'
        mine = self._estimate(family, teacher)[0]
        return any(self._estimate(family, t)[0] >= self.args.route_ratio * mine for t in others)

    def _no_verdict(self, record):
        """Every episode ended unresolved before its first model turn or as an infrastructure error from exhausted provider
        retries (e.g. a 429 mid-episode; a model-call budget stop is the teacher's own verdict): the provider failed (rate limit, outage), not the teacher. Such a record is no teacher verdict; it goes to deferred.jsonl and the task is retried.
        A harness infrastructure error (e.g. the collector's record digest failing while ts-host/dist is rebuilt) is
        no verdict either: it would count against the teacher's family yield and use up one of the task's --attempts."""
        return 'unsupported' not in record and all(
            e['outcome'] == 'infrastructure_error' or (e['outcome'] == 'unresolved' and e['turns'] == 0)
            for e in record['episodes'])

    def _claim(self, index, teacher):
        """Several teachers share one WORK and steal tasks from one ordered list: WORK/claims/tINDEX names the process
        running a task (state running) or the one that recorded it (state done). A dead process's claim is taken over."""
        path = self.work / 'claims' / f't{index:06d}'
        mine = json.dumps({'pid': os.getpid(), 'teacher': teacher.model_id, 'state': 'running'})
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            try:
                held = json.loads(path.read_text() or '{}')
            except (OSError, ValueError):
                return False   # being written by its claimant
            if held.get('state') == 'done' and not (index in self.tried and not self._finished(index)):
                self.done.add(index)
                return False
            if held.get('state') != 'done':   # a done claim with a verdict here goes on to another teacher (--attempts)
                try:
                    os.kill(int(held.get('pid', 0)), 0)
                    return False   # its claimant is alive
                except (ProcessLookupError, ValueError):
                    pass
                except PermissionError:
                    return False
            tmp = path.with_suffix(f'.{os.getpid()}')
            tmp.write_text(mine)
            os.replace(tmp, path)
            time.sleep(0.2)
            return json.loads(path.read_text() or '{}').get('pid') == os.getpid()
        with os.fdopen(fd, 'w') as handle:
            handle.write(mine)
        return True

    def _release(self, index, state, teacher):
        path = self.work / 'claims' / f't{index:06d}'
        if state == 'done':
            tmp = path.with_suffix(f'.{os.getpid()}')
            tmp.write_text(json.dumps({'pid': os.getpid(), 'teacher': teacher.model_id, 'state': 'done'}))
            os.replace(tmp, path)
        else:
            path.unlink(missing_ok=True)

    def _take(self, teacher):
        """The first task of the ordered queue (the first --limit tasks; all without it) this teacher may take: not
        done, not running, not waiting for its provider retry, eligible (``_eligible``), not a HARD_TIERS task for a weak
        teacher, and not one it leaves to a better-placed teacher (``_passes``); when only such passed tasks are left
        for it, the first of them. A task a teacher failed here stays at its place for the next teacher."""
        while time.time() < teacher.cool_until and not self.stop.is_set():
            self.stop.wait(5)
        with self.lock:
            if not teacher.enabled:
                return None
            end = min(len(self.tasks), self.args.limit or len(self.tasks))
            now, weak, passed = time.time(), self.weak(teacher), None
            for index in range(end):
                if (index in self.done or index in self.active or self.retry.get(index, 0) > now
                        or index % self.args.shard[1] != self.args.shard[0]):
                    continue
                if weak and self.tasks[index].get('base_rank') in HARD_TIERS:
                    continue
                if not self._eligible(index, teacher):
                    continue
                if self._passes(index, teacher):
                    passed = index if passed is None else passed
                    continue
                if self._claim(index, teacher):
                    return self._took(index)
            if passed is not None and self._claim(passed, teacher):
                return self._took(passed)
            return None

    def _took(self, index):
        self.active.add(index)
        self.retry.pop(index, None)
        return index

    def run_task(self, index, teacher):
        from natlang_neuralese.common.jsonio import read_jsonl
        from natlang_neuralese.rl import task_rollouts
        a, task = self.args, self.tasks[index]
        row = task['row']
        attempt = self.recorded.get(index, 0)   # a later attempt keeps the earlier ones' episodes (voided included)
        directory = self.work / 'episodes' / (f't{index:06d}' if not attempt else f't{index:06d}-a{attempt}')
        if directory.exists():   # an unscored attempt (stopped mid-episode) is redone whole
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        (directory / 'cases.jsonl').write_text(json.dumps(row) + '\n')
        task_rollouts.prepare(SimpleNamespace(
            cases=directory / 'cases.jsonl', out=directory, rows=1, offset=0, group=max(2, a.samples), temperature=a.temperature,
            case_seconds=a.case_seconds, max_model_requests=a.max_model_requests, max_turns=a.max_turns,
            seed_base=a.seed_base + 100 * index, root_mode=a.root_mode, match_seeds_from=None,
            policy_checkpoint_sha256=None))
        # prepare builds GRPO groups (>= 2 samples); a teacher may want one: the queue keeps the first `samples`
        entries = list(read_jsonl(directory / 'queue.jsonl'))[:a.samples]
        (directory / 'queue.jsonl').write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in entries))
        started = time.time()
        for entry in entries:
            entry.update(prompt_rendering='full', context_tokens=a.context_tokens, request_retries=a.request_retries)
            key = entry['key']
            queue_file = directory / f'{key}.queue.jsonl'
            queue_file.write_text(json.dumps(entry, sort_keys=True) + '\n')
            command = queue_command(teacher, self.work, queue_file, directory / f'{key}.journal.jsonl')
            with (directory / f'{key}.log').open('a') as handle:
                try:
                    env = episode_env(teacher, self.env)
                except Exception as error:   # no key: the request fails as a provider failure (deferred, no verdict)
                    handle.write(json.dumps({'event': 'api key command failed', 'error': repr(error)[:300]}) + '\n')
                    env = self.env
                process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                while process.poll() is None:
                    if self.stop.wait(2):
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait()
                        return
            if process.returncode < 0:
                return   # killed: no verdict; redone on the next launch
        with (directory / 'journal.jsonl').open('w') as merged:
            for path in sorted(directory.glob('*.journal.jsonl')):
                if path.name != 'journal.jsonl':
                    merged.write(path.read_text())
        report = task_rollouts.score(directory, directory / 'rollouts.jsonl', neuralese_bonus=0., efficiency_weight=0.)
        rollouts = list(read_jsonl(directory / 'rollouts.jsonl'))
        record = {'index': index, 'task_id': row['id'], 'family': row.get('family'), 'base_rank': task['base_rank'],
                  'root_mode': a.root_mode,
                  'teacher': teacher_identity(teacher),
                  'seconds': round(time.time() - started, 1),
                  'episodes': [{'sample': r['sample'], 'outcome': r['outcome'],
                                'reward': (r.get('reward') or {}).get('quality'),
                                'graded': (r.get('reward') or {}).get('graded'), 'turns': len(r['turns']),
                                'completion_tokens': sum((t.get('usage') or {}).get('completion_tokens') or 0
                                                         for t in r['turns'])} for r in rollouts],
                  'rollouts': str(directory / 'rollouts.jsonl')}
        logs = ''.join(path.read_text(errors='replace') for path in directory.glob('*.log'))
        record['provider_failure'] = PROVIDER_FAILURE in logs
        if UNSUPPORTED_DECIDE in logs:   # a deterministic harness limit, not a provider failure: recorded, never retried
            record['unsupported'] = 'decide-native-capture'
        with self.lock:
            if self._no_verdict(record):
                refusal = next((m for m in PERMANENT_PROVIDER_FAILURES if m in logs), None)
                if refusal and teacher.enabled:
                    teacher.enabled = False
                    print(json.dumps({'event': 'teacher retired', 'teacher': teacher.model_id, 'reason': refusal}),
                          flush=True)
                self.attempts[index] = self.attempts.get(index, 0) + 1
                with (self.work / 'deferred.jsonl').open('a') as handle:
                    handle.write(json.dumps({**record, 'attempt': self.attempts[index]}) + '\n')
                provider = record.get('provider_failure') or any(e['outcome'] == 'unresolved' for e in record['episodes'])
                if provider and not refusal:   # the harness failing is no reason to back the teacher off
                    teacher.failures += 1
                    teacher.cool_until = max(teacher.cool_until, time.time() + min(
                        3600., self.args.provider_cooldown * 2 ** (teacher.failures - 1)))
                if self.attempts[index] < 3:
                    self.retry[index] = time.time() + 600
                else:
                    self.done.add(index)     # three provider failures: no further attempt in this launch
                self._release(index, 'deferred', teacher)
                print(json.dumps({'event': 'task deferred', 'index': index, 'attempt': self.attempts[index]}), flush=True)
                return report
            teacher.failures = 0
            with self.manifest.open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            record_episode_files(self.work, directory)
            self._note_verdict(record)
            self.recorded[index] = self.recorded.get(index, 0) + 1
            if self._finished(index):
                self.done.add(index)
                self._release(index, 'done', teacher)
            else:   # another teacher tries it (it now counts this one in failed_by), at its place in the queue
                self._release(index, 'retry', teacher)
        print(json.dumps({'event': 'task scored', **{k: record[k] for k in ('index', 'task_id', 'seconds', 'episodes')}}),
              flush=True)
        return report

    def slot(self, teacher):
        while not self.stop.is_set() and teacher.enabled:
            if (self.work / 'DRAIN').exists():   # a newer queue replaces this one: running episodes finish, then exit
                return
            if (self.work / 'PAUSE').exists() or mem_available_gb() < self.args.pause_below_available_gb:
                self.stop.wait(10)
                continue
            index = self._take(teacher)
            if index is None:
                with self.lock:
                    pending = self.running or self.retry
                if not pending:
                    return
                self.stop.wait(30)   # a running task may still come back for another teacher
                continue
            with self.lock:
                self.running += 1
            try:
                self.run_task(index, teacher)
            except Exception as error:   # one task's failure never stops the generation
                self._release(index, 'failed', teacher)
                print(json.dumps({'event': 'task failed', 'index': index, 'error': repr(error)[:400]}), flush=True)
            finally:
                with self.lock:
                    self.running -= 1
                    self.active.discard(index)


def validate_pool_resume(work, args):
    """An explicit source selection cannot silently reuse another campaign's task queue."""
    from natlang_neuralese.common.hashing import sha256_file_hex
    receipt_path = Path(work) / 'inputs.json'
    if not receipt_path.exists():
        # Historical default-pool runs already in flight predate this receipt.
        # Preserve that student route; new explicit source/root requests need
        # their own authenticated queue, rather than a hand-written tasks.jsonl.
        if args.pool or args.held_manifest or args.root_mode != 'student':
            raise ValueError('existing teacher queue has no source receipt; use a fresh --work for explicit pools/root mode')
        return
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('schema') != 'natlang.teacher-pool-inputs/1' or receipt.get('root_mode') != args.root_mode:
        raise ValueError('teacher pool root mode changed; use a new --work')
    for name, field in (('tasks.jsonl', 'tasks_sha256'), ('pool.json', 'pool_sha256')):
        if receipt.get(field) != sha256_file_hex(Path(work) / name):
            raise ValueError(f'teacher pool input changed: {name}')
    pool = json.loads((Path(work) / 'pool.json').read_text())
    if args.pool:
        requested = [(str(Path(p).resolve()), sha256_file_hex(Path(p))) for p in args.pool]
        recorded = [(str(Path(p['path']).resolve()), p['sha256']) for p in pool['pools']]
        if requested != recorded:
            raise ValueError('requested source pools differ from this campaign; use a new --work')
    if args.held_manifest and (Path(args.held_manifest).resolve() != Path(pool['held_manifest']).resolve() or
                              sha256_file_hex(Path(args.held_manifest)) != pool['held_manifest_sha256']):
        raise ValueError('requested held exclusion manifest differs from this campaign; use a new --work')


def campaign_id(work):
    return data_registry.dataset_id(work, 'teacher-gen')


def record_campaign(work, args, teachers):
    """Registry records at every launch: the task sources the campaign selected from (WORK/pool.json, the held closure,
    rates, coverage) as a consumption, and the campaign itself as a growing dataset (each scored task adds its episode
    files: ``record_episode_files``). Data no registry entry holds stops the launch."""
    pool = json.loads((work / 'pool.json').read_text()) if (work / 'pool.json').exists() else {}
    run = data_registry.run_name('teacher-gen', work)
    data_registry.consume([p['path'] for p in pool.get('pools') or []], run=run, how='teacher task pool',
                          amount=pool.get('eligible'))
    data_registry.consume([x for x in [pool.get('held_manifest'), args.base_rates, *args.covered_by, *args.priority_pool,
                                       *[e['path'] for e in pool.get('excluded_errata') or []]] if x],
                          run=run, how='teacher task selection (held closure, rates, coverage, errata)')
    try:
        data_registry.declare_dataset(
            campaign_id(work), work, kind='teacher-demos',
            format='teacher-gen campaign: manifest.jsonl (one scored task: teacher, episodes, outcomes), '
                   'episodes/tNNNNNN/{cases,rollouts}.jsonl, pool.json, inputs.json, launches.jsonl',
            inputs=[p['path'] for p in pool.get('pools') or []],
            license='teacher outputs of ' + ', '.join(sorted({f"{t.model_id} ({t.provider or t.endpoint})" for t in teachers}))
                    + '; tasks keep their pools\' licenses')
        data_registry.record_files(campaign_id(work), [p for p in (work / name for name in (
            'pool.json', 'inputs.json', 'tasks.jsonl', 'voided.jsonl')) if p.exists()], work)
    except Exception as error:
        print(json.dumps({'event': 'data registry error', 'error': repr(error)[:300]}), flush=True)


def record_episode_files(work, directory):
    try:
        data_registry.record_files(campaign_id(work), [p for p in (directory / n for n in (
            'cases.jsonl', 'rollouts.jsonl', 'score.json')) if p.exists()], work)
    except Exception as error:
        print(json.dumps({'event': 'data registry error', 'error': repr(error)[:300]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--work', required=True)
    p.add_argument('--pool', action='append', default=[],
                   help='explicit canonical ProgramRecord JSONL pool; repeatable, replaces the default GRPO pools')
    p.add_argument('--held-manifest', help='held exclusion manifest (default: the registry role held-eval, including predecessor closure)')
    p.add_argument('--root-mode', choices=('student', 'authored'), default='student',
                   help='student executes the real program entry; authored requires its actual reference root eval')
    p.add_argument('--teacher', action='append', default=[], type=json.loads, metavar='JSON',
                   help='one teacher of several in this process (the task list is loaded once): {"model_id", "endpoint", '
                        '"chat_completions_url", "api_key_env", "slots", "chat_config"} or {"model_id", "provider", '
                        '"reasoning_effort", "slots"} (a natlang Pi provider, e.g. openai-codex); repeatable. Without it the '
                        'single-teacher flags below describe the one teacher')
    p.add_argument('--endpoint', help='teacher server base URL (OpenAI-compatible)')
    p.add_argument('--chat-completions-url', help='exact completion URL (default ENDPOINT/v1/chat/completions)')
    p.add_argument('--model-id')
    p.add_argument('--api-key-env', help='environment variable holding the API key (name only)')
    p.add_argument('--chat-config', type=json.loads, help='JSON request controls passed unchanged (default: top_p 0.95)')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--slots', type=int, default=2)
    p.add_argument('--request-retries', type=int, default=8,
                   help='same-request retries after a provider failure (429, 5xx), with backoff honouring Retry-After')
    p.add_argument('--provider-cooldown', type=float, default=120,
                   help='seconds all of a teacher\'s slots wait before taking a new task after an episode the provider failed '
                        '(doubling per consecutive failure, at most an hour)')
    p.add_argument('--pause-below-available-gb', type=float, default=0,
                   help='take no new task while MemAvailable is below this (a co-located learner phase needs the memory)')
    p.add_argument('--samples', type=int, default=1, help='episodes per task')
    p.add_argument('--limit', type=int, default=0, help='stop after the first N ordered tasks (0: all)')
    p.add_argument('--shard', type=int, nargs=2, default=(0, 1), metavar=('K', 'N'),
                   help='take only tasks with index %% N == K (several teachers share one ordered list)')
    p.add_argument('--context-tokens', type=int, default=65536)
    p.add_argument('--case-seconds', type=int, default=3600)
    p.add_argument('--max-model-requests', type=int, default=None, help='default: EPISODE_MODEL_CALLS (40)')
    p.add_argument('--max-turns', type=int, default=128)
    p.add_argument('--pass-band', type=float, nargs=2, default=(0.1, 0.9))
    p.add_argument('--base-rates', default=BASE_RATES)
    p.add_argument('--exclude-family-regex', default=r'^ext_codebase-(py|natlang)_codebase_',
                   help="families left out (grpo's default; '(?!)' keeps every family, as a run with the codebase repos does)")
    p.add_argument('--covered-by', action='append', default=[], metavar='WORK',
                   help='gap campaign: earlier teacher_gen WORKs; tasks with a teacher success there are left out, the rest '
                        'ordered by GAP_TIERS, and a task a teacher failed there goes to another teacher first (repeatable)')
    p.add_argument('--student-groups', action='append', default=[], metavar='GLOB',
                   help='gap campaign: GRPO group directories (glob) whose student samples add to --base-rates (repeatable)')
    p.add_argument('--priority-pool', action='append', default=[], metavar='POOL',
                   help='gap campaign: rows of these pools (new task supply) are kept whatever their student rate (repeatable)')
    p.add_argument('--rare-rate', type=float, default=0.25,
                   help='gap campaign: a task the student solves at most this often is in the first tier (student_rare)')
    p.add_argument('--student-rates', action='append', default=[], metavar='FILE',
                   help='gap campaign: rates files (expert_iteration select tasks.rates.json: every collection) over the base '
                        'rates, before --student-groups (repeatable; later files win)')
    p.add_argument('--frontier-rates', action='append', default=[], metavar='FRONTIER',
                   help="gap campaign: the newest round frontier.json; its tasks' rates win over every other measurement")
    p.add_argument('--weak-yield', type=float, default=0.25,
                   help="a teacher whose posterior yield on fresh (teacher_untried) tasks is below this leaves the "
                        "student-rare tiers to the stronger teachers (0: off)")
    p.add_argument('--attempts', type=int, default=2,
                   help='distinct teachers of this campaign that try a task until one succeeds (1: one verdict per task)')
    p.add_argument('--route-ratio', type=float, default=2.0,
                   help='a teacher leaves a task to another one whose yield on its family is at least this many times its '
                        'own; it takes it itself once only such tasks are left for it (0: no family routing)')
    p.add_argument('--route-floor', type=float, default=0.03,
                   help='a teacher whose family yield is below this (over at least --route-min-scored episodes) never '
                        'takes that family\'s tasks')
    p.add_argument('--route-min-scored', type=int, default=30)
    p.add_argument('--seed', type=int, default=20261115)
    p.add_argument('--seed-base', type=int, default=50_000_000)
    args = p.parse_args()
    from natlang_neuralese.rl.held_eval import EPISODE_MODEL_CALLS
    args.max_model_requests = args.max_model_requests or EPISODE_MODEL_CALLS
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    from natlang_neuralese.common.code_provenance import code_provenance
    if not (work / 'tasks.jsonl').exists():
        import grpo
        pool_args = SimpleNamespace(pool=args.pool or grpo.default_pools(), held_manifest=args.held_manifest or grpo.default_held(),
                                    exclude_manifest=[], task_ids=None,
                                    exclude_family_regex=args.exclude_family_regex, exclude_ids=[],
                                    no_default_errata=False, seed=args.seed, pass_rates=args.base_rates,
                                    pass_band=args.pass_band, pass_min_scored=1, explore=None,
                                    max_model_requests=args.max_model_requests, case_seconds=args.case_seconds)
        rows, excluded = grpo.load_tasks(pool_args, work)   # also writes WORK/pool.json (pool shas, held closure)
        rates = json.loads(Path(args.base_rates).read_text()) if args.base_rates else {}
        from natlang_neuralese.common.hashing import sha256_file_hex
        selection = {}
        if args.covered_by:
            from natlang_neuralese.common.jsonio import read_jsonl
            coverage = teacher_coverage(args.covered_by)
            student = student_rates(rates, args.student_groups, args.student_rates, args.frontier_rates)
            priority = {row['id'] for path in args.priority_pool for row in read_jsonl(path)}
            rank, tiers = gap_rank(coverage, student, priority, args.rare_rate, args.pass_band[1]), GAP_TIERS
            selection = {'covered_by': [{'work': str(Path(w).resolve()),
                                         'manifest_sha256': sha256_file_hex(Path(w) / 'manifest.jsonl')} for w in args.covered_by],
                         'student_groups': args.student_groups, 'student_measured_tasks': len(student),
                         'student_rates': [{'path': str(p), 'sha256': sha256_file_hex(Path(p))}
                                           for p in [*args.student_rates, *args.frontier_rates]],
                         'solved_rate': args.pass_band[1],
                         'priority_pools': [{'path': str(p), 'sha256': sha256_file_hex(Path(p))} for p in args.priority_pool],
                         'rare_rate': args.rare_rate, 'tiers': GAP_TIERS}
        else:
            coverage, rank, tiers = {}, base_rank(rates, args.pass_band), BASE_TIERS
        ordered = order_tasks(rows, rank, args.seed,
                              sub=lambda row: len((coverage.get(row['id']) or {}).get('failed_by') or ()))
        counts = {}
        with (work / 'tasks.jsonl.tmp').open('w') as handle:
            for row in ordered:
                tier = tiers[rank(row)]
                counts[tier] = counts.get(tier, 0) + 1
                failed = sorted((coverage.get(row['id']) or {}).get('failed_by') or ())
                handle.write(json.dumps({'row': row, 'base_rank': tier, **({'failed_by': failed} if failed else {})}) + '\n')
        os.replace(work / 'tasks.jsonl.tmp', work / 'tasks.jsonl')
        (work / 'inputs.json').write_text(json.dumps({'schema': 'natlang.teacher-pool-inputs/1',
            'root_mode': args.root_mode, 'tasks_sha256': sha256_file_hex(work / 'tasks.jsonl'),
            'pool_sha256': sha256_file_hex(work / 'pool.json'), 'eligible': len(rows), 'tiers': counts,
            **({'selection': selection} if selection else {})}, indent=2) + '\n')
    validate_pool_resume(work, args)
    specs = args.teacher or [{'model_id': args.model_id, 'endpoint': args.endpoint, 'api_key_env': args.api_key_env,
                              'chat_completions_url': args.chat_completions_url, 'slots': args.slots,
                              'chat_config': args.chat_config}]
    teachers = []
    for spec in specs:
        try:
            teacher = make_teacher({k: v for k, v in spec.items() if v is not None})
        except ValueError as error:
            p.error(str(error))
        if not teacher.provider:
            chat_config_path(work, teacher.model_id).write_text(json.dumps(teacher.chat_config) + '\n')
        teachers.append(teacher)
    with (work / 'launches.jsonl').open('a') as handle:
        handle.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'argv': sys.argv,
                                 'code': code_provenance(),
                                 'root_mode': args.root_mode,
                                 'teachers': [{k: v for k, v in vars(t).items() if k not in ('cool_until', 'failures')} for t in teachers]}) + '\n')
    record_campaign(work, args, teachers)
    generator = Generator(args, teachers)
    for teacher in teachers:   # the routing evidence of this launch, in its log
        rate, n = generator.strength(teacher)
        print(json.dumps({'event': 'teacher routing', 'teacher': teacher.model_id, 'fresh_yield': round(rate, 3),
                          'fresh_scored': n, 'weak': generator.weak(teacher), 'min_failed': teacher.min_failed,
                          'slots': teacher.slots}), flush=True)

    def stop(signum, frame):
        generator.stop.set()
    signal.signal(signal.SIGTERM, stop)
    threads = [threading.Thread(target=generator.slot, args=(teacher,), daemon=True)
               for teacher in teachers for _ in range(teacher.slots)]
    for thread in threads:
        thread.start()
    for thread in threads:
        while thread.is_alive():
            thread.join(5)


if __name__ == '__main__':
    main()
