#!/usr/bin/env python3
"""Daily check, qualification and adoption of free LLM APIs as teachers (DECISIONS 2026-11-24 "free teacher providers",
2026-11-25 "free teacher providers, round 2").

Free models on the hosted APIs rotate in and out (promotional `:free` models, new NIM catalogue entries, AI Studio
previews). The daily run (timer ``natlang-teacher-free-models``) takes every candidate through:

  1. check    list the free models of every provider (PROVIDER_SPECS) and append each newly seen one to
              ``ops/teacher-free-models/candidates.jsonl`` (``status: new``); a provider without our key is recorded
              with the owner's signup step (``signups`` prints them)
  2. probe    one tool-call request per new candidate after a name screen (chat models, >= 30B when the name gives a
              size, a free tier that holds one request): ``probed``, ``rejected`` (permanent refusal: no free quota,
              client-only, gone), ``needs-key``, or left ``new`` (transient: 503, 429 with a retry)
  3. adopt    decide finished qualifications: yield >= 25% over >= 12 verdicts, <= 900 s per episode, <= half
              deferred; a qualified model enters ``ops/ei/teachers.json`` (the campaign reads it at each launch) with
              slots sized to its provider's limits and its quota group's remaining slots (``qualified`` when the group
              is full; a model already a teacher through another gateway is not added twice)
  4. qualify  start teacher_gen with one slot on the 20-task qualification set (``ops/ei/teacher-qual-20261124``:
              10 Luna-solved, 10 Luna-failed tasks) for probed candidates, as low-priority ledger units, at most
              MAX_QUALIFYING at once and one per quota group, at the teacher campaign's pin

Providers: OpenRouter (``:free`` or zero-priced; 1,000 requests/day for the account), the Kilo gateway (``:free``/
``-free``; keyless allowance spent, an account key carries on), NVIDIA NIM (one account-wide rate limit), Google AI
Studio (per-model per-project free quotas), Z.ai (``*-flash``), Groq (8k tokens/min: screened out), OpenCode Zen (free
models only through the official client: rejected by the probe), and the providers that need an owner signup or have no
free tier (Cerebras, SambaNova, Chutes, Hugging Face, Mistral, Cloudflare Workers AI, LLM7, ModelScope; public
listings are watched for zero-priced models). Keys come from ``~/.config/natlang/*.env`` and are never printed.

  teacher_free_models.py daily [--pin DIR]   # check, probe, adopt, qualify (the timer's command)
  teacher_free_models.py check | probe | adopt | qualify [--pin DIR] | signups
  teacher_free_models.py add PROVIDER MODEL  # enqueue a model the listing already knew
  teacher_free_models.py mark MODEL STATUS [--note TEXT]
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'training/neuralese'), str(HERE.parent)]

from natlang_neuralese.common.paths import resolve_str  # noqa: E402

OUT = Path(resolve_str('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'teacher-free-models'))
ENV_DIR = Path.home() / '.config/natlang'
UA = {'User-Agent': 'natlang-teacher-free-models/1'}
OPEN = 'new'


def env_key(name, file):
    """A key from the environment or ~/.config/natlang/FILE.env (never printed)."""
    if os.environ.get(name):
        return os.environ[name]
    path = ENV_DIR / f'{file}.env'
    if path.exists():
        for line in path.read_text().splitlines():
            key, _, value = line.partition('=')
            if key.strip().removeprefix('export ').strip() == name:
                return value.strip().strip('"').strip("'")
    return None


def get_json(url, headers=None, timeout=60):
    request = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def bearer(name, file):
    key = env_key(name, file)
    return {'Authorization': f'Bearer {key}'} if key else None


def tools_and_reasoning(model):
    params = model.get('supported_parameters') or []
    return {'tools': 'tools' in params, 'reasoning': 'reasoning' in params}


def openrouter_free(model):
    pricing = model.get('pricing') or {}
    return model['id'].endswith(':free') or (pricing.get('prompt') == '0' and pricing.get('completion') == '0')


def list_openrouter():
    return {m['id']: {'context': m.get('context_length'), **tools_and_reasoning(m)}
            for m in get_json('https://openrouter.ai/api/v1/models')['data'] if openrouter_free(m)}


def list_kilo():
    data = get_json('https://ai-gateway.kilo.ai/api/v1/models')
    return {m['id']: {'context': m.get('context_length'), **tools_and_reasoning(m)} for m in data.get('data', data)
            if m['id'].endswith(':free') or m['id'].endswith('-free') or m['id'] == 'kilo-auto/free'}


def list_nim():
    headers = bearer('NVIDIA_API_KEY', 'nvidia')
    if headers is None:
        raise RuntimeError('no NVIDIA_API_KEY')
    skip = ('embed', 'safety', 'guard', 'clip', 'riva', 'retriever', 'detector', 'pii', 'reward', 'parse', 'ocr')
    return {m['id']: {} for m in get_json('https://integrate.api.nvidia.com/v1/models', headers)['data']
            if not any(s in m['id'] for s in skip)}


def list_ai_studio():
    key = env_key('GEMINI_API_KEY', 'gemini')
    if not key:
        raise RuntimeError('no GEMINI_API_KEY')
    data = get_json('https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000', {'x-goog-api-key': key})
    skip = ('tts', 'image', 'embedding', 'robotics', 'computer-use', 'transcribe', 'live', 'lyria', 'banana', 'research')
    return {m['name'].split('/', 1)[1]: {'thinking': bool(m.get('thinking')), 'context': m.get('inputTokenLimit')}
            for m in data['models'] if 'generateContent' in (m.get('supportedGenerationMethods') or [])
            and not any(s in m['name'] for s in skip)}


def list_zai():
    headers = bearer('ZAI_API_KEY', 'zai')
    if headers is None:
        raise RuntimeError('no ZAI_API_KEY')
    return {m['id']: {} for m in get_json('https://api.z.ai/api/paas/v4/models', headers)['data'] if 'flash' in m['id']}


def list_groq():
    headers = bearer('GROQ_API_KEY', 'groq')
    if headers is None:
        raise RuntimeError('no GROQ_API_KEY')
    skip = ('whisper', 'orpheus', 'guard', 'safeguard', 'allam')
    return {m['id']: {'context': m.get('context_window')} for m in get_json('https://api.groq.com/openai/v1/models', headers)['data']
            if not any(s in m['id'] for s in skip)}


def list_opencode():
    """OpenCode Zen's free models (``*-free``, ``big-pickle``). Its direct API answers them with 429
    FreeUsageLimitError ("Free models work in OpenCode"): free only through the official OpenCode client, which the
    probe records (plans/OPENCODE_GENERATION.md: the client's identity is never spoofed)."""
    headers = bearer('OPENCODE_API_KEY', 'opencode')
    if headers is None:
        raise RuntimeError('no OPENCODE_API_KEY')
    return {m['id']: {} for m in get_json('https://opencode.ai/zen/v1/models', headers)['data']
            if m['id'].endswith('-free') or m['id'] == 'big-pickle'}


CHAT_SKIP = ('embed', 'image', 'video', 'tts', 'whisper', 'voxtral', 'guard', 'rerank', 'ocr', 'moderation', 'flux',
             'banana', 'live', 'audio', 'vision-exp')


def chat_models(ids):
    return {i: {} for i in ids if not any(x in i.lower() for x in CHAT_SKIP)}


def list_keyed(provider):
    """A provider of PROVIDER_SPECS listed by its ``models`` URL: its free chat models (``free``: how a listing marks
    them; None: every chat model is free for a free account). A public listing needs no key (``public``); a keyed one
    without a key raises MissingKey (the owner's signup)."""
    spec = PROVIDER_SPECS[provider]
    key = env_key(spec['key'], spec['file'])
    if not key and not spec.get('public'):
        raise MissingKey(provider)
    url = spec['models'].format(**{v: env_key(v, spec['file']) or '' for v in spec.get('url_env', ())})
    data = get_json(url, {'Authorization': f'Bearer {key}'} if key else {})
    rows = data.get('data', data.get('result', data)) if isinstance(data, dict) else data
    rows = [m for m in rows if isinstance(m, dict) and (m.get('id') or m.get('name'))]
    free = FREE_FILTERS.get(spec.get('free'), lambda m: True)
    return chat_models([m.get('id') or m.get('name') for m in rows if free(m)])


def zero_price(model):
    pricing = model.get('pricing') or {}
    values = [pricing.get(k) for k in ('prompt', 'completion', 'input', 'output') if pricing.get(k) is not None]
    return bool(values) and all(float(v) == 0 for v in values)


FREE_FILTERS = {'zero-price': zero_price,
                'hf-is-free': lambda m: any(p.get('is_free') and p.get('supports_tools') for p in m.get('providers') or []),
                'llm7-tier': lambda m: m.get('tier') != 'pro' and m.get('model_type', 'chat') == 'chat'}


class MissingKey(RuntimeError):
    """The provider needs a key we do not have: an owner signup (SIGNUPS)."""


# Every provider the check knows: how to reach it, its key, the quota its free models share (``group``: models of one
# group draw on one limit, so the group's teacher slots are capped together; ``{model}`` makes it per model) and the
# documented or measured free limits that size a teacher's slots (rpm, rpd: requests; tpm: tokens per minute; slots:
# the group's slot cap). ``signup``: the owner's step when the key is missing.
PROVIDER_SPECS = {
    'openrouter': {'endpoint': 'https://openrouter.ai/api/v1', 'key': 'OPENROUTER_API_KEY', 'file': 'openrouter',
                   'group': 'openrouter-free', 'rpm': 20, 'rpd': 1000, 'slots': 1},
    'kilo': {'endpoint': 'https://ai-gateway.kilo.ai/api/v1', 'key': 'KILO_API_KEY', 'file': 'kilo', 'group': 'kilo',
             'slots': 2, 'signup': 'sign in at https://app.kilo.ai (GitHub or Google account, no card), create an API '
             'key under Profile > API keys, write KILO_API_KEY=... to ~/.config/natlang/kilo.env'},
    'nvidia-nim': {'endpoint': 'https://integrate.api.nvidia.com/v1', 'key': 'NVIDIA_API_KEY', 'file': 'nvidia',
                   'group': 'nim', 'rpm': 40, 'slots': 3},   # one account-wide limit (429 at ~12 slots)
    'ai-studio': {'endpoint': 'https://generativelanguage.googleapis.com/v1beta/openai', 'key': 'GEMINI_API_KEY',
                  'file': 'gemini', 'group': 'ai-studio:{model}', 'rpm': 10, 'slots': 1},   # per-model, per-project quotas
    'zai': {'endpoint': 'https://api.z.ai/api/paas/v4', 'key': 'ZAI_API_KEY', 'file': 'zai', 'group': 'zai', 'slots': 1},
    'groq': {'endpoint': 'https://api.groq.com/openai/v1', 'key': 'GROQ_API_KEY', 'file': 'groq', 'group': 'groq:{model}',
             'tpm': 8000, 'slots': 1},
    'opencode-zen': {'endpoint': 'https://opencode.ai/zen/v1', 'key': 'OPENCODE_API_KEY', 'file': 'opencode',
                     'group': 'opencode', 'slots': 1},
    # 2026-11-25 (web research, DECISIONS "free teacher providers, round 2"): Cerebras' free path is a one-time $5
    # trial that needs a verified payment method (gpt-oss-120b 5 RPM, 30k TPM, 1M tokens/day); SambaNova, Chutes and
    # Hugging Face (free users) have no free API tier: their public listings are watched for zero-priced models;
    # Mistral's Free mode has no API access (our key: 401); GitHub Models retired 2026-07-30 (not listed)
    'cerebras': {'endpoint': 'https://api.cerebras.ai/v1', 'models': 'https://api.cerebras.ai/v1/models',
                 'key': 'CEREBRAS_API_KEY', 'file': 'cerebras', 'group': 'cerebras:{model}', 'rpm': 5, 'tpm': 30000,
                 'slots': 1,
                 'signup': 'sign up at https://cloud.cerebras.ai (needs a verified payment method; $5 one-time trial '
                 'credit, 30 days), API Keys > Generate, CEREBRAS_API_KEY=... in ~/.config/natlang/cerebras.env'},
    'sambanova': {'endpoint': 'https://api.sambanova.ai/v1', 'models': 'https://api.sambanova.ai/v1/models',
                  'public': True, 'free': 'zero-price', 'key': 'SAMBANOVA_API_KEY', 'file': 'sambanova',
                  'group': 'sambanova:{model}', 'slots': 1,
                  'signup': 'no free tier (credits must be bought): https://cloud.sambanova.ai, SAMBANOVA_API_KEY=... '
                  'in ~/.config/natlang/sambanova.env'},
    'chutes': {'endpoint': 'https://llm.chutes.ai/v1', 'models': 'https://llm.chutes.ai/v1/models', 'public': True,
               'free': 'zero-price', 'key': 'CHUTES_API_KEY', 'file': 'chutes', 'group': 'chutes', 'slots': 1,
               'signup': 'no free tier (pay as you go): https://chutes.ai, CHUTES_API_KEY=... in ~/.config/natlang/chutes.env'},
    'huggingface': {'endpoint': 'https://router.huggingface.co/v1', 'models': 'https://router.huggingface.co/v1/models',
                    'public': True, 'free': 'hf-is-free', 'key': 'HF_TOKEN', 'file': 'huggingface',
                    'group': 'huggingface', 'slots': 1,
                    'signup': 'free users get no Inference Providers credit (PRO: $2/month): https://huggingface.co/join, '
                    'a fine-grained token with "Make calls to Inference Providers", HF_TOKEN=... in '
                    '~/.config/natlang/huggingface.env'},
    'mistral': {'endpoint': 'https://api.mistral.ai/v1', 'models': 'https://api.mistral.ai/v1/models',
                'key': 'MISTRAL_API_KEY', 'file': 'mistral', 'group': 'mistral', 'rpm': 60, 'slots': 1,
                'signup': 'the key in mistral.env is refused (401) and Mistral\'s Free mode has no API access: a paid '
                'plan at https://console.mistral.ai, then replace MISTRAL_API_KEY in ~/.config/natlang/mistral.env'},
    'cloudflare': {'endpoint': 'https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1',
                   'models': 'https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/models/search?task=Text%20Generation',
                   'url_env': ('CLOUDFLARE_ACCOUNT_ID',), 'key': 'CLOUDFLARE_API_TOKEN', 'file': 'cloudflare',
                   'group': 'cloudflare', 'slots': 1,   # 10,000 neurons/day for the account: ~2-13 episodes a day
                   'signup': 'a free Cloudflare account (https://dash.cloudflare.com/sign-up, no card), My Profile > API '
                   'Tokens > Create Token with "Workers AI: Read" and "Workers AI: Edit"; CLOUDFLARE_API_TOKEN=... and '
                   'CLOUDFLARE_ACCOUNT_ID=... (the dashboard URL\'s account id) in ~/.config/natlang/cloudflare.env'},
    'llm7': {'endpoint': 'https://api.llm7.io/v1', 'models': 'https://api.llm7.io/v1/models', 'public': True,
             'free': 'llm7-tier', 'key': 'LLM7_API_KEY', 'file': 'llm7', 'group': 'llm7', 'rpm': 120, 'slots': 1,
             'signup': 'a free token at https://token.llm7.io (email, no card), LLM7_API_KEY=... in ~/.config/natlang/llm7.env'},
    'modelscope': {'endpoint': 'https://api-inference.modelscope.cn/v1', 'models': 'https://api-inference.modelscope.cn/v1/models',
                   'key': 'MODELSCOPE_API_KEY', 'file': 'modelscope', 'group': 'modelscope:{model}', 'rpd': 500,
                   'slots': 1,   # 2,000 calls a day per account, 500 per model (reported)
                   'signup': 'a ModelScope account (https://modelscope.cn, Alibaba Cloud account or phone), bind it to '
                   'an Alibaba Cloud account, create an access token (https://modelscope.cn/my/myaccesstoken), '
                   'MODELSCOPE_API_KEY=... in ~/.config/natlang/modelscope.env'},
}


def _keyed(provider):
    return lambda: list_keyed(provider)


PROVIDERS = {'openrouter': list_openrouter, 'kilo': list_kilo, 'nvidia-nim': list_nim, 'ai-studio': list_ai_studio,
             'zai': list_zai, 'groq': list_groq, 'opencode-zen': list_opencode,
             **{name: _keyed(name) for name in ('cerebras', 'sambanova', 'chutes', 'mistral', 'cloudflare',
                                                'huggingface', 'llm7', 'modelscope')}}


def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()] if path.exists() else []


def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def check(out=OUT, providers=None, now=None):
    """List every provider, append newly seen models as candidates; returns the run record. A provider without a key
    is recorded with its owner signup step (``needs_key``); one that fails is reported and skipped (its earlier models
    stay seen). Every newly seen model is a candidate (the 2026-11-24 baselines of the first six providers stay seen:
    their models were qualified by hand or added with ``add``)."""
    out.mkdir(parents=True, exist_ok=True)
    now = now or time.strftime('%Y-%m-%dT%H:%M:%S%z')
    seen_path = out / 'seen.json'
    seen = json.loads(seen_path.read_text()) if seen_path.exists() else {}
    record = {'time': now, 'providers': {}, 'new': []}
    new_rows = []
    for name, lister in (providers or PROVIDERS).items():
        try:
            models = lister()
        except MissingKey:
            record['providers'][name] = {'needs_key': True, 'signup': PROVIDER_SPECS.get(name, {}).get('signup')}
            continue
        except Exception as error:   # one provider down never blocks the others
            record['providers'][name] = {'error': repr(error)[:300]}
            continue
        known = seen.setdefault(name, {})
        fresh = sorted(set(models) - set(known))
        for model in fresh:
            known[model] = now
            new_rows.append({'provider': name, 'model': model, 'first_seen': now, 'status': OPEN, **models[model]})
        record['providers'][name] = {'free_models': len(models), 'new': len(fresh),
                                     'gone': sorted(set(known) - set(models))}
    record['new'] = [f"{r['provider']}:{r['model']}" for r in new_rows]
    write_atomic(seen_path, json.dumps(seen, indent=1, sort_keys=True) + '\n')
    if new_rows:
        with (out / 'candidates.jsonl').open('a') as handle:
            handle.writelines(json.dumps(r) + '\n' for r in new_rows)
    with (out / 'runs.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\n')
    return record


def update_candidates(out, change):
    """Rewrite candidates.jsonl with ``change(row)`` applied to every row (returns the rows)."""
    path = out / 'candidates.jsonl'
    rows = read_jsonl(path)
    for row in rows:
        change(row)
    write_atomic(path, ''.join(json.dumps(r) + '\n' for r in rows))
    return rows


def mark(model, status, note=None, out=OUT):
    """Rewrite the candidate lines of MODEL with STATUS (an open candidate is ``new``)."""
    path = out / 'candidates.jsonl'
    rows = read_jsonl(path)
    hit = [r for r in rows if r['model'] == model or f"{r['provider']}:{r['model']}" == model]
    if not hit:
        raise SystemExit(f'no candidate {model}')
    for row in hit:
        row['status'] = status
        if note:
            row['note'] = note
    write_atomic(path, ''.join(json.dumps(r) + '\n' for r in rows))
    return hit


def add(provider, model, out=OUT, now=None):
    """Enqueue a model the listing already knew (a baseline model never qualified) as an open candidate."""
    if provider not in PROVIDER_SPECS:
        raise SystemExit(f'unknown provider {provider}')
    rows = read_jsonl(out / 'candidates.jsonl')
    if any(r['provider'] == provider and r['model'] == model for r in rows):
        return None
    row = {'provider': provider, 'model': model, 'first_seen': now or time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'status': OPEN, 'added': True}
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'candidates.jsonl').open('a') as handle:
        handle.write(json.dumps(row) + '\n')
    return row


# candidate states: new (to probe) -> probed (answers a tool call) -> qualifying (a qualification runs) ->
# admitted (in teachers.json) | qualified (passed, its quota group full) | rejected (with a note)
OPEN_STATES = (OPEN, 'probed', 'qualifying')
NEEDS_KEY = 'needs-key'   # listed publicly, our key missing: probed again once the key exists


def open_candidates(out=OUT):
    """The candidates not yet decided: the ops dashboard's open-items rows."""
    return [r for r in read_jsonl(Path(out) / 'candidates.jsonl') if r.get('status') in OPEN_STATES]


# ---- probe: one tool call -----------------------------------------------------------------------------------------
# a candidate worth a qualification: a chat model of at least MIN_PARAMS_B billion parameters when its name gives a size
# (small models fail the qualification set: Lightning 30B-A3B 4/18, Gemma 4 26B-A4B 2/14)
MIN_PARAMS_B = 30
SMALL = re.compile(r'(?<![0-9.])(\d+(?:\.\d+)?)b(?![a-z])', re.I)
NOT_TEACHERS = CHAT_SKIP + ('nano', 'bolt', 'starcoder', 'cosmos', 'calibration', 'lyria', 'tts', 'omni',
                            'antigravity', 'lfm', 'customtools', 'research', 'laguna-xs', '-latest')
# a free tier whose tokens per minute cannot hold one of our requests (10-30k tokens of prompt) never carries an episode
MIN_TPM = 30000
PROBE_TOOLS = [{'type': 'function', 'function': {'name': 'calc', 'description': 'evaluate arithmetic',
                                                 'parameters': {'type': 'object', 'properties': {'expr': {'type': 'string'}},
                                                                'required': ['expr']}}}]
# answers that end a candidate (no retry the next day): no key or access, the model gone, no free quota, client-only
PERMANENT_PROBE = ('FreeUsageLimitError', 'limit: 0', 'no longer available', 'Insufficient balance',
                   'pro_access_required', 'requires a funded account', 'not enabled for', 'Invalid API Key',
                   'does not exist', 'not found')


def screen(row):
    """Why a listed model is no teacher candidate (None: probe it)."""
    model = row['model'].lower()
    if any(x in model for x in NOT_TEACHERS):
        return 'not a chat teacher (name)'
    sizes = [float(x) for x in SMALL.findall(model)]
    if sizes and max(sizes) < MIN_PARAMS_B:
        return f'small model ({max(sizes):g}B < {MIN_PARAMS_B}B)'
    if row.get('tools') is False:
        return 'no tool calling'
    tpm = PROVIDER_SPECS.get(row['provider'], {}).get('tpm')
    if tpm and tpm < MIN_TPM:
        return f'free tier {tpm} tokens/min: below one request'
    return None


def provider_key(provider):
    spec = PROVIDER_SPECS[provider]
    return env_key(spec['key'], spec['file'])


def endpoint_of(provider):
    spec = PROVIDER_SPECS[provider]
    return spec['endpoint'].format(**{v: env_key(v, spec['file']) or '' for v in spec.get('url_env', ())})


def post_json(url, body, headers, timeout=180):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={**UA, 'Content-Type': 'application/json', **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            limits = {k.lower(): v for k, v in response.headers.items() if 'ratelimit' in k.lower()}
            return response.status, json.loads(response.read()), limits
    except urllib.error.HTTPError as error:
        return error.code, error.read()[:2000].decode('utf-8', 'replace'), {}


def probe(row):
    """One chat request with a tool, as an episode's first call: (ok, note, rate-limit headers)."""
    key = provider_key(row['provider'])
    if not key and row['provider'] != 'kilo':
        return 'needs-key', PROVIDER_SPECS[row['provider']].get('signup') or 'no key', {}
    body = {'model': row['model'], 'messages': [{'role': 'user', 'content': 'What is 17*23? Use the calc tool.'}],
            'tools': PROBE_TOOLS, 'max_tokens': 2000}
    started = time.time()
    try:
        status, data, limits = post_json(endpoint_of(row['provider']).rstrip('/') + '/chat/completions', body,
                                         {'Authorization': f'Bearer {key}'} if key else {})
    except Exception as error:     # transport: retried the next day
        return None, repr(error)[:200], {}
    if status != 200:
        if 'PROMOTION_MODEL_LIMIT_REACHED' in str(data):   # Kilo's keyless allowance: a free account carries on
            return NEEDS_KEY, PROVIDER_SPECS[row['provider']].get('signup') or 'an account key', limits
        return (False if any(m in str(data) for m in PERMANENT_PROBE) else None), f'{status}: {str(data)[:300]}', limits
    message = (data.get('choices') or [{}])[0].get('message') or {}
    if not message.get('tool_calls'):
        return False, 'answered without a tool call', limits
    tokens = limits.get('x-ratelimit-limit-tokens')
    if tokens and tokens.isdigit() and int(tokens) < MIN_TPM:
        return False, f'{tokens} tokens/min: below one request', limits
    return True, f'tool call in {time.time() - started:.1f}s', limits


def probe_all(out=OUT):
    """Probe every ``new`` candidate: probed (answers a tool call), rejected (screened out or a permanent refusal), or
    left new (a transient failure: 503, 429 with a retry, transport) for the next run."""
    results = {}

    def change(row):
        if row.get('status') not in (OPEN, NEEDS_KEY):
            return
        reason = screen(row)
        ok, note, limits = (False, reason, {}) if reason else probe(row)
        if ok == NEEDS_KEY:   # waits, off the open items, until the owner's signup puts the key in place
            row['status'] = NEEDS_KEY
            results[f"{row['provider']}:{row['model']}"] = NEEDS_KEY
            return
        row['probe'] = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'ok': ok, 'note': note, **({'limits': limits} if limits else {})}
        if ok:
            row['status'] = 'probed'
        elif ok is False:
            row['status'], row['note'] = 'rejected', note
        elif row['status'] == NEEDS_KEY:
            row['status'] = OPEN
        results[f"{row['provider']}:{row['model']}"] = row['status'] if ok is not None else f'retry ({note[:80]})'
    update_candidates(out, change)
    return results


# ---- qualification: the 20-task teacher set (DECISIONS 2026-11-24 "teacher providers") ----------------------------
DATA = Path(resolve_str('data_nvme', 'lfm3-rl-dgx-20261029'))
QUAL_SET = DATA / 'ops/ei/teacher-qual-20261124'          # qual-tasks.jsonl: 10 Luna-solved, 10 Luna-failed tasks
QUAL_SEED = QUAL_SET / 'nemotron-3.5-super'               # its pool.json / inputs.json (the receipt of that task list)
QUAL_ROOT = DATA / 'ops/ei/teacher-qual'
TEACHERS_FILE = DATA / 'ops/ei/teachers.json'
MEMORY_LEDGER = Path(__file__).resolve().parent / 'memory_ledger.py'
QUAL_TASKS = 20
# qualified: at least MIN_YIELD successes per verdict over at least MIN_VERDICTS verdicts, at most MAX_EPISODE_S mean
# seconds per episode (MiniMax and Mistral took 1,500-1,800 s and failed), at most MAX_DEFERRED of the attempts deferred
MIN_YIELD, MIN_VERDICTS, MAX_EPISODE_S, MAX_DEFERRED = 0.25, 12, 900, 0.5
REQUESTS_PER_EPISODE = 38
# qualifications running at once (each one slot on a free tier; quota groups never run two at once)
MAX_QUALIFYING = 6
# key files the campaign's launch script sources (expert_iteration.TEACHER_ENV_FILES at the pins in use); a teacher
# whose key lives elsewhere reads it with an api_key_command (works at every pin)
SOURCED_KEY_FILES = ('nvidia', 'gemini', 'anthropic', 'openrouter')


def slug(text):
    return re.sub(r'[^A-Za-z0-9]+', '-', text).strip('-').lower()[:60]


def group_of(provider, model):
    return PROVIDER_SPECS[provider]['group'].format(model=model)


def teacher_spec(provider, model, slots=1):
    """The teacher_gen --teacher spec of a free model."""
    spec = PROVIDER_SPECS[provider]
    out = {'model_id': model, 'endpoint': endpoint_of(provider), 'slots': slots}
    if provider == 'kilo' and not provider_key('kilo'):
        return out
    out['api_key_env'] = spec['key']
    if spec['file'] not in SOURCED_KEY_FILES:
        out['api_key_command'] = ['/bin/sh', '-c', f'. "$HOME/.config/natlang/{spec["file"]}.env" && printf %s "${spec["key"]}"']
    return out


def current_pin():
    """The checkout the teacher campaign runs: the ``cd`` of the newest aei1 teacher launch script, else this repo."""
    scripts = sorted(DATA.glob('ops/ei/*/teacher-*/launch.sh'), key=lambda p: p.stat().st_mtime)
    for script in reversed(scripts):
        for line in script.read_text().splitlines():
            if line.startswith('cd ') and Path(line[3:].strip(), 'scripts/neuralese_data/teacher_gen.py').exists():
                return Path(line[3:].strip())
    return Path(__file__).resolve().parents[1]


def qual_work(row):
    return QUAL_ROOT / f"{slug(row['provider'])}--{slug(row['model'])}"


def seed_work(work):
    """A qualification WORK: the fixed 20-task list and its pool receipt (teacher_gen resumes it as is)."""
    work.mkdir(parents=True, exist_ok=True)
    if not (work / 'tasks.jsonl').exists():
        shutil.copyfile(QUAL_SET / 'qual-tasks.jsonl', work / 'tasks.jsonl')
        for name in ('pool.json', 'inputs.json'):
            shutil.copyfile(QUAL_SEED / name, work / name)


def unit_of(work):
    return 'tq-' + slug(work.name)[:50]


def unit_active(unit):
    states = [subprocess.run(['systemctl', '--user', 'is-active', f'{u}.service'], capture_output=True, text=True).stdout.strip()
              for u in (unit, unit + '-waiter')]
    return any(s in ('active', 'activating') for s in states)


def launch_qualification(row, pin=None, runner=subprocess.run):
    """teacher_gen on the qualification set with one slot of ROW's model, as a detached ledger unit."""
    pin = Path(pin or current_pin())
    work = qual_work(row)
    seed_work(work)
    spec = teacher_spec(row['provider'], row['model'])
    python = pin / '.venv-neuralese/bin/python'
    command = [str(python if python.exists() else sys.executable), str(pin / 'scripts/neuralese_data/teacher_gen.py'),
               '--work', str(work), '--attempts', '1', '--limit', str(QUAL_TASKS), '--route-ratio', '0',
               '--teacher', json.dumps(spec)]
    script = work / 'run.sh'
    script.write_text('#!/bin/bash\n# teacher qualification (scripts/teacher_free_models.py qualify)\nset -u\n'
                      f'set -a; for f in {" ".join(SOURCED_KEY_FILES)}; do [ -f "$HOME/.config/natlang/$f.env" ] && '
                      '. "$HOME/.config/natlang/$f.env"; done; set +a\n'
                      f'cd {shlex.quote(str(pin))}\nexec ' + ' '.join(map(shlex.quote, command)) + '\n')
    script.chmod(0o755)
    unit = unit_of(work)
    runner([sys.executable, str(MEMORY_LEDGER), 'run', '--unit', unit, '--budget-gb', '1.5', '--priority', 'low',
            '--wait', '21600', '--detach', '--env', f'NATLANG_REPO={pin}',
            '--env', f'PYTHONPATH={pin}/training/neuralese:{pin}', '--',
            '/bin/bash', '-c', f'exec {shlex.quote(str(script))} >> {shlex.quote(str(work / "run.log"))} 2>&1'],
           check=True, capture_output=True, text=True)
    return work, unit, str(pin)


def qualify_all(out=OUT, pin=None, runner=subprocess.run, active=unit_active):
    """Start qualifications for probed candidates, at most MAX_QUALIFYING at once and one per quota group."""
    rows = read_jsonl(out / 'candidates.jsonl')
    running = [r for r in rows if r.get('status') == 'qualifying']
    busy = {group_of(r['provider'], r['model']) for r in running}
    started = []

    def change(row):
        if row.get('status') != 'probed' or len(running) + len(started) >= MAX_QUALIFYING:
            return
        group = group_of(row['provider'], row['model'])
        if group in busy:
            return
        work, unit, used_pin = launch_qualification(row, pin, runner)
        busy.add(group)
        row.update(status='qualifying', work=str(work), unit=unit, pin=used_pin,
                   qualification_started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        started.append(f"{row['provider']}:{row['model']}")
    update_candidates(out, change)
    return started


def qualification_result(work):
    """Verdicts of a qualification WORK: successes, verdicts, mean seconds per episode, deferred attempts."""
    work = Path(work)
    successes = verdicts = 0
    seconds = 0.
    for record in read_jsonl(work / 'manifest.jsonl'):
        seconds += record.get('seconds') or 0
        for episode in record['episodes']:
            verdicts += 1
            successes += episode['outcome'] == 'task_success' and (episode.get('reward') or 0) >= 1
    deferred = len(read_jsonl(work / 'deferred.jsonl'))
    return {'successes': successes, 'verdicts': verdicts, 'mean_s': round(seconds / max(verdicts, 1)),
            'deferred': deferred, 'yield': round(successes / max(verdicts, 1), 3)}


# a free tier that serves fewer requests a day than MIN_DAILY_EPISODES episodes cannot pay for its qualification (AI
# Studio's Gemini 3.x Flash: 20 requests/day per model on 2026-11-25, half an episode)
MIN_DAILY_EPISODES = 4
DAILY_LIMIT = re.compile(r'limit: (\d+), model: [^,]+, description: [^.]*requests per day')
# and one whose input tokens per minute cannot hold one request (AI Studio's Gemma 4: 16,000)
TOKEN_LIMIT = re.compile(r'limit: (\d+), model: [^,]+, description: [^.]*input tokens per minute')


def named_limit(work, pattern):
    """The smallest limit of ``pattern`` a provider named in a qualification's logs (AI Studio's free-tier 429 bodies,
    in the episode logs and a running episode's per-call evidence), or None."""
    limits = []
    for path in [*Path(work).glob('episodes/*/*.log'), *Path(work).glob('episodes/*/row-*/sample-*/jobs/*evidence*.jsonl')]:
        try:
            limits += [int(n) for n in pattern.findall(path.read_text(errors='replace'))]
        except OSError:
            continue
    return min(limits) if limits else None


def daily_request_limit(work):
    return named_limit(work, DAILY_LIMIT)


def stop_unit(unit):
    for name in (unit + '-waiter', unit):
        subprocess.run(['systemctl', '--user', 'stop', f'{name}.service'], capture_output=True)


def verdict(result):
    """(qualified, reason) of a finished qualification."""
    if result['verdicts'] < MIN_VERDICTS:
        return False, f"{result['verdicts']} verdicts ({result['deferred']} deferred): the provider did not carry the set"
    if result['deferred'] > MAX_DEFERRED * (result['verdicts'] + result['deferred']):
        return False, f"{result['deferred']} of {result['verdicts'] + result['deferred']} attempts deferred"
    if result['mean_s'] > MAX_EPISODE_S:
        return False, f"{result['mean_s']} s per episode"
    if result['yield'] < MIN_YIELD:
        return False, f"yield {result['successes']}/{result['verdicts']}"
    return True, f"yield {result['successes']}/{result['verdicts']}, {result['mean_s']} s per episode"


def slots_for(provider, mean_s, group_slots_used=0):
    """Teacher slots a free tier carries: one slot issues REQUESTS_PER_EPISODE requests per mean_s seconds; slots up to
    70% of the provider's requests per minute, within its group's slot cap, at least 1 (a daily request limit binds per
    day, not per slot: more slots only finish it sooner)."""
    spec = PROVIDER_SPECS[provider]
    per_slot_rpm = REQUESTS_PER_EPISODE * 60 / max(mean_s, 1)
    slots = int(0.7 * spec['rpm'] / per_slot_rpm) if spec.get('rpm') else spec.get('slots', 1)
    if spec.get('rpd'):
        slots = 1
    return max(0, min(max(1, slots), spec.get('slots', 1) - group_slots_used, 4))


def expected_successes_per_hour(provider, result, slots):
    """slots x episodes per slot-hour x qualification yield, capped by the daily request limit (an upper bound: the
    campaign's queue is harder than the qualification set)."""
    per_hour = slots * 3600 / max(result['mean_s'], 1)
    rpd = result.get('requests_per_day') or PROVIDER_SPECS[provider].get('rpd')
    if rpd:
        per_hour = min(per_hour, rpd / REQUESTS_PER_EPISODE / 24)
    return round(per_hour * result['yield'], 1)


def group_slots(teachers):
    """Slots each quota group already holds in teachers.json (teachers with an endpoint of a known provider)."""
    used = {}
    for t in teachers:
        provider = next((p for p, s in PROVIDER_SPECS.items() if t.get('endpoint') and
                         t['endpoint'].rstrip('/') == s['endpoint'].split('{')[0].rstrip('/')), None)
        if provider:
            group = group_of(provider, t['model_id'])
            used[group] = used.get(group, 0) + int(t.get('slots', 1))
    return used


def adopt_all(out=OUT, teachers_file=TEACHERS_FILE, active=unit_active, now=None, stop=stop_unit):
    """Decide every finished qualification; write each qualified model into teachers.json with slots sized to its
    provider's limits and its quota group's remaining slots (a full group: ``qualified``, not admitted)."""
    teachers_file = Path(teachers_file)
    teachers = json.loads(teachers_file.read_text()) if teachers_file.exists() else []
    decided = []

    def change(row):
        if row.get('status') != 'qualifying':
            return
        result = qualification_result(row['work'])
        rpd = daily_request_limit(row['work'])
        if rpd is not None:
            result['requests_per_day'] = rpd
        tpm = named_limit(row['work'], TOKEN_LIMIT)
        if rpd is not None and rpd < MIN_DAILY_EPISODES * REQUESTS_PER_EPISODE:
            stop(row['unit'])
            ok, reason = False, f'free tier {rpd} requests/day (~{rpd / REQUESTS_PER_EPISODE:.1f} episodes a day)'
        elif tpm is not None and tpm < MIN_TPM:
            stop(row['unit'])
            ok, reason = False, f'free tier {tpm} input tokens/min: below one request'
        elif result['verdicts'] < QUAL_TASKS and active(row['unit']):
            return
        else:
            ok, reason = verdict(result)
        row['qualification'] = result
        if not ok:
            row['status'], row['note'] = 'rejected', reason
        elif any(t['model_id'] == row['model'] for t in teachers):
            row['status'], row['note'] = 'admitted', reason + '; already a teacher'
        else:
            slots = slots_for(row['provider'], result['mean_s'], group_slots(teachers).get(group_of(row['provider'], row['model']), 0))
            if slots < 1:
                row['status'], row['note'] = 'qualified', reason + f"; quota group {group_of(row['provider'], row['model'])} full"
            else:
                spec = {**teacher_spec(row['provider'], row['model'], slots),
                        'free_check': {'qualified': (now or time.strftime('%Y-%m-%d'))[:10], 'yield': f"{result['successes']}/{result['verdicts']}",
                                       'mean_s': result['mean_s'], 'group': group_of(row['provider'], row['model'])}}
                teachers.append(spec)
                row['status'], row['note'] = 'admitted', reason + f'; {slots} slots'
                row['expected_successes_per_hour'] = expected_successes_per_hour(row['provider'], result, slots)
        decided.append({k: row.get(k) for k in ('provider', 'model', 'status', 'note', 'expected_successes_per_hour')})
    update_candidates(out, change)
    if any(d['status'] == 'admitted' and 'already' not in (d['note'] or '') for d in decided):
        write_atomic(teachers_file, json.dumps(teachers, indent=1) + '\n')
    return decided


def daily(out=OUT, pin=None):
    """The timer's run: list, probe the new candidates, decide finished qualifications, start new ones."""
    record = check(out)
    probed = probe_all(out)
    decided = adopt_all(out)
    started = qualify_all(out, pin)
    summary = {'providers': record['providers'], 'new': record['new'], 'probed': probed, 'decided': decided,
               'qualifying': started}
    with (out / 'daily.jsonl').open('a') as handle:
        handle.write(json.dumps({'time': record['time'], **summary}) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('check')
    d = sub.add_parser('daily')
    d.add_argument('--pin', help='checkout the qualifications run (default: the teacher campaign\'s)')
    sub.add_parser('probe')
    q = sub.add_parser('qualify')
    q.add_argument('--pin')
    sub.add_parser('adopt')
    a = sub.add_parser('add')
    a.add_argument('provider', choices=sorted(PROVIDER_SPECS))
    a.add_argument('model')
    m = sub.add_parser('mark')
    m.add_argument('model')
    m.add_argument('status')
    m.add_argument('--note')
    sub.add_parser('signups')
    args = p.parse_args()
    if args.cmd == 'check':
        record = check()
        print(json.dumps({'providers': record['providers'], 'new': record['new']}))
    elif args.cmd == 'daily':
        print(json.dumps(daily(pin=args.pin)))
    elif args.cmd == 'probe':
        print(json.dumps(probe_all()))
    elif args.cmd == 'qualify':
        print(json.dumps(qualify_all(pin=args.pin)))
    elif args.cmd == 'adopt':
        print(json.dumps(adopt_all()))
    elif args.cmd == 'add':
        print(json.dumps(add(args.provider, args.model)))
    elif args.cmd == 'signups':
        print(json.dumps({n: s['signup'] for n, s in PROVIDER_SPECS.items()
                          if s.get('signup') and not provider_key(n)}, indent=1))
    else:
        print(json.dumps(mark(args.model, args.status, args.note)))


if __name__ == '__main__':
    main()
