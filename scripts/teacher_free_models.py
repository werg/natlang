#!/usr/bin/env python3
"""Daily check of the free LLM APIs for new teacher candidates (DECISIONS 2026-11-24 "free teacher providers").

Free models on the hosted APIs rotate in and out (promotional `:free` models, new NIM catalogue entries, AI Studio
previews). Each run lists the free models of every provider below, compares them with what earlier runs saw and
appends each newly seen model to ``ops/teacher-free-models/candidates.jsonl`` with ``status: new``: a candidate to
qualify on the teacher qualification set (``ops/ei/teacher-qual-20261124``). The ops dashboard lists the open ones
beside OPEN_ITEMS; a candidate leaves that list when its line is rewritten with another status (``qualified``,
``rejected``, ``admitted``) by whoever qualifies it (``teacher_free_models.py mark MODEL STATUS``).

Providers: OpenRouter (``:free`` or zero-priced), the Kilo gateway (keyless; ``:free``/``-free``), NVIDIA NIM (the whole
API catalogue is free trial use), Google AI Studio (models with generateContent; the free tier's per-model quotas are not
listed by the API), Z.ai (``*-flash`` models are free), Groq (free tier). Keys come from ``~/.config/natlang/*.env``
and are never printed. A provider that fails is reported in the run record and skipped (its earlier models stay seen).

  teacher_free_models.py check            # list, diff, append new candidates; prints a summary
  teacher_free_models.py mark MODEL STATUS [--note TEXT]
"""
import argparse
import json
import os
import sys
import time
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


PROVIDERS = {'openrouter': list_openrouter, 'kilo': list_kilo, 'nvidia-nim': list_nim, 'ai-studio': list_ai_studio,
             'zai': list_zai, 'groq': list_groq}


def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()] if path.exists() else []


def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def check(out=OUT, providers=None, now=None):
    """List every provider, append newly seen models as candidates; returns the run record."""
    out.mkdir(parents=True, exist_ok=True)
    now = now or time.strftime('%Y-%m-%dT%H:%M:%S%z')
    seen_path = out / 'seen.json'
    seen = json.loads(seen_path.read_text()) if seen_path.exists() else {}
    record = {'time': now, 'providers': {}, 'new': []}
    new_rows = []
    for name, lister in (providers or PROVIDERS).items():
        try:
            models = lister()
        except Exception as error:   # one provider down never blocks the others
            record['providers'][name] = {'error': repr(error)[:300]}
            continue
        known = seen.setdefault(name, {})
        # a provider's first listing is its baseline (those models are known to the qualification already or are the
        # starting catalogue): only models added after it are new candidates
        baseline = not known
        fresh = sorted(set(models) - set(known))
        for model in fresh:
            known[model] = now
            if not baseline:
                new_rows.append({'provider': name, 'model': model, 'first_seen': now, 'status': OPEN, **models[model]})
        record['providers'][name] = {'free_models': len(models), 'new': 0 if baseline else len(fresh),
                                     'baseline': baseline, 'gone': sorted(set(known) - set(models))}
    record['new'] = [f"{r['provider']}:{r['model']}" for r in new_rows]
    write_atomic(seen_path, json.dumps(seen, indent=1, sort_keys=True) + '\n')
    if new_rows:
        with (out / 'candidates.jsonl').open('a') as handle:
            handle.writelines(json.dumps(r) + '\n' for r in new_rows)
    with (out / 'runs.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\n')
    return record


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


def open_candidates(out=OUT):
    """The candidates still to qualify (status ``new``): the ops dashboard's open-items rows."""
    return [r for r in read_jsonl(Path(out) / 'candidates.jsonl') if r.get('status') == OPEN]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('check')
    m = sub.add_parser('mark')
    m.add_argument('model')
    m.add_argument('status')
    m.add_argument('--note')
    args = p.parse_args()
    if args.cmd == 'check':
        record = check()
        print(json.dumps({'providers': record['providers'], 'new': record['new']}))
    else:
        print(json.dumps(mark(args.model, args.status, args.note)))


if __name__ == '__main__':
    main()
