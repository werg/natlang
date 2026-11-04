"""Adapter fits for memetic episodes: the bank coefficient fit and adapter-composed evaluation (fallow register M2).

A memetic supervisor's ``fit(spec)`` fits per-group mixture coefficients of the published LoRA bank
(`tune/learned/meta_bank.py`, its ``inner_fit`` law) on the program's verified support runs, and ``evaluate`` then
runs the program with that adapter on the support worlds; ``score`` on the query worlds.

Mechanism (measured from the code, 2026-11-03):

- The native HTTP server (`serve/http.py`) applies weight adapters per request: ``x_natlang_adapters: [{id, scale}]``
  on ``/v1/chat/completions``, after the adapter's coefficient artifact and its basis artifact are in the server's
  store (``PUT /v1/neuralese/native-tensor-artifacts/<id>``). The native engine substitutes the effective weights
  (`model.native_adapters.run_native`, effective weights cached per stack); the SGLang actor
  (`serve/sglang_actor.py`, production default 4 resident adapter slots of summed rank <= 32 per module) serves a
  bank adapter (rank = the basis rank, 32) on SGLang and otherwise falls back to its lazily loaded native engine.
  Every captured turn records the effective adapted identity (``sampling.policy``) and its ``adapter_binding``.
- The production collector forwards the stack through its chat-request config (``collect(adapters=...)``), bound
  in full into the collection binding: an adapter-composed actor is a different actor, recorded as such.
- Fitting needs forward/backward passes of the parent on the support rows: `BankFit` runs the published law on an
  engine, served by ``python -m natlang_neuralese.rl.memetic_fit serve`` (one GPU process holding the parent and the
  bank, ledger-friendly, low priority). The workspace talks to it through `bank_fitter`.

Leakage: rows are the parent's own oracle-verified successful attempts on the episode's support worlds
(`meta_bank.rollout_rows`, protected worlds and other actors refused); query/dev/confirmation worlds and gold never
reach the fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex
from ..common.jsonio import write_json_durable

FIT_SCHEMA = 'natlang.memetic-bank-fit/1'
ADAPTER_SCHEMA = 'natlang.memetic-adapter/1'
ACTOR_SCHEMA = 'natlang.adapter-actor/1'
SCOPES = ('all', 'attention', 'convolution')
INITS = ('learned', 'zero')
MAX_STEPS = 48


class FitProblem(ValueError):
    """A fit request that cannot run; ``problems`` are the supervisor-facing explanations."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__('; '.join(self.problems))


# -- the request: which worlds, which rows, which coefficients ---------------------------------------------------
def support_rollouts(spec):
    """The parent's unchanged support collection of an episode: ``spec['evidence']['rollouts']`` when recorded,
    else the `memetic_stage.build` layout (``<program>/baselines/<parent16>/evals/<parent16>/baseline-support``)."""
    recorded = (spec.get('evidence') or {}).get('rollouts')
    if recorded:
        return [str(Path(path)) for path in ([recorded] if isinstance(recorded, str) else recorded)]
    parent = spec['parent_sha256'][:16]
    program = Path(spec['out']).parents[1]
    return [str(program / 'baselines' / parent / 'evals' / parent / 'baseline-support' / 'collection' / 'rollouts.jsonl')]


def normalize_fit_spec(spec, fit_spec, info):
    """The supervisor's ``FitSpec`` checked against the episode and the bank ``info`` (`BankFit.info`).
    Returns ``{worlds, components, steps, scope, init}``; raises `FitProblem`."""
    fit_spec = fit_spec or {}
    problems = []
    unknown = sorted(set(fit_spec) - {'worlds', 'components', 'steps', 'scope', 'init'})
    if unknown:
        problems.append(f'unknown fit fields {unknown}; FitSpec has worlds, components, steps, scope, init')
    support = [case['id'] for case in spec['support']]
    worlds = fit_spec.get('worlds')
    if worlds is None:
        worlds = support
    elif not isinstance(worlds, list) or not worlds or any(world not in support for world in worlds):
        problems.append(f'worlds must be a non-empty list of support worlds: {support}')
    K = info['components']
    components = fit_spec.get('components', K)
    if type(components) is not int or not 1 <= components <= K:
        problems.append(f'components must be an integer in [1, {K}] (the bank\'s leading components)')
    steps = fit_spec.get('steps', info['steps'])
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        problems.append(f'steps must be an integer in [1, {MAX_STEPS}] (default {info["steps"]})')
    scope = fit_spec.get('scope', 'all')
    if scope not in SCOPES:
        problems.append(f'scope must be one of {list(SCOPES)}')
    init = fit_spec.get('init', 'learned')
    if init not in INITS:
        problems.append(f'init must be one of {list(INITS)}')
    if problems:
        raise FitProblem(problems)
    return {'worlds': sorted(worlds), 'components': components, 'steps': steps, 'scope': scope, 'init': init}


def coefficient_mask(K, groups, components, scope):
    """``(K, G)`` 0/1 mask: the leading ``components`` bank components, on the groups of ``scope``."""
    import torch
    mask = torch.zeros(K, len(groups))
    columns = [g for g, label in enumerate(groups) if scope == 'all' or label.endswith(':' + scope)]
    mask[:components, columns] = 1.
    return mask


# -- the fit (engine side) -----------------------------------------------------------------------------------------
def import_artifacts(source_store, store, ids):
    """Copy named artifacts from a verified store directory (read only: no lock, no index write) into ``store``."""
    from ..model.native_artifact import decode_native_tensor_artifact
    for ident in ids:
        path = Path(source_store) / 'blocks' / ident.split('_', 1)[1][:2] / f'{ident}.safetensors'
        artifact, actual = decode_native_tensor_artifact(path.read_bytes())
        if actual != ident:
            raise ValueError(f'{path} does not hold {ident}')
        store.put_native_tensor_artifact(artifact)


def _file_binding(path):
    data = Path(path).read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


class BankFit:
    """The published bank's inner fit on one engine (the parent). ``manifest``: the bank manifest whose basis, component
    tensors and inner-fit law are in ``engine.store``; ``out``: fit records, coefficient files and exported artifacts."""

    def __init__(self, engine, manifest, out, *, protected=None, max_tokens=16384):
        from ..tune.learned.bank import load_bank
        from ..tune.learned.fit import placeholder_adapters
        from ..tune.learned.meta_bank import INNER_FIT_LAW, SCOPE
        self.engine, self.manifest, self.out = engine, manifest, Path(out)
        self.protected, self.max_tokens = protected, max_tokens
        law = manifest.get('inner_fit') or {}
        if law.get('law') != INNER_FIT_LAW:
            raise ValueError('the bank carries no meta-learned inner-fit law')
        self.bank = load_bank(engine, manifest)
        if dict(self.bank.basis.native_identity) != dict(engine.native_identity):
            # Bank coordinates are weight-update coordinates of the parent they were built on; another parent takes
            # the recorded meta-learner transition (`BankMetaLearner` with its parent_sha256) first.
            raise ValueError('the published bank was built on another parent than this engine')
        tensors = engine.store.get_native_tensor_artifact(law['tensors']).artifact.tensors
        self.init, self.log_step, self.steps = tensors['init'], tensors['log_step'], int(law['steps'])
        self.scope = SCOPE
        self.placeholder = placeholder_adapters(engine, self.bank, [SCOPE])[SCOPE]
        self.lock = threading.Lock()

    def info(self):
        return {'schema': FIT_SCHEMA, 'bank': self.bank.snapshot, 'inner_fit': self.manifest['inner_fit']['tensors'],
                'components': len(self.bank.components), 'groups': list(self.bank.group_labels), 'steps': self.steps,
                'parent_sha256': self.engine.checkpoint_sha256, 'native_identity': dict(self.engine.native_identity)}

    def rows(self, rollouts, worlds, parent):
        from ..tune.learned.meta_bank import rollout_rows
        rows, counts = rollout_rows(self.engine, rollouts, protected=self.protected, max_tokens=self.max_tokens,
                                    policy=parent)
        kept = [row for row in rows if row['world'] in set(worlds)]
        return kept, {**counts, 'outside_worlds': len(rows) - len(kept), 'rows': len(kept)}

    def _leaves(self, a):
        slots = list(range(len(self.bank.components)))
        return {self.placeholder: self.bank.adapter_tensors(self.bank.composed_cores(slots, a))}

    def fit(self, request):
        """One fit request (``bank_fitter``'s body). Cached by request digest: the same request returns its saved
        result. Returns the result record or ``{'problems': [...]}``."""
        import torch
        from ..tune.learned.meta_bank import meta_sgd, support_set
        # The key binds the rollout bytes too: a changed support collection is a new fit.
        sources = [_file_binding(path)['sha256'] if Path(path).exists() else None for path in request['rollouts']]
        key = canonical_json_sha256_hex({**request, 'bank': self.bank.snapshot, 'rollout_sha256': sources})
        directory = self.out / 'fits' / key[:24]
        saved = directory / 'fit.json'
        if saved.exists():
            return json.loads(saved.read_text())
        if request.get('parent_sha256') != self.engine.checkpoint_sha256:
            return {'problems': ['the fit service serves another parent than this episode']}
        fit = request['fit']
        with self.lock:
            rows, counts = self.rows(request['rollouts'], fit['worlds'], request['parent_sha256'])
            if not rows:
                return {'problems': ['no verified successful support run of the parent to fit on '
                                     f'({counts}); use guidance, skills or patches'], 'counts': counts}
            support = support_set(self.engine, rows)
            K, G = self.init.shape
            mask = coefficient_mask(K, self.bank.group_labels, fit['components'], fit['scope'])
            a0 = (self.init if fit['init'] == 'learned' else torch.zeros(K, G)) * mask
            with torch.no_grad():  # the parent: every coefficient zero (`BankMetaLearner.score`)
                before = support.accumulate({self.scope: self.placeholder}, lambda: self._leaves(torch.zeros(K, G)),
                                            backward=False)
            a, _, history = meta_sgd(a0, self.log_step, fit['steps'],
                                     lambda a: support.accumulate({self.scope: self.placeholder},
                                                                  lambda: self._leaves(a * mask)))
            executed = a * mask
            with torch.no_grad():
                after = support.accumulate({self.scope: self.placeholder}, lambda: self._leaves(executed), backward=False)
            slots = list(range(K))
            record = self.bank.compose_adapter(self.engine, self.bank.composed_cores(slots, executed), provenance={
                'memetic_fit': key, 'support': [row['graph']['id'] for row in rows], 'law': self.manifest['inner_fit']['law']})
            directory.mkdir(parents=True, exist_ok=True)
            coefficients = directory / 'coefficients.pt'
            torch.save({'coefficients': executed.detach().cpu(), 'mask': mask, 'groups': list(self.bank.group_labels)},
                       coefficients)
            artifacts = {ident: self._export(ident) for ident in (self.bank.basis_artifact, record.id)}
            binding = {'schema': ADAPTER_SCHEMA, 'bank': self.bank.snapshot,
                       'inner_fit': self.manifest['inner_fit']['tensors'], 'parent_sha256': self.engine.checkpoint_sha256,
                       'native_identity': dict(self.engine.native_identity), 'adapter_id': record.id,
                       'basis_id': self.bank.basis_artifact, 'artifacts': artifacts,
                       'coefficients': _file_binding(coefficients), 'fit': fit,
                       'expected_effective_checkpoint_sha256': effective_checkpoint(self.engine.native_identity, record.id),
                       'support': {'rows': [row['graph']['id'] for row in rows], 'counts': counts}}
            result = {'schema': FIT_SCHEMA, 'request': request, 'adapter': binding,
                      'support_loss': {'before': float(before), 'after': float(after),
                                       'history': [float(x) for x in history]}}
            write_json_durable(saved, result)
            return result

    def _export(self, ident):
        """The artifact's exact store bytes beside the fit, for upload to the episode's actor."""
        from ..model.native_artifact import encode_native_tensor_artifact
        path = self.out / 'artifacts' / f'{ident}.safetensors'
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix('.partial')
            partial.write_bytes(encode_native_tensor_artifact(self.engine.store.get_native_tensor_artifact(ident).artifact))
            partial.replace(path)
        return _file_binding(path)


def effective_checkpoint(native_identity, adapter_id, scale=1.):
    """The effective actor identity the native engine records for one adapter at ``scale``
    (`model.native_adapters.effective_identity`)."""
    from ..model.native_adapters import SCHEMA
    return canonical_json_sha256_hex({'schema': SCHEMA, 'base_native_identity': dict(native_identity),
                                      'adapters': [{'id': adapter_id, 'scale_hex': float(scale).hex()}]})


# -- the fit service (HTTP) ----------------------------------------------------------------------------------------
def serve_fit(service, host='127.0.0.1', port=0):
    """``GET /v1/memetic/info`` and ``POST /v1/memetic/fit`` over one `BankFit`. Returns the (unstarted) server."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('content-type', 'application/json')
            self.send_header('content-length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/v1/memetic/info':
                return self._reply(200, service.info())
            self._reply(404, {'error': self.path})

        def do_POST(self):
            if self.path != '/v1/memetic/fit':
                return self._reply(404, {'error': self.path})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('content-length') or 0)) or b'{}')
                self._reply(200, service.fit(body))
            except Exception as error:  # noqa: BLE001 - the caller gets the exact failure
                self._reply(500, {'error': f'{type(error).__name__}: {error}'})

    return ThreadingHTTPServer((host, port), Handler)


def _http(endpoint, path, value=None, timeout=7200):
    request = urllib.request.Request(endpoint.rstrip('/') + path, data=None if value is None else json.dumps(value).encode(),
                                     headers={'content-type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise ValueError(f'fit service {path}: {error.read().decode(errors="replace")[:500]}') from error


def bank_fitter(spec, fit_spec, *, endpoint, request=_http):
    """``Workspace(fitter=...)``: fit the bank on the episode's verified support runs through the fit service at
    ``endpoint``. Returns ``{'adapter': binding, 'support_loss': {'before', 'after'}}``; raises `FitProblem`."""
    info = request(endpoint, '/v1/memetic/info')
    if info.get('schema') != FIT_SCHEMA:
        raise FitProblem(['the configured fit service is not a memetic bank fit service'])
    if info['parent_sha256'] != spec['parent_sha256']:
        raise FitProblem(['the bank fit service serves another parent than this episode'])
    fit = normalize_fit_spec(spec, fit_spec, info)
    body = {'parent_sha256': spec['parent_sha256'], 'rollouts': support_rollouts(spec), 'fit': fit,
            'program': spec['program']['id']}
    result = request(endpoint, '/v1/memetic/fit', body)
    if result.get('problems'):
        raise FitProblem(result['problems'])
    if result.get('schema') != FIT_SCHEMA or (result.get('adapter') or {}).get('schema') != ADAPTER_SCHEMA:
        raise ValueError('the fit service answered without an adapter binding')
    loss = result['support_loss']
    return {'adapter': result['adapter'], 'support_loss': {'before': loss['before'], 'after': loss['after']}}


# -- adapter-composed evaluation -----------------------------------------------------------------------------------
def actor_binding(adapter, base_identity):
    """The ``collect(adapters=...)`` binding of one fitted adapter on the actor serving ``base_identity``."""
    if adapter.get('schema') != ADAPTER_SCHEMA:
        raise ValueError('adapter evaluation needs a natlang.memetic-adapter/1 binding')
    if any(adapter['native_identity'].get(key) != value for key, value in base_identity.items()):
        raise ValueError('the adapter was fitted on another parent than the actor serves')
    return {'schema': ACTOR_SCHEMA, 'entries': [{'id': adapter['adapter_id'], 'scale': 1.0}],
            'base': dict(base_identity), 'adapter': canonical_json_sha256_hex(adapter),
            'bank': adapter['bank'], 'coefficients_sha256': adapter['coefficients']['sha256'],
            'effective_checkpoint_sha256': adapter['expected_effective_checkpoint_sha256']}


def upload_adapter(endpoint, adapter, *, put=None):
    """Put the basis and the adapter (in that order) into the actor's store from their verified exported bytes."""
    from .actor_client import NativeActorClient
    client = NativeActorClient(endpoint)
    put = put or (lambda path, data: client.put_bytes(path, data, owner=f"memetic-adapter:{adapter['adapter_id']}"))
    for ident in (adapter['basis_id'], adapter['adapter_id']):
        record = adapter['artifacts'][ident]
        data = Path(record['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError(f'exported artifact {ident} changed since the fit')
        put(f'/v1/neuralese/native-tensor-artifacts/{ident}', data)


def verify_adapted_turns(rollouts_path, binding):
    """Every captured turn ran the bound adapter: its sampling policy is the expected effective identity and its
    adapter binding names exactly the bound stack. Returns the number of checked turns."""
    expected = [{'id': entry['id'], 'scale_hex': float(entry['scale']).hex()} for entry in binding['entries']]
    checked = 0
    path = Path(rollouts_path)
    for line in (path.read_text().splitlines() if path.exists() else []):
        rollout = json.loads(line)
        for turn in rollout.get('turns') or []:
            sampling = turn.get('sampling') or {}
            if ((sampling.get('policy') or {}).get('checkpoint_sha256') != binding['effective_checkpoint_sha256']
                    or (sampling.get('adapter_binding') or {}).get('adapters') != expected):
                raise ValueError(f"{rollout.get('task_id')}: a turn did not run the bound adapter actor")
            checked += 1
    return checked


def summarize(cases, out, report):
    """The workspace evaluator's summary of one collection in ``out/collection``."""
    from ..eval.task_packet import read_jsonl
    results = []
    rewards = read_jsonl(out / 'collection' / 'rewards.jsonl')
    for reward in rewards:
        path = out / 'collection' / f"row-{reward['group']:03d}" / f"sample-{reward['sample']:02d}" / 'results.jsonl'
        result = (read_jsonl(path) or [None])[0] if path.exists() else None
        if result is not None:
            result.setdefault('outcome', {})['status'] = reward['outcome']
        results.append((cases[reward['group']]['id'], result))
    scored = [r for r in rewards if r['reward'] is not None]
    return {'success': (sum(r['reward'] for r in scored) / len(scored)) if scored else 0., 'scored': len(scored),
            'results': results, 'dir': str(out),
            'report': {key: report.get(key) for key in ('successes', 'scored', 'rollouts')}}


def adapter_evaluator(spec, endpoint, *, base=None, collect=None, identity=None, upload=upload_adapter):
    """The workspace evaluator with adapters: ``evaluate(cases, guidance, adapter, *, name, out=None)``. Without an
    adapter it is ``base`` (default `memetic_workspace.production_evaluator`); with one it uploads the adapter to
    the episode's actor and collects with that exact adapter stack (``collect(adapters=...)``), under the effective
    actor's own directory, and verifies every captured turn ran it."""
    if base is None:
        from .memetic_workspace import production_evaluator
        base = production_evaluator(spec, endpoint)
    if collect is None:
        from .production import collect
    settings = spec['eval']
    resolved = {}

    def actor():
        # Resolved on the first adapted evaluation only: unadapted episodes never need it.
        if 'identity' not in resolved:
            if identity is not None:
                resolved['identity'] = identity
            else:
                from .memetic_program import actor_identity
                resolved['identity'] = actor_identity(endpoint)
        return resolved['identity']

    def evaluate(cases, guidance, adapter, *, name, out=None, samples=None):
        if adapter is None:
            return base(cases, guidance, adapter, name=name, out=out, samples=samples)
        if guidance:
            # A guidance state written by the base parent and imported by the adapted actor needs the
            # same-base-adapter-state consumption mode on the program-guidance channel; not wired yet.
            raise ValueError('evaluate an adapter fit without guidance changes (combined sets are not supported yet)')
        binding = actor_binding(adapter, actor())
        out = Path(out or spec['out']) / 'evals' / binding['effective_checkpoint_sha256'][:16] / name
        from .memetic_stage import _locked
        with _locked(out):
            return _adapted(cases, adapter, binding, out, samples)

    def _adapted(cases, adapter, binding, out, samples):
        if (out / 'summary.json').exists():
            return json.loads((out / 'summary.json').read_text())
        out.mkdir(parents=True, exist_ok=True)
        write_json_durable(out / 'adapter-actor.json', {**binding, 'adapter_binding': adapter})
        upload(endpoint, adapter)
        packet = out / 'cases.jsonl'
        packet.write_text(''.join(json.dumps(case, ensure_ascii=False) + '\n' for case in cases))
        report = collect(cases=packet, out=out / 'collection', endpoints=[endpoint],
                         policy_id=actor()['checkpoint_sha256'], rows=len(cases), group=samples or settings['samples'],
                         workers=settings['workers'], root_mode='student', seed_base=settings['seed_base'],
                         case_seconds=settings['case_seconds'], max_model_requests=settings['max_model_requests'],
                         max_turns=settings['max_turns'], prompt_rendering='full', adapters=binding)
        turns = verify_adapted_turns(out / 'collection' / 'rollouts.jsonl', binding)
        summary = {**summarize(cases, out, report), 'adapter_actor': binding['effective_checkpoint_sha256'],
                   'adapted_turns': turns}
        write_json_durable(out / 'summary.json', summary)
        return summary
    return evaluate


# -- CLI -----------------------------------------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description='memetic bank fit service (one GPU process: parent + published bank)')
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('serve')
    s.add_argument('--checkpoint', required=True)
    s.add_argument('--base', default=None)
    s.add_argument('--bank-manifest', required=True, type=Path, help="the published bank's bank-manifest.json")
    s.add_argument('--bank-store', required=True, type=Path, help='the verified store holding its artifacts (read only)')
    s.add_argument('--out', required=True, type=Path, help='fit records, working store and exported artifacts')
    s.add_argument('--device', default='cuda')
    s.add_argument('--dtype', default='bfloat16')
    s.add_argument('--host', default='127.0.0.1')
    s.add_argument('--port', type=int, default=0)
    s.add_argument('--max-tokens', type=int, default=16384)
    args = parser.parse_args(argv)
    from ..common.custody import ProtectedWorlds
    from ..tune.learned.bank_stage import _engine
    manifest = json.loads(args.bank_manifest.read_text())
    engine = _engine(args.checkpoint, args.device, args.dtype, args.out / 'store')
    import_artifacts(args.bank_store, engine.store, [manifest['basis'], manifest['tensors'], manifest['inner_fit']['tensors']])
    service = BankFit(engine, manifest, args.out, protected=ProtectedWorlds.load(), max_tokens=args.max_tokens)
    server = serve_fit(service, args.host, args.port)
    print(json.dumps({'listening': f'http://{server.server_address[0]}:{server.server_address[1]}', **service.info()}),
          flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
