"""One DiLoCo fragment exchange through a worker's relay, with the session's real masters (CPU; replay only).

Runs inside a diloco_session_replay container after the worker's controller admitted its role: the learner takes the
admitted role of this worker from the coordinator, builds its surface from the worker's parent checkpoint (the
coordinator's own master surface), installs the seed fragments (each must equal the parent), perturbs fragment 0 by
a worker-specific epsilon, takes the session's local steps (no training) until fragment 0 uploads, and then keeps
stepping until fragment 0 installs a merged global version. With ``--expect N`` probes on one coordinator, it also
waits until N learners contributed to that merge round (the coordinator's merge count of fragment 0).
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--session', required=True)
    ap.add_argument('--worker', type=int, required=True)
    ap.add_argument('--parent', required=True)
    ap.add_argument('--expect', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=1800)
    a = ap.parse_args(argv)
    import torch
    from natlang_neuralese.train.diloco.coordinator import TOKEN_ENV, TOKEN_SECRET, credential
    from natlang_neuralese.train.diloco.protocol import TensorSurface, pack_fragment
    from natlang_neuralese.train.diloco.worker import DilocoLearner, DilocoSettings, Transport
    from natlang_neuralese.train.native_diloco import _coordinator_surface

    def say(event, **fields):
        print(json.dumps({'at': round(time.time(), 1), 'event': event, **fields}, default=str), flush=True)

    session = json.loads(Path(a.session).read_text())
    url = f'http://127.0.0.1:{session["relay_port"]}'
    transport = Transport(url, credential(TOKEN_ENV, TOKEN_SECRET), 600.0)
    manifest = transport.json('GET', '/v1/manifest')
    authority = transport.json('GET', f'/v1/role-contracts?worker={a.worker}')
    roles = authority['roles'][str(a.worker)]
    if not roles:
        raise SystemExit(f'worker {a.worker} has no admitted role')
    identity = roles[-1]['identity']
    diloco = authority['diloco']
    settings = DilocoSettings(run_id=diloco['run_id'], workers=diloco['workers'], local_steps=diloco['local_steps'],
                              wire_dtype=diloco['wire_dtype'], coordinator_url=url, worker_id=a.worker,
                              start_grace_seconds=0)
    started = time.monotonic()
    masters = {n: v.clone() for n, v in _coordinator_surface(a.parent, None).items()}
    say('surface_loaded', tensors=len(masters), seconds=round(time.monotonic() - started, 1),
        fragments=len(manifest['fragments']))
    learner = DilocoLearner(settings, surface=TensorSurface(masters=masters), identity=identity,
                            output=Path(f'/tmp/probe-{a.worker}'), transport=transport)
    learner.start()
    say('seed_installed', contract_sha256=learner.contract_sha256, versions=learner.versions[:4])
    fragment = learner.fragments[0]
    before = pack_fragment(masters, fragment).clone()
    # Perturb exactly fragment 0's slices by a worker-specific epsilon (masters are this learner's weights).
    with torch.no_grad():
        for part in fragment.slices:
            masters[part.name].view(-1)[part.offset:part.offset + part.length].add_(1e-3 * (a.worker + 1))
    deadline = time.monotonic() + a.timeout
    steps = 0
    while time.monotonic() < deadline:
        steps += 1
        result = learner.after_step(tokens=1)
        if learner.versions[0] >= 1:
            break
        time.sleep(0.05 if steps < settings.local_steps else 2.0)
    else:
        raise SystemExit(f'fragment 0 never installed a merged version (steps {steps}, last error {learner.last_error})')
    after = pack_fragment(masters, fragment)
    status = transport.json('GET', '/v1/manifest')
    say('merged_installed', version=learner.versions[0], steps=steps, merges=learner.merges,
        changed=bool(not torch.equal(before, after)), max_delta=float((after.float() - before.float()).abs().max()),
        result=result, manifest_versions=status.get('versions', [])[:4])
    learner.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
