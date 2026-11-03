"""One DiLoCo role inside a CoreWeave stage-shard sandbox (scripts/coreweave/rl_cloud.py, ``tune.module`` =
``natlang_neuralese.train.diloco.cloud_role``).

The cloud lineage machinery restores the parent checkpoint (exact ZIP-delta chain), installs the code snapshot and runs
this module with the lineage's argv. Secrets arrive as files under ``$NATLANG_CW_SECRETS`` (never in argv/env of the
job): ``diloco-token`` (bearer token, every role), and for learners ``diloco-url`` (the coordinator's public URL) and
``diloco-gateway`` (the coordinator sandbox's share token).

    cloud_role coordinator --root R --parent P --settings JSON --trainer-argv JSON [--port 8080] [-- INIT ARGS]
        initialize the coordinator root from the parent once (native_diloco init), then serve it
    cloud_role learner --worker I --settings JSON --out-link L --out-target D -- TRAINER ARGS
        run native_state_sft on worker I's shard with --diloco filled in (coordinator URL from the secret file).

Every learner names the same trainer argv (the coordinator identity binds it), including ``--out L``. ``L`` is a
symlink to this sandbox's own stage directory ``D``, so the learner state is exported with its stage while the
identity stays common to all workers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .worker import WORKER_ID_ENV

URL_SECRET = 'diloco-url'


def secret(name):
    directory = os.environ.get('NATLANG_CW_SECRETS')
    path = Path(directory) / name if directory else None
    if path is None or not path.is_file():
        raise SystemExit(f'missing DiLoCo secret {name} under $NATLANG_CW_SECRETS')
    return path.read_text().strip()


def coordinator(args, extra):
    from ..native_diloco import main as diloco_main
    from .__main__ import main as serve_main
    secret('diloco-token')
    root = Path(args.root)
    if not (root / 'manifest.json').is_file():
        diloco_main(['init', '--root', str(root), '--parent', str(args.parent), '--trainer-argv', args.trainer_argv,
                     '--settings', args.settings, *extra])
    serve_main(['serve', '--root', str(root), '--port', str(args.port)])


def learner_argv(settings, worker, url, trainer):
    if any(x == '--diloco' or x.startswith('--diloco=') for x in trainer):
        raise SystemExit('the trainer argv must not carry its own --diloco block')
    block = {**json.loads(settings), 'coordinator_url': url, 'worker_id': worker}
    return [*trainer, '--diloco', json.dumps(block, sort_keys=True)]


def link_out(link, target):
    link, target = Path(link), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if Path(os.readlink(link)) != target:
            raise SystemExit(f'{link} links elsewhere than {target}')
        return
    if link.exists():
        raise SystemExit(f'{link} exists and is not this learner\'s link')
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def learner(args, trainer):
    from ..native_state_sft import main as trainer_main
    secret('diloco-token')
    secret('diloco-gateway')
    url = secret(URL_SECRET)
    link_out(args.out_link, args.out_target)
    os.environ[WORKER_ID_ENV] = str(args.worker)
    argv = learner_argv(args.settings, args.worker, url, trainer)
    sys.argv = [sys.argv[0], *argv]  # the trainer binds its identity from sys.argv
    trainer_main(argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if '--' in argv:
        index = argv.index('--')
        argv, extra = argv[:index], argv[index + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    roles = parser.add_subparsers(dest='role', required=True)
    c = roles.add_parser('coordinator')
    c.add_argument('--root', required=True)
    c.add_argument('--parent', required=True)
    c.add_argument('--settings', required=True)
    c.add_argument('--trainer-argv', required=True)
    c.add_argument('--port', type=int, default=8080)
    w = roles.add_parser('learner')
    w.add_argument('--worker', type=int, required=True)
    w.add_argument('--settings', required=True)
    w.add_argument('--out-link', required=True)
    w.add_argument('--out-target', required=True)
    args = parser.parse_args(argv)
    if args.role == 'coordinator':
        coordinator(args, extra)
    else:
        learner(args, extra)


if __name__ == '__main__':
    main()
