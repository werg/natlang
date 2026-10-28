"""Learning-rate anneal branch for release / qualification candidates (TRAINING_INSIGHTS_20261028.md section 2).

The main lineage runs a constant learning rate and never commits to an end step. A release candidate comes from a
short anneal *branch*: a separate continuation of the live full state (weights, optimizer history, RNG, exposure
cursor, weight EMA) that decays the learning rate over a declared number of updates and writes only to its own
output directory. Nothing in the main lineage's directory, cursor or checkpoint is modified, and the branch is marked
in its checkpoint options and reports so that no controller adopts it as the earned parent.

Trainer option ``--lr-anneal`` (JSON)::

    {"steps": 200, "shape": "linear" | "cosine", "floor": 0.0}

For the anneal's k-th update (k = 0 .. steps-1) every optimizer group trains at ``base_lr * f(k / steps)`` with
``linear``: ``floor + (1 - floor) * (1 - x)``; ``cosine``: ``floor + (1 - floor) * (1 + cos(pi x)) / 2``.
``floor`` is a fraction of the base rate. After the update the groups return to the base rates, so the saved
optimizer state is shaped exactly like the lineage's. The trainer stops after ``steps`` updates.

Launcher (snapshots the live checkpoint, runs the trainer with the lineage's recorded options, exports the EMA)::

    python -m natlang_neuralese.train.lr_anneal run --from LIVE_CKPT --out BRANCH_DIR \\
        --steps 200 [--shape cosine] [--floor 0.0] [-- extra trainer arguments]
    python -m natlang_neuralese.train.lr_anneal run ... --print-argv      # snapshot + argv only
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

SCHEMA = 'natlang.native-lr-anneal/1'
SHAPES = ('linear', 'cosine')


def parse_lr_anneal(text):
    if text is None or text == '':
        return None
    spec = json.loads(text) if isinstance(text, str) else dict(text)
    unknown = set(spec) - {'steps', 'shape', 'floor'}
    if unknown:
        raise ValueError('unknown lr anneal options: ' + ', '.join(sorted(unknown)))
    steps = spec.get('steps')
    if type(steps) is not int or steps < 1:
        raise ValueError('lr anneal steps must be a positive integer')
    shape = spec.get('shape', 'linear')
    if shape not in SHAPES:
        raise ValueError('lr anneal shape must be one of ' + ', '.join(SHAPES))
    floor = spec.get('floor', 0.0)
    if type(floor) not in (int, float) or not (math.isfinite(floor) and 0 <= floor < 1):
        raise ValueError('lr anneal floor is a fraction of the base rate in [0, 1)')
    return {'steps': steps, 'shape': shape, 'floor': float(floor)}


def multiplier(spec, local_update):
    """Rate multiplier of the anneal's ``local_update``-th update (0-based)."""
    progress = min(max(local_update / spec['steps'], 0.0), 1.0)
    floor = spec['floor']
    if spec['shape'] == 'linear':
        return floor + (1 - floor) * (1 - progress)
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


class LrAnneal:
    """Scales every optimizer group around its base rate for one update at a time."""

    def __init__(self, optimizer, spec, *, start_global_step):
        from .optim_restore import _children
        self.spec, self.start = spec, int(start_global_step)
        self.groups = [(group, group['lr'], group.get('qat_latent_lr'))
                       for _, child in _children(optimizer) for group in child.param_groups]
        if not self.groups:
            raise ValueError('lr anneal needs optimizer groups')

    def remaining(self, global_step):
        return max(0, self.spec['steps'] - (int(global_step) - self.start))

    def scale(self, global_step):
        """Set the groups for the update that follows ``global_step`` earlier updates; returns the multiplier."""
        factor = multiplier(self.spec, int(global_step) - self.start)
        for group, base, qat in self.groups:
            group['lr'] = base * factor
            if qat is not None:
                group['qat_latent_lr'] = qat * factor
        return factor

    def restore(self):
        for group, base, qat in self.groups:
            group['lr'] = base
            if qat is not None:
                group['qat_latent_lr'] = qat


class LrWarmup:
    """Linear learning-rate warmup over the lineage's first ``updates`` optimizer updates (``--lr-warmup N``).

    The update that follows ``global_step`` earlier updates trains every group at ``rate * min(1, (global_step + 1) /
    N)``; from update N on the rate is the declared constant. The factor multiplies whatever rate the group holds for
    that update (so it composes with an anneal) and the groups get their rates back afterwards, so the saved optimizer
    state is shaped exactly like the lineage's. It is a function of the lineage ``global_step`` only: a resume or a
    continuation past N trains at the constant rate, and a DiLoCo learner warms up over its own local updates."""

    def __init__(self, optimizer, updates):
        from .optim_restore import _children
        if type(updates) is not int or updates < 0:
            raise ValueError('lr warmup is a non-negative number of updates')
        self.updates = updates
        self.groups = [group for _, child in _children(optimizer) for group in child.param_groups]
        self.saved = None

    def factor(self, global_step):
        return 1.0 if self.updates == 0 else min(1.0, (int(global_step) + 1) / self.updates)

    def scale(self, global_step):
        factor = self.factor(global_step)
        if factor < 1.0:
            self.saved = [(group, group['lr'], group.get('qat_latent_lr')) for group in self.groups]
            for group, base, qat in self.saved:
                group['lr'] = base * factor
                if qat is not None:
                    group['qat_latent_lr'] = qat * factor
        return factor

    def restore(self):
        for group, base, qat in self.saved or ():
            group['lr'] = base
            if qat is not None:
                group['qat_latent_lr'] = qat
        self.saved = None


# Launcher ---------------------------------------------------------------------------------------------------------

# Production publication, resident adoption and DiLoCo membership belong to the main lineage.
DROPPED_OPTIONS = {'resident_data_directory', 'resident_geometry_qualification', 'diloco', 'checkpoint_publication',
                   'checkpoint_publication_namespace', 'checkpoint_publication_readonly_input_roots',
                   'continue_from', 'out', 'lr_anneal', 'refresh_after_updates', 'refresh_at_checkpoint',
                   'foundation_qualification_request', 'source_correction', 'native_optimizer_migration'}


def trainer_argv(options, parser):
    """Rebuild trainer arguments from the options a checkpoint recorded (only declared parser destinations)."""
    argv = []
    for action in parser._actions:
        dest = action.dest
        if dest in ('help',) or dest in DROPPED_OPTIONS or dest not in options or not action.option_strings:
            continue
        value = options[dest]
        flag = action.option_strings[0]
        if value is None or value == action.default:
            continue
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(flag)
        elif isinstance(action, argparse.BooleanOptionalAction):
            argv.append(flag if value else '--no-' + flag[2:])
        elif isinstance(action, argparse._AppendAction):
            for item in value:
                argv += [flag, str(item)]
        else:
            argv += [flag, value if isinstance(value, str) else json.dumps(value) if isinstance(value, (dict, list)) else str(value)]
    return argv


def snapshot_parent(source, destination):
    """Copy the live full-state checkpoint (atomically replaced by its writer) with a before/after identity check."""
    from ..common.hashing import sha256_file_hex
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        before = source.stat()
        temporary = destination.with_name(destination.name + '.partial')
        shutil.copyfile(source, temporary)
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns):
            temporary.replace(destination)
            return sha256_file_hex(destination)
        temporary.unlink()
    raise RuntimeError('the live checkpoint changed while it was being copied; retry at a quieter boundary')


def plan_branch(source, out, spec, *, extra_argv=()):
    """Snapshot ``source`` into ``out`` and return the branch trainer argv and provenance."""
    from .checkpoint_policy import CheckpointPolicy
    from .native_state_sft import build_parser
    source, out = Path(source).resolve(strict=True), Path(out).resolve()
    lineage = source.parent
    if out == lineage or lineage in out.parents or out in lineage.parents:
        raise ValueError('the anneal branch needs its own namespace outside the main lineage directory')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'anneal branch namespace {out} is not empty')
    state = CheckpointPolicy.load(source, map_location='cpu', semantic_only=True)
    training = state.get('training') or {}
    options = training.get('options') or {}
    if not options:
        raise ValueError('checkpoint lacks the recorded trainer options')
    parent = out / 'anneal-parent' / 'checkpoint.pt'
    parent_sha = snapshot_parent(source, parent)
    argv = trainer_argv(options, build_parser())
    argv += ['--continue-from', str(parent), '--out', str(out / 'branch'),
             '--lr-anneal', json.dumps(spec, sort_keys=True), *extra_argv]
    provenance = {'schema': SCHEMA, 'anneal': spec, 'main_lineage_directory': str(lineage),
                  'source_checkpoint': str(source), 'parent_snapshot': str(parent),
                  'parent_checkpoint_sha256': parent_sha,
                  'parent_global_step': training.get('global_step', training.get('step')),
                  'parent_cursor': training.get('cursor'), 'parent_has_weight_ema': 'ema' in training,
                  'main_lineage_modified': False, 'argv': argv}
    (out / 'anneal-provenance.json').write_text(json.dumps(provenance, sort_keys=True, indent=1) + '\n')
    return argv, provenance


def run(args, extra):
    spec = parse_lr_anneal(json.dumps({'steps': args.steps, 'shape': args.shape, 'floor': args.floor}))
    argv, provenance = plan_branch(args.source, args.out, spec, extra_argv=extra)
    if args.print_argv:
        print(json.dumps(provenance, sort_keys=True))
        return 0
    from .native_state_sft import main as train
    status = train(argv)
    branch = Path(args.out) / 'branch'
    if status == 0 and provenance['parent_has_weight_ema'] and (branch / 'checkpoint.pt').exists():
        from .weight_ema import export_checkpoint
        candidate = Path(args.out) / 'release-candidate-ema'
        export = export_checkpoint(branch / 'checkpoint.pt', candidate, base=args.base, device='cpu',
                                   provenance={'anneal_branch': provenance})
        print(json.dumps({'release_candidate_ema': str(candidate), **export}, sort_keys=True))
    return status


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if '--' in argv:
        split = argv.index('--')
        argv, extra = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    branch = commands.add_parser('run')
    branch.add_argument('--from', dest='source', type=Path, required=True)
    branch.add_argument('--out', type=Path, required=True)
    branch.add_argument('--steps', type=int, required=True)
    branch.add_argument('--shape', choices=SHAPES, default='linear')
    branch.add_argument('--floor', type=float, default=0.0)
    branch.add_argument('--base', type=Path)
    branch.add_argument('--print-argv', action='store_true')
    args = parser.parse_args(argv)
    return run(args, extra)


if __name__ == '__main__':
    raise SystemExit(main())
