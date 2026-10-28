"""Weight EMA of the native trainable parameters (constant-lr training with averaged weights).

Production stages run a constant learning rate and keep an exponential moving average of the weights that
evaluation, selection and release candidates read (plans/neuralese/TRAINING_INSIGHTS_20261028.md section 2).

* The EMA is stored in each parameter's own dtype (BF16 latents give a BF16 EMA: no full-precision copy on the
  LFM3 line). Because ``(1 - decay) * (weight - ema)`` is far below BF16 resolution, BF16 entries are written with
  unbiased stochastic rounding (``optim.stochastic_round_``); deterministic rounding would freeze the average.
* The rounding noise comes from a dedicated ``torch.Generator`` per device, whose state is part of the EMA state.
  The global RNG that the optimizer's own stochastic rounding and the data pipeline consume is never touched, so
  enabling the EMA does not change the training trajectory by a single bit, and an exact resume reproduces the
  EMA bit for bit.
* The EMA is updated after each committed optimizer update and earns no optimizer-update, exposure or epoch
  credit. ``updates`` counts EMA steps only.
* ``applied()`` swaps the EMA into the live parameters and back (in place, tensor by tensor) for evaluation.
  ``export_checkpoint`` writes a model-only checkpoint holding the EMA weights; it is an evaluation / release
  artifact and deliberately not a resumable parent.

    python -m natlang_neuralese.train.weight_ema export --checkpoint CKPT --out DIR [--base BASE]

Declaration (JSON, ``--weight-ema``): ``{"decay": 0.999, "warmup": 10, "eval": "raw"|"ema"|"both", "seed": 0}``.
``warmup`` N >= 2 uses ``min(decay, (1 + n) / (N + n))`` at the n-th update (0 disables it). ``"off"`` explicitly
discards an EMA carried by the parent.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch

from .diloco.protocol import ema_decay_at
from .optim import stochastic_round_

SCHEMA = 'natlang.native-weight-ema/1'
EXPORT_SCHEMA = 'natlang.native-weight-ema-export/1'
EVAL_MODES = ('raw', 'ema', 'both')
CHUNK_ELEMENTS = 1 << 24


def parse_weight_ema(text):
    """Validate the ``--weight-ema`` declaration. None when absent, ``{'off': True}`` when explicitly discarded."""
    if text is None or text == '':
        return None
    if text == 'off':
        return {'off': True}
    spec = json.loads(text) if isinstance(text, str) else dict(text)
    unknown = set(spec) - {'decay', 'warmup', 'eval', 'seed'}
    if unknown:
        raise ValueError('unknown weight EMA options: ' + ', '.join(sorted(unknown)))
    decay = spec.get('decay')
    if type(decay) not in (int, float) or not (math.isfinite(decay) and 0 < decay < 1):
        raise ValueError('weight EMA decay must be a number in (0, 1)')
    warmup = spec.get('warmup', 0)
    if type(warmup) is not int or warmup < 0 or warmup == 1:
        raise ValueError('weight EMA warmup must be 0 (off) or an integer >= 2')
    mode = spec.get('eval', 'raw')
    if mode not in EVAL_MODES:
        raise ValueError('weight EMA eval must be one of ' + ', '.join(EVAL_MODES))
    seed = spec.get('seed', 0)
    if type(seed) is not int:
        raise ValueError('weight EMA seed must be an integer')
    return {'decay': float(decay), 'warmup': warmup, 'eval': mode, 'seed': seed}


decay_at = ema_decay_at  # one schedule for the trainer EMA and the DiLoCo coordinator EMA


class WeightEMA:
    def __init__(self, named, *, decay, warmup=0, seed=0, provenance=None):
        self.named = list(named)
        if not self.named:
            raise ValueError('weight EMA needs trainable parameters')
        self.decay, self.warmup, self.seed = float(decay), int(warmup), int(seed)
        self.updates = 0
        self.tensors = {name: parameter.detach().clone(memory_format=torch.contiguous_format)
                        for name, parameter in self.named}
        self.generators = {}
        self.provenance = {'schema': SCHEMA, 'events': []}
        if provenance:
            self.provenance['events'].append(dict(provenance, event='initialized'))

    # RNG ---------------------------------------------------------------------------------------------------------
    def _generator(self, device):
        key = str(device)
        generator = self.generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            self.generators[key] = generator
        return generator

    # Update ------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def update(self):
        """``ema <- ema + (1 - decay) * (weight - ema)``: FP32 arithmetic, written in the EMA's dtype."""
        weight = 1.0 - decay_at(self.decay, self.warmup, self.updates)
        for name, parameter in self.named:
            target = self.tensors[name].view(-1)
            source = parameter.detach().reshape(-1)
            for start in range(0, target.numel(), CHUNK_ELEMENTS):
                stop = min(start + CHUNK_ELEMENTS, target.numel())
                value = torch.lerp(target[start:stop].float(), source[start:stop].float(), weight)
                window = target[start:stop]
                if window.dtype == torch.bfloat16:
                    stochastic_round_(window, value, self._generator(value.device))
                else:
                    window.copy_(value)
        self.updates += 1

    # Evaluation --------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _swap(self):
        for name, parameter in self.named:
            ema = self.tensors[name]
            held = parameter.detach().clone()
            parameter.copy_(ema)
            ema.copy_(held)

    @contextmanager
    def applied(self):
        """Evaluate with the EMA weights in the live parameters; the raw weights return on exit (exactly)."""
        self._swap()
        try:
            yield self
        finally:
            self._swap()

    @torch.no_grad()
    def load_into(self, named):
        """Copy the EMA into ``named`` parameters (export and release candidates)."""
        for name, parameter in named:
            parameter.copy_(self.tensors[name])

    def distance(self):
        """Relative distance of the EMA from the live weights (diagnostic; one pass, call at telemetry cadence)."""
        with torch.no_grad():
            difference = torch.zeros((), dtype=torch.float64)
            scale = torch.zeros((), dtype=torch.float64)
            for name, parameter in self.named:
                difference += torch.linalg.vector_norm(self.tensors[name].float() - parameter.detach().float()).double().cpu() ** 2
                scale += torch.linalg.vector_norm(parameter.detach().float()).double().cpu() ** 2
        return float((difference / scale.clamp_min(1e-30)).sqrt())

    # State -------------------------------------------------------------------------------------------------------
    def state_dict(self):
        """Live tensors (an asynchronous checkpoint snapshots them); generator states are small host tensors."""
        return {'schema': SCHEMA, 'decay': self.decay, 'warmup': self.warmup, 'seed': self.seed,
                'updates': self.updates, 'tensors': self.tensors,
                'rng': {key: generator.get_state() for key, generator in self.generators.items()},
                'provenance': self.provenance}

    @classmethod
    def restore(cls, named, state, *, decay=None, warmup=None, seed=None, event=None):
        """Continue a saved EMA exactly. ``decay``/``warmup`` may be tuned at a continuation (the averaged values
        and RNG stay). Parameters absent from the saved state start from their current weights, and saved entries
        without a parameter are dropped; both are recorded. A shape change is an error."""
        if state.get('schema') != SCHEMA:
            raise ValueError('unsupported weight EMA state')
        named = list(named)
        result = cls(named, decay=state['decay'] if decay is None else decay,
                     warmup=state['warmup'] if warmup is None else warmup,
                     seed=state['seed'] if seed is None else seed)
        result.updates = int(state['updates'])
        result.provenance = json.loads(json.dumps(state.get('provenance') or {'schema': SCHEMA, 'events': []}))
        saved = state['tensors']
        reinitialized, recast = [], []
        for name, parameter in named:
            value = saved.get(name)
            if value is None:
                reinitialized.append(name)
                continue
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f'saved weight EMA entry {name} has shape {tuple(value.shape)}, '
                                 f'the parameter has {tuple(parameter.shape)}')
            if value.dtype != parameter.dtype:
                recast.append(name)
            result.tensors[name] = value.to(device=parameter.device, dtype=parameter.dtype, copy=True).contiguous()
        dropped = sorted(set(saved) - {name for name, _ in named})
        for key, generator_state in (state.get('rng') or {}).items():
            try:
                device = torch.device(key)
                if device.type == 'cuda' and not torch.cuda.is_available():
                    continue
                result._generator(device).set_state(generator_state)
            except (RuntimeError, ValueError):
                continue
        notes = {'event': 'restored', 'updates': result.updates}
        if reinitialized:
            notes['reinitialized'] = reinitialized
        if dropped:
            notes['dropped'] = dropped
        if recast:
            notes['recast_to_parameter_dtype'] = recast
        if event:
            notes.update(event)
        result.provenance['events'].append(notes)
        return result

    def summary(self):
        return {'schema': SCHEMA, 'decay': self.decay, 'warmup': self.warmup, 'updates': self.updates,
                'effective_decay': decay_at(self.decay, self.warmup, self.updates),
                'storage_dtypes': sorted({str(value.dtype) for value in self.tensors.values()})}


# Export -----------------------------------------------------------------------------------------------------------

def write_weights_checkpoint(model, config, schema, step, export, out):
    """Model-only native checkpoint of the weights now in ``model`` with its EMA export provenance."""
    from ..model.native_model import model_state
    from .checkpoint_policy import CheckpointPolicy
    CheckpointPolicy(out).save({'schema': schema, 'model_config': config, **model_state(model),
                                'step': step, 'ema_export': export})


def export_checkpoint(checkpoint, out, *, base=None, device='cpu', provenance=None):
    """Write a model-only native checkpoint whose trainable weights are the saved EMA (never a resumable parent)."""
    from ..model.native_model import load_native_checkpoint
    from .native_state_sft import native_trainable_parameters
    from ..common.hashing import sha256_file_hex_cached

    model, _, _, config, state = load_native_checkpoint(checkpoint, base=base, device=device)
    training = state.get('training') or {}
    saved = training.get('ema')
    if saved is None:
        raise ValueError('checkpoint carries no weight EMA state')
    named = native_trainable_parameters(model)
    ema = WeightEMA.restore(named, saved)
    missing = [event for event in ema.provenance['events'][-1:] if event.get('reinitialized')]
    if missing:
        raise ValueError('checkpoint EMA does not cover every trainable parameter: '
                         + ', '.join(missing[0]['reinitialized'][:5]))
    ema.load_into(named)
    step = training.get('global_step', training.get('step', 0))
    export = {'schema': EXPORT_SCHEMA, 'source_checkpoint_sha256': sha256_file_hex_cached(checkpoint),
              'source_global_step': step, 'ema': ema.summary(), 'ema_events': ema.provenance['events'],
              **(provenance or {})}
    write_weights_checkpoint(model, config, state['schema'], step, export, out)
    return export


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('--checkpoint', type=Path, required=True)
    export.add_argument('--out', type=Path, required=True)
    export.add_argument('--base', type=Path)
    export.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    print(json.dumps(export_checkpoint(args.checkpoint, args.out, base=args.base, device=args.device),
                     sort_keys=True))


if __name__ == '__main__':
    main()
