"""The recipe ``quantization`` component: QAT woven into every training stage (owner 2026-10-10).

A recipe declares precision points once at its top level; every stage named in ``stage_progress`` inherits them. The
backbone trains as BF16 latents (``latent`` policy, or ``full`` on dense backbones); each point is a fake-quantized
view of the same latents, ramped in over curriculum progress, and the stage's objective is taken under several
points per update. BF16 stays the teacher: every forward outside a point's context (references, baselines, the BF16
gate column) sees the unquantized latents. Checkpoints hold the latents and optimizer state only: a point is a
function of the latents and the step.

Component (JSON, validated by ``validate_spec``)::

    {"schema": "natlang.neuralese-quantization/1",
     "loss": {"mode": "sum" | "sample", "bf16_weight": 1.0, "seed": 0},
     "stage_progress": {"core_text_warmup": [0.0, 0.5], "recurrence_warmup": [0.5, 1.0]},
     "points": [
       {"name": "int4", "gate": "required" | "report" | "none", "weight": 0.5,
        "ramp": {"start": 0.2, "end": 0.4},
        "modules": {"experts": {"format": "int4", "group": 32}, "mlp": {"format": "int4", "group": 32}}},
       {"name": "ternary-experts", "gate": "report", "weight": 1.0, "ramp": {"start": 0.4, "end": 0.8},
        "modules": {"experts": {"format": "ternary", "nested_group": 32}, "attention": {"format": "int4"}},
        "layers": {"least_sensitive": 0.5, "probe": {"path": "...", "sha256": "..."}}}]}

- Progress: a stage spans ``stage_progress[stage] = [p0, p1]`` of the curriculum; at update ``step`` of an update
  range ``[start, end)`` its progress is ``p0 + (p1 - p0) * (step - start) / (end - start)``. Stages not named run at
  BF16 (the foundation's frozen-backbone stages).
- A point's mix ``λ`` ramps linearly from 0 at ``ramp.start`` to 1 at ``ramp.end``; a point with ``λ = 0`` is
  inactive. Its forward value is ``w + λ·(Q(w) − w)`` per selected module group, straight-through to the latent.
- Loss ``sum``: ``bf16_weight·L(bf16) + Σ weight_p·L(p)`` over active points (each its own forward/backward).
  ``sample``: one of {bf16} ∪ active points per update, drawn ∝ its weight from ``Random((seed, step))``, its loss
  scaled by the total weight (an unbiased estimate of the sum at the cost of one pass).
- Gates: every gated point (``required``/``report``) is evaluated at its deploy precision (``λ = 1``) next to the
  BF16 column. A stage requires the ``required`` points whose ramp ends within its progress range (``ramp.end ≤ p1``)
  to pass the same thresholds as BF16; its report names the precisions that passed.
- Module groups: ``experts`` (MoE experts), ``attention``, ``conv`` (LFM short convolutions), ``mlp`` (dense FFN),
  ``router``, ``embeddings``, ``heads`` (an untied output projection). A group a backbone lacks selects nothing, so
  one component serves both lines (Mellum: experts + attention; LFM: mlp + conv + attention).
- ``layers``: ``"all"`` (default), a list of layer indices, or the least sensitive fraction of layers according to a
  pinned sensitivity probe (``python -m natlang_neuralese.train.quantization sensitivity``): experts first, attention later or never.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import parametrize

from ..common.hashing import sha256_file_hex

SCHEMA = 'natlang.neuralese-quantization/1'
FORMATS = ('bf16', 'int4', 'ternary')
GROUPS = ('experts', 'attention', 'conv', 'mlp', 'router', 'embeddings', 'heads')
GATES = ('required', 'report', 'none')
STAGE_KINDS = ('core_text_warmup', 'raw_recurrence_training')


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_spec(spec, stages=None):
    """Validate a quantization component. ``stages`` maps stage id -> kind (the recipe's), when known."""
    if not isinstance(spec, dict) or spec.get('schema') != SCHEMA:
        raise ValueError('quantization must be an object with schema ' + SCHEMA)
    unknown = set(spec) - {'schema', 'loss', 'stage_progress', 'points', 'description', 'preserve', 'gate'}
    if unknown:
        raise ValueError('unknown quantization keys: ' + ', '.join(sorted(unknown)))
    loss = spec.get('loss', {})
    if not isinstance(loss, dict) or set(loss) - {'mode', 'bf16_weight', 'seed'}:
        raise ValueError('quantization.loss has mode, bf16_weight and seed')
    if loss.get('mode', 'sum') not in ('sum', 'sample'):
        raise ValueError('quantization.loss.mode must be sum or sample')
    if not _number(loss.get('bf16_weight', 1.0)) or loss.get('bf16_weight', 1.0) < 0:
        raise ValueError('quantization.loss.bf16_weight must be finite and nonnegative')
    if type(loss.get('seed', 0)) is not int:
        raise ValueError('quantization.loss.seed must be an integer')
    progress = spec.get('stage_progress', {})
    if not isinstance(progress, dict):
        raise ValueError('quantization.stage_progress maps stage ids to [start, end]')
    for stage, span in progress.items():
        if (not isinstance(span, list) or len(span) != 2 or not all(_number(v) for v in span)
                or not 0 <= span[0] <= span[1] <= 1):
            raise ValueError('stage progress must be [start, end] with 0 <= start <= end <= 1: ' + str(stage))
        if stages is not None:
            if stage not in stages:
                raise ValueError('quantization names an unknown stage: ' + str(stage))
            if stages[stage] not in STAGE_KINDS:
                raise ValueError('quantization applies to trained-backbone stages only, not ' + str(stage))
    preserve = spec.get('preserve')
    if preserve is not None:
        if (not isinstance(preserve, dict) or set(preserve) - {'teacher', 'weight', 'prompt_weight', 'max_tokens', 'seed'}
                or not isinstance(preserve.get('teacher'), dict) or set(preserve['teacher']) != {'artifact'}
                or not _number(preserve.get('weight', 0)) or preserve.get('weight', 0) < 0
                or not _number(preserve.get('prompt_weight', 0.25)) or not 0 <= preserve.get('prompt_weight', 0.25) <= 1
                or type(preserve.get('max_tokens', 4096)) is not int or preserve.get('max_tokens', 4096) < 16):
            raise ValueError('quantization.preserve is {teacher: {artifact}, weight, prompt_weight, max_tokens, seed}')
    gate = spec.get('gate')
    if gate is not None:
        if (not isinstance(gate, dict) or set(gate) - {'probes', 'held_rows', 'prompt_weight', 'max_gate_drop',
                                                       'max_held_kl', 'reference'}
                or not isinstance(gate.get('reference', {}), dict) or set(gate.get('reference', {})) - {'gate_pass', 'source'}
                or not isinstance(gate.get('probes'), dict) or set(gate['probes']) != {'path', 'sha256'}
                or not all(_number(gate.get(k, 0)) and gate.get(k, 0) >= 0
                           for k in ('max_gate_drop', 'max_held_kl', 'prompt_weight'))
                or type(gate.get('held_rows', 48)) is not int):
            raise ValueError('quantization.gate is {probes: {path, sha256}, held_rows, prompt_weight, max_gate_drop, '
                             'max_held_kl}')
        if 'max_held_kl' in gate and preserve is None:
            raise ValueError('quantization.gate.max_held_kl needs the preserve teacher (its held records)')
    points = spec.get('points')
    if not isinstance(points, list) or not points:
        raise ValueError('quantization.points must be a nonempty list')
    names = set()
    for point in points:
        if not isinstance(point, dict) or set(point) - {'name', 'gate', 'weight', 'ramp', 'modules', 'layers'}:
            raise ValueError('a precision point has name, gate, weight, ramp, modules and optional layers')
        name = point.get('name')
        if not isinstance(name, str) or not name or name == 'bf16' or name in names:
            raise ValueError('precision point names are unique, nonempty and not bf16')
        names.add(name)
        if point.get('gate', 'report') not in GATES:
            raise ValueError('precision point gate must be required, report or none')
        if not _number(point.get('weight', 1.0)) or point.get('weight', 1.0) < 0:
            raise ValueError('precision point weight must be finite and nonnegative')
        ramp = point.get('ramp')
        if (not isinstance(ramp, dict) or set(ramp) != {'start', 'end'} or not all(_number(v) for v in ramp.values())
                or not 0 <= ramp['start'] < ramp['end'] <= 1):
            raise ValueError('precision point ramp is {start, end} with 0 <= start < end <= 1: ' + name)
        modules = point.get('modules')
        if not isinstance(modules, dict) or not modules or set(modules) - set(GROUPS):
            raise ValueError('precision point modules map module groups to formats: ' + name)
        for group, config in modules.items():
            if not isinstance(config, dict) or config.get('format') not in FORMATS or \
                    set(config) - {'format', 'group', 'nested_group'}:
                raise ValueError(f'module {group} of {name} needs a format in {FORMATS}')
            for key in ('group', 'nested_group'):
                if key in config and (type(config[key]) is not int or config[key] < 1):
                    raise ValueError(f'{key} must be a positive integer')
            if 'nested_group' in config and config['format'] != 'ternary':
                raise ValueError('nested_group applies to ternary (the coarsest level of the int4 grid)')
        layers = point.get('layers', 'all')
        if not (layers == 'all' or (isinstance(layers, list) and all(type(i) is int and i >= 0 for i in layers))
                or (isinstance(layers, dict) and set(layers) == {'least_sensitive', 'probe'}
                    and _number(layers['least_sensitive']) and 0 < layers['least_sensitive'] <= 1
                    and isinstance(layers['probe'], dict) and set(layers['probe']) == {'path', 'sha256'})):
            raise ValueError('precision point layers is "all", a list of layer indices, or '
                             '{"least_sensitive": fraction, "probe": {"path", "sha256"}}')
    return spec


def stage_component(spec, stage, *, start, end, init_mix=None):
    """The component as one stage receives it (``--quantization``): the recipe's spec plus the stage's progress range
    and update range. ``None`` when the stage is not named in ``stage_progress`` (it trains at BF16)."""
    if not spec or stage not in spec.get('stage_progress', {}):
        return None
    if type(start) is not int or type(end) is not int or end <= start:
        raise ValueError('a quantized stage needs an update range start < end')
    span = spec['stage_progress'][stage]
    component = {**json.loads(json.dumps(spec)), 'stage': {'id': stage, 'progress': list(span), 'start': start,
                                                          'end': end}}
    if init_mix:
        component['init_mix'] = dict(init_mix)
    return component


def ramp_mix(ramp, progress):
    """λ of a point at curriculum ``progress``: 0 before ``start``, 1 after ``end``, linear in between."""
    if progress <= ramp['start']:
        return 0.0
    if progress >= ramp['end']:
        return 1.0
    return (progress - ramp['start']) / (ramp['end'] - ramp['start'])


def classify(name, module):
    """The module group of a named backbone module, or None."""
    from ..maple.model import DenseExperts

    if isinstance(module, DenseExperts) or ('.experts.' in name + '.' and isinstance(module, nn.Linear)):
        return 'experts'
    if isinstance(module, nn.Embedding):
        return 'embeddings' if 'embed' in name else None
    if not isinstance(module, nn.Linear):
        return None
    last = name.rsplit('.', 1)[-1]
    if name.endswith('lm_head') or last == 'lm_head':
        return 'heads'
    if last in ('gate', 'router') and ('.mlp' in name or '.feed_forward' in name or 'router' in name):
        return 'router'
    if '.self_attn.' in name or '.attn.' in name or '.attention.' in name:
        return 'attention'
    if '.conv.' in name:
        return 'conv'
    if '.mlp.' in name or '.feed_forward.' in name:
        return 'mlp'
    return None


def layer_of(name):
    marker = 'layers.'
    if marker not in name:
        return None
    try:
        return int(name.split(marker, 1)[1].split('.', 1)[0])
    except ValueError:
        return None


def selected_layers(point, layer_count, *, root=None):
    """The layer indices a point quantizes (None = every module, including non-layer ones)."""
    layers = point.get('layers', 'all')
    if layers == 'all':
        return None
    if isinstance(layers, list):
        return set(layers)
    probe = layers['probe']
    path = Path(probe['path'])
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    if sha256_file_hex(path) != probe['sha256']:
        raise ValueError('sensitivity probe changed since the recipe pinned it: ' + str(path))
    scores = json.loads(path.read_text())['layers']  # {layer index: sensitivity}, larger = more sensitive
    ranked = sorted(range(layer_count), key=lambda i: (scores.get(str(i), math.inf), i))
    return set(ranked[:max(1, round(layers['least_sensitive'] * layer_count))])


class Quantization:
    """One stage's precision schedule over a backbone module (``backbone.hf``)."""

    def __init__(self, component, *, root=None):
        validate_spec({k: v for k, v in component.items() if k not in ('stage', 'init_mix')})
        self.init_mix = dict(component.get('init_mix') or {})
        if set(self.init_mix) - {p['name'] for p in component['points']} or \
                not all(_number(v) and 0 <= v <= 1 for v in self.init_mix.values()):
            raise ValueError('init_mix maps declared points to a starting λ in [0, 1]')
        stage = component.get('stage')
        if not isinstance(stage, dict) or set(stage) != {'id', 'progress', 'start', 'end'}:
            raise ValueError('a stage quantization component carries its stage id, progress and update range')
        self.component, self.stage = component, stage
        self.points = {p['name']: p for p in component['points']}
        loss = component.get('loss', {})
        self.mode, self.bf16_weight, self.seed = loss.get('mode', 'sum'), float(loss.get('bf16_weight', 1.0)), loss.get('seed', 0)
        self.root = root
        self.installed = {}  # group -> number of modules
        self.layers = {}  # point -> selected layer set or None

    # Schedule --------------------------------------------------------------------------------------------------------
    def progress(self, step):
        p0, p1 = self.stage['progress']
        start, end = self.stage['start'], self.stage['end']
        fraction = min(1.0, max(0.0, (step - start) / (end - start)))
        return p0 + (p1 - p0) * fraction

    def mixes(self, step):
        progress = self.progress(step)
        # A lineage initialized from latents trained at some λ (init.ramp_floor, e.g. conversion v3) keeps that point
        # at least at its starting λ: the ramp position carries over.
        return {name: max(self.init_mix.get(name, 0.0), ramp_mix(point['ramp'], progress))
                for name, point in self.points.items()}

    def active(self, step):
        """Active points (λ > 0) with their λ, in declaration order."""
        return [(name, mix) for name, mix in self.mixes(step).items() if mix > 0]

    def resolve(self, name, mix):
        """The ``PRECISION["groups"]`` value of point ``name`` at mix ``mix`` (None = BF16)."""
        if name in (None, 'bf16'):
            return None
        point = self.points[name]
        layers = self.layers.get(name)
        groups = {}
        for group, config in point['modules'].items():
            options = {k: v for k, v in config.items() if k != 'format'}
            if layers is not None:
                options['layers'] = layers
            groups[group] = (config['format'], float(mix), options)
        return groups

    def plan(self, step):
        """The precision passes of update ``step``: [(point name or 'bf16', λ, loss weight)]."""
        candidates = [('bf16', 0.0, self.bf16_weight)] + [(n, m, float(self.points[n].get('weight', 1.0)))
                                                         for n, m in self.active(step)]
        candidates = [c for c in candidates if c[2] > 0]
        if not candidates:
            raise ValueError('no precision point has a positive loss weight at step ' + str(step))
        if self.mode == 'sum' or len(candidates) == 1:
            return candidates
        return [self.sampled(step)]

    def sampled(self, step):
        """One precision pass for ``step`` drawn by weight, whatever the loss mode (the recurrence trainer's single
        pass per update): (point, λ, weight sum). Its expected gradient is the weight-normalized mixture."""
        candidates = [('bf16', 0.0, self.bf16_weight)] + [(n, m, float(self.points[n].get('weight', 1.0)))
                                                         for n, m in self.active(step)]
        candidates = [c for c in candidates if c[2] > 0]
        total = sum(c[2] for c in candidates)
        draw = random.Random(f'{self.seed}:{self.stage["id"]}:{step}').random() * total
        for name, mix, weight in candidates:
            draw -= weight
            if draw < 0:
                return name, mix, total
        return candidates[-1][0], candidates[-1][1], total

    def set_active(self, name, mix):
        """Set the active precision outright (no nesting): ``bf16`` clears it."""
        from ..maple.ternary import PRECISION
        PRECISION['groups'], PRECISION['point'] = self.resolve(name, mix), (None if name == 'bf16' else name)

    def context(self, name, mix):
        """Context manager running forwards at point ``name`` with mix ``mix`` (``bf16``: unquantized)."""
        from ..maple.ternary import active_precision
        return active_precision(self.resolve(name, mix), None if name == 'bf16' else name)

    def gated_points(self):
        return [name for name, point in self.points.items() if point.get('gate', 'report') != 'none']

    def required_points(self):
        """Gated ``required`` points whose ramp completes inside this stage's progress range."""
        end = self.stage['progress'][1]
        return [name for name, point in self.points.items()
                if point.get('gate', 'report') == 'required' and point['ramp']['end'] <= end]

    def describe(self, step):
        return {'schema': SCHEMA, 'stage': self.stage['id'], 'progress': self.progress(step),
                'mixes': self.mixes(step), 'init_mix': dict(self.init_mix), 'mode': self.mode,
                'installed': dict(self.installed),
                'required': self.required_points(), 'gated': self.gated_points()}

    # Installation ----------------------------------------------------------------------------------------------------
    def install(self, module, layer_count=None):
        """Route every selected module's weight through the active precision point: a ``PrecisionSTE``
        parametrization on linear/embedding weights, ``precision_group`` on dense MoE experts. Weights stay the same
        trainable parameters (latents); with no active point they are untouched."""
        from ..maple.model import DenseExperts
        from ..maple.ternary import PrecisionSTE

        wanted = {}
        for name, point in self.points.items():
            for group in point['modules']:
                wanted.setdefault(group, set()).add(name)
        if layer_count is None:
            layer_count = 1 + max((layer_of(n) or 0) for n, _ in module.named_modules())
        for name, point in self.points.items():
            self.layers[name] = selected_layers(point, layer_count, root=self.root)
        seen = set()
        for name, child in module.named_modules():
            group = classify(name, child)
            if group not in wanted:
                continue
            layer = layer_of(name)
            # Keyed ``group@layer``: a point's layer selection (``options["layers"]``) decides per pass.
            if isinstance(child, DenseExperts):
                if not isinstance(child.gate_up, nn.Parameter):
                    child.make_latent(quantize=False)
                child.precision_group = self._group_key(group, layer)
            else:
                weight = child.weight
                if id(weight) in seen:
                    continue  # tied weights are parametrized once
                seen.add(id(weight))
                if parametrize.is_parametrized(child, 'weight'):
                    raise ValueError('module already parametrized; precision points need plain latents: ' + name)
                parametrize.register_parametrization(child, 'weight', PrecisionSTE(self._group_key(group, layer)),
                                                     unsafe=True)
            self.installed[group] = self.installed.get(group, 0) + 1
        return self.installed

    @staticmethod
    def _group_key(group, layer):
        return group if layer is None else f'{group}@{layer}'



def teacher_shards(artifact):
    """The registered BF16 teacher top-k shards (``qat_convert teacher --records``): (train, held, meta), every shard
    checked against the artifact manifest."""
    from ..artifacts import resolve
    from ..maple.qat_convert import _load_record_shards

    meta_path, _ = resolve(artifact, 'teacher.json')
    meta = json.loads(meta_path.read_text())
    for name in meta['shards']['train'] + meta['shards']['test']:
        resolve(artifact, name)
    directory = meta_path.parent
    return (_load_record_shards(directory, meta['shards']['train']),
            _load_record_shards(directory, meta['shards']['test']), meta)


class Preserve:
    """The behaviour-preservation stream of a quantization component: KL to the BF16 model's own top-k on its own
    chat/thinking/tool-call responses (``preserve.teacher``, e.g. conversion v3's teacher shards), one record per
    update at ``preserve.weight``, under whatever precision the update runs. It keeps the backbone's behaviour while
    the precision ramps; the BF16 teacher is fixed data, so no second model is held."""

    def __init__(self, config, stage_id, shards=None):
        self.weight = float(config.get('weight', 0.0))
        self.prompt_weight = float(config.get('prompt_weight', 0.25))
        self.max_tokens = int(config.get('max_tokens', 4096))
        self.seed, self.stage_id = config.get('seed', 0), stage_id
        self.train, self.held, self.meta = shards if shards is not None else teacher_shards(config['teacher']['artifact'])

    def record(self, step):
        return random.Random(f'preserve:{self.seed}:{self.stage_id}:{step}').randrange(len(self.train['ids']))

    def applies(self, model, step) -> bool:
        """Whether the stream has a gradient path at ``step``: it depends only on the backbone, so while every backbone
        weight is frozen (a projection-first phase) it is skipped, logged once per change, never given a dummy grad."""
        active = any(parameter.requires_grad for parameter in model.parameters())
        if active != getattr(self, '_active', None):
            self._active = active
            print(json.dumps({'event': 'preserve_applied' if active else 'preserve_skipped', 'step': step,
                              'stage': self.stage_id, 'reason': None if active else 'no trainable backbone weight'}),
                  flush=True)
        return active

    def loss(self, model, step):
        """(weighted KL loss, record index) of update ``step``; ``model`` is the causal LM (``backbone.hf``)."""
        from ..maple.qat_convert import weighted_topk_kl

        index = self.record(step)
        ids = self.train['ids'][index][:self.max_tokens]
        device = next(model.parameters()).device
        inner = getattr(model, 'model', None)
        if inner is not None and hasattr(inner, 'checkpoint_layers'):
            inner.checkpoint_layers = True  # a long record through every layer: recompute rather than keep
        positions = len(ids) - 1  # cropped records keep the teacher rows of their own positions
        scores = weighted_topk_kl(model, ids[None].long().to(device), self.train['top_ids'][index][:positions],
                                  self.train['top_logp'][index][:positions],
                                  min(self.train['prompt_len'][index], len(ids)), self.prompt_weight)
        return self.weight * scores['loss_kl'], {'preserve_kl': float(scores['loss_kl'].detach()),
                                                  'preserve_record': index}


class Behaviour:
    """The per-precision behaviour gate (``quantization.gate``): conversion v3's generation gate and held KL
    (``maple.generation_gate.behaviour_report``) at the active precision; a precision passes when its generation pass
    rate is at most ``max_gate_drop`` below the BF16 column's and its held KL is at most ``max_held_kl``."""

    def __init__(self, config, tokenizer, held=None, *, root=None):
        from ..maple.distill_data import render

        path = Path(config['probes']['path'])
        if not path.is_absolute() and root is not None:
            path = Path(root) / path
        if sha256_file_hex(path) != config['probes']['sha256']:
            raise ValueError('gate probes changed since the recipe pinned them: ' + str(path))
        self.probes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.config, self.tokenizer, self.held, self.render = config, tokenizer, held, render
        self.eos = tokenizer.convert_tokens_to_ids('<|im_end|>')

    def report(self, model):
        from ..maple.generation_gate import behaviour_report
        return behaviour_report(model, self.tokenizer, self.probes, self.render, self.eos, self.held,
                                self.config.get('held_rows', 48), self.config.get('prompt_weight', 0.25))

    def passed(self, report, bf16=None):
        """Against ``bf16`` (a live BF16 report) or the declared ``reference`` (the original BF16 model's measured
        generation pass rate, e.g. 0.75 at conversion v3's step 0), so a stage's own BF16 column is gated too."""
        bf16 = bf16 or self.config['reference']
        ok = report['gate_pass'] >= bf16['gate_pass'] - self.config.get('max_gate_drop', 0.0) - 1e-9
        if 'max_held_kl' in self.config:
            ok = ok and report.get('held_kl', math.inf) <= self.config['max_held_kl']
        return bool(ok)


def load_component(value, *, root=None):
    """A stage's ``--quantization`` value (JSON text or a path to JSON) as a ``Quantization`` (None when absent)."""
    if value in (None, ''):
        return None
    text = Path(value).read_text() if not value.lstrip().startswith('{') else value
    return Quantization(json.loads(text), root=root)


def multi_precision_backward(quantization, step, passes, refresh=None):
    """Run ``passes(name)`` (yielding (loss, metrics) per objective pass, as the warm-up objective does) under each
    precision of ``quantization.plan(step)`` and backpropagate ``weight·loss``. Returns {point: [metrics...]} and the
    per-point mean loss. ``refresh`` rebuilds cached parametrized weights after a precision switch."""
    results, losses = {}, {}
    for name, mix, weight in quantization.plan(step):
        with quantization.context(name, mix):
            if refresh is not None:
                refresh()
            values = []
            for loss, metrics in passes(name):
                if not torch.isfinite(loss):
                    raise RuntimeError(f'nonfinite loss at precision {name}')
                (weight * loss).backward()
                if refresh is not None:
                    refresh()
                values.append(float(loss.detach()))
                results.setdefault(name, []).append(metrics)
            losses[name] = sum(values) / max(len(values), 1)
    return results, losses


def gate_columns(quantization, evaluate, passed, step=None):
    """Evaluate each gated point at its deploy precision (λ = 1). ``evaluate()`` returns a report; ``passed(report)``
    applies the stage's thresholds. Returns the per-precision columns (each with its wall time) and the names that
    passed. Trainers call this only where a gate decision is made (owner 2026-10-10: sparse, cheap evaluations). With
    ``step``, a point that is not required in this stage and whose ramp has not started (λ = 0) is skipped: its deploy
    column would measure what the stage is not training yet."""
    import time
    columns = {}
    mixes = quantization.mixes(step) if step is not None else {}
    required = set(quantization.required_points())
    for name in quantization.gated_points():
        if step is not None and name not in required and mixes.get(name, 0.0) <= 0.0:
            columns[name] = {'passed': None, 'skipped': 'ramp not started in this stage', 'report': {}}
            continue
        started = time.perf_counter()
        with quantization.context(name, 1.0):
            report = evaluate()
        seconds = time.perf_counter() - started
        print(json.dumps({'event': 'precision_column', 'point': name, 'step': step, 'seconds': round(seconds, 1)}),
              flush=True)
        columns[name] = {'passed': bool(passed(report)), 'report': report, 'seconds': seconds}
    return columns


def precision_verdict(quantization, columns, bf16_passed):
    """The stage's precision certificate: BF16 plus every required point must pass; names those that passed."""
    passed = (['bf16'] if bf16_passed else []) + [n for n, c in columns.items() if c['passed'] is True]
    required = ['bf16'] + quantization.required_points()
    return {'precisions_passed': passed, 'precisions_required': required,
            'precisions_qualified': all(name in passed for name in required)}


def plain_parameter_name(name):
    """``x.parametrizations.weight.original`` -> ``x.weight`` (the latent under its unparametrized name)."""
    return name.replace('.parametrizations.weight.original', '.weight')


@torch.no_grad()
def layer_sensitivity(model, windows, *, group, fmt, options=None, layer_count, forward=None):
    """Per-layer sensitivity of ``model`` to one precision format on one module group: the mean next-token
    KL(BF16 ‖ quantized) over ``windows`` (lists of token ids) when only that layer's ``group`` modules are quantized
    at λ = 1. Larger = more sensitive. ``model`` must have the group installed (``Quantization.install``)."""
    from ..maple.ternary import active_precision

    forward = forward or (lambda ids: model(ids).logits)
    device = next(model.parameters()).device
    batches = [torch.tensor([w], device=device) for w in windows]
    reference = [torch.log_softmax(forward(ids).float(), -1) for ids in batches]
    scores = {}
    for layer in range(layer_count):
        total, count = 0.0, 0
        options_layer = {**(options or {}), 'layers': {layer}}
        with active_precision({group: (fmt, 1.0, options_layer)}, 'sensitivity'):
            for ids, ref in zip(batches, reference):
                logp = torch.log_softmax(forward(ids).float(), -1)
                total += float((ref.exp() * (ref - logp)).sum(-1).sum())
                count += ids.shape[1]
        scores[str(layer)] = total / max(count, 1)
    return scores


def main(argv=None):
    """``sensitivity``: write a pinned per-layer probe for a ``layers.least_sensitive`` selection.

        python -m natlang_neuralese.train.quantization sensitivity --heads HEADS.pt --text-data T.jsonl \
            --group attention --format ternary --windows 8 --tokens 512 --out probe.json
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    probe = sub.add_parser('sensitivity')
    probe.add_argument('--heads', required=True, help='a port heads checkpoint naming the backbone')
    probe.add_argument('--text-data', required=True, help='JSONL with a "text" field')
    probe.add_argument('--group', choices=GROUPS, required=True)
    probe.add_argument('--format', choices=FORMATS, required=True)
    probe.add_argument('--option-group', type=int, default=None, help='int4 block (default 32)')
    probe.add_argument('--nested-group', type=int, default=None, help='ternary over the int4 grid of this block')
    probe.add_argument('--windows', type=int, default=8)
    probe.add_argument('--tokens', type=int, default=512)
    probe.add_argument('--device', default='cuda')
    probe.add_argument('--out', type=Path, required=True)
    init = sub.add_parser('init-gate', help='decide a recipe init: registered latents at λ = 0 against BF16')
    init.add_argument('--model', required=True, help='the BF16 published model directory')
    init.add_argument('--recipe', type=Path, required=True, help='the recipe declaring init and quantization.gate')
    init.add_argument('--train-log', type=Path, default=None,
                      help="the latents' trainer log (qat_convert train.jsonl) to read λ at their step")
    init.add_argument('--device', default='cuda')
    init.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'init-gate':
        return init_gate(args)
    from ..serve import load_engine

    engine = load_engine(heads_checkpoint=args.heads, device=args.device)
    backbone = engine.backbone
    windows, buffer = [], []
    for line in open(args.text_data):
        buffer += engine.tokenizer(json.loads(line)['text'], add_special_tokens=False).input_ids
        while len(buffer) >= args.tokens and len(windows) < args.windows:
            windows.append(buffer[:args.tokens])
            buffer = buffer[args.tokens:]
        if len(windows) >= args.windows:
            break
    options = {k: v for k, v in (('group', args.option_group), ('nested_group', args.nested_group)) if v}
    spec = {'schema': SCHEMA, 'points': [{'name': 'probe', 'ramp': {'start': 0.0, 'end': 1.0},
                                          'modules': {args.group: {'format': args.format, **options}}}]}
    quantization = Quantization({**spec, 'stage': {'id': 'probe', 'progress': [1.0, 1.0], 'start': 0, 'end': 1}})
    quantization.install(backbone.hf, layer_count=backbone.num_layers)
    scores = layer_sensitivity(backbone.hf, windows, group=args.group, fmt=args.format, options=options,
                               layer_count=backbone.num_layers)
    args.out.write_text(json.dumps({'schema': 'natlang.neuralese-quant-sensitivity/1', 'heads': str(args.heads),
                                    'text_data': str(args.text_data), 'group': args.group, 'format': args.format,
                                    'options': options, 'windows': len(windows), 'tokens': args.tokens,
                                    'metric': 'mean next-token KL(bf16 || one layer quantized)', 'layers': scores},
                                   indent=1) + '\n')
    print(json.dumps({'out': str(args.out), 'most_sensitive': sorted(scores, key=scores.get)[-3:]}))



def init_gate(args):
    """Recipe ``init``: start from registered latents (e.g. conversion v3's best weights) only when, at λ = 0 (the
    latents as BF16 weights), they still pass the BF16 behaviour gate: generation pass rate at most ``max_gate_drop``
    below BF16 Mellum's and held KL to BF16 at most ``max_held_kl`` (the recipe's ``init.gate``, the shared
    ``Behaviour`` gate). Writes a receipt naming the artifact, its step and λ at that step; ``passed`` false means the
    lineage falls back to BF16 (foundation_heads --init-receipt records either choice)."""
    from transformers import AutoTokenizer

    from ..artifacts import resolve
    from ..common.paths import root as path_root
    from ..maple.model import load_maple
    from ..maple.student import load_init_latents
    from .recipe import load_recipe

    recipe = load_recipe(args.recipe)
    declared, spec = recipe.get('init'), recipe.get('quantization') or {}
    if not declared or declared['source'] == 'bf16':
        raise SystemExit('the recipe declares no latents init')
    source = declared['source']
    path, sha = resolve(source['artifact'], source['file'])
    preserve = spec.get('preserve')
    held = teacher_shards(preserve['teacher']['artifact'])[1] if preserve else None
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    gate = Behaviour({**spec['gate'], **declared.get('gate', {})}, tokenizer, held, root=path_root('repo'))
    model = load_maple(args.model, device=args.device, dtype=torch.bfloat16, ternary_attention=False).eval()
    bf16 = gate.report(model)
    step = load_init_latents(model, path)
    latents = gate.report(model)
    mix = None
    if args.train_log is not None and args.train_log.exists():
        rows = [json.loads(line) for line in args.train_log.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if 'mix' in r and r.get('step', -1) <= step]
        mix = max(rows, key=lambda r: r['step'])['mix'] if rows else None
    receipt = {'schema': 'natlang.neuralese-init-gate/1', 'artifact': source['artifact'], 'file': source['file'],
               'sha256': sha, 'step': step, 'mix': mix, 'model': args.model, 'recipe': str(args.recipe),
               'thresholds': {**spec['gate'], **declared.get('gate', {})}, 'bf16': bf16, 'latents': latents,
               'passed': gate.passed(latents, bf16)}
    args.out.write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({'out': str(args.out), 'passed': receipt['passed'], 'step': step, 'mix': mix,
                      'bf16_gate_pass': bf16['gate_pass'], 'latents_gate_pass': latents['gate_pass'],
                      'latents_held_kl': latents.get('held_kl')}))
    return 0


def validate_init(init, spec):
    """A recipe's ``init``: ``{"source": "bf16"}`` or ``{"source": {"artifact", "file"}, "fallback": "bf16",
    "gate": {...}, "ramp_floor": point, "optimizer": "fresh"}``. The ramp floor point keeps at least the latents' λ."""
    if not isinstance(init, dict) or 'source' not in init:
        raise ValueError('init needs a source')
    if init['source'] == 'bf16':
        if set(init) != {'source'}:
            raise ValueError('a bf16 init has no other fields')
        return init
    if set(init) - {'source', 'fallback', 'gate', 'ramp_floor', 'optimizer', 'description'} or \
            not isinstance(init['source'], dict) or set(init['source']) != {'artifact', 'file'}:
        raise ValueError('init source is bf16 or {artifact, file}')
    if init.get('fallback', 'bf16') != 'bf16' or init.get('optimizer', 'fresh') != 'fresh':
        raise ValueError('init falls back to bf16 and starts a fresh optimizer (a new objective is not an optimizer '
                         'handoff)')
    if not spec or 'gate' not in spec:
        raise ValueError('a latents init is gated by the quantization component gate')
    if init.get('ramp_floor') is not None and init['ramp_floor'] not in {p['name'] for p in spec['points']}:
        raise ValueError('init ramp_floor names a declared precision point')
    gate = init.get('gate', {})
    if set(gate) - {'max_gate_drop', 'max_held_kl'} or not all(_number(v) and v >= 0 for v in gate.values()):
        raise ValueError('init gate overrides max_gate_drop and max_held_kl')
    return init


def init_mix_for(recipe, identity):
    """The ``init_mix`` a lineage's stages receive, after checking the heads' recorded init decision against the
    recipe: the declared latents (with a passing receipt) or the BF16 fallback (with a failing one)."""
    init = recipe.get('init')
    recorded = (identity or {}).get('init')
    if init is None or init['source'] == 'bf16':
        if recorded is not None and recorded.get('source') == 'artifact':
            raise ValueError('the heads start from latents the recipe does not declare')
        return {}
    if recorded is None:
        raise ValueError('the recipe declares a latents init; build the heads with foundation_heads --init-receipt')
    receipt = json.loads(Path(recorded['receipt']).read_text())
    if sha256_file_hex(recorded['receipt']) != recorded['receipt_sha256']:
        raise ValueError('init gate receipt changed since the heads recorded it')
    if (receipt['artifact'], receipt['file']) != (init['source']['artifact'], init['source']['file']):
        raise ValueError('init gate receipt belongs to other latents than the recipe declares')
    if recorded['source'] == 'artifact':
        if not receipt['passed'] or recorded['sha256'] != receipt['sha256']:
            raise ValueError('heads start from latents whose init gate did not pass')
        if init.get('ramp_floor') and recorded.get('mix') is not None:
            return {init['ramp_floor']: float(recorded['mix'])}
        return {}
    if receipt['passed']:
        raise ValueError('the init gate passed; the heads must start from the declared latents')
    return {}


if __name__ == '__main__':
    main()
