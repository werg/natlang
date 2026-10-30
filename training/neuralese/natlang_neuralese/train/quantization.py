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
     "stage_progress": {"joint_sft": [0.0, 1.0]},
     "points": [
       {"name": "int4", "gate": "required" | "report" | "none", "weight": 0.5,
        "ramp": {"start": 0.2, "end": 0.4},
        "modules": {"experts": {"format": "int4", "group": 32}, "mlp": {"format": "int4", "group": 32}}},
       {"name": "ternary-experts", "gate": "report", "weight": 1.0, "ramp": {"start": 0.4, "end": 0.8},
        "modules": {"experts": {"format": "ternary", "nested_group": 32}, "attention": {"format": "int4"}},
        "layers": {"least_sensitive": 0.5, "probe": {"path": "...", "sha256": "..."}}}]}

- Progress: a stage spans ``stage_progress[stage] = [p0, p1]`` of the curriculum; at update ``step`` of an update
  range ``[start, end)`` its progress is ``p0 + (p1 - p0) * (step - start) / (end - start)``. Stages not named run at
  BF16.
- A point's mix ``λ`` ramps linearly from 0 at ``ramp.start`` to 1 at ``ramp.end``; a point with ``λ = 0`` is
  inactive. Its forward value is ``w + λ·(Q(w) − w)`` per selected module group, straight-through to the latent.
- Loss ``sum``: ``bf16_weight·L(bf16) + Σ weight_p·L(p)`` over active points (each its own forward/backward).
  ``sample``: one of {bf16} ∪ active points per update, drawn ∝ its weight from ``Random((seed, step))``, its loss
  scaled by the total weight (an unbiased estimate of the sum at the cost of one pass).
- Gates: a point's ``gate`` (``required``/``report``/``none``) is recorded in ``describe``; ``required_points`` are the
  ``required`` points whose ramp ends within the stage's progress range (``ramp.end ≤ p1``).
- Module groups: ``experts`` (MoE experts), ``attention``, ``conv`` (LFM short convolutions), ``mlp`` (dense FFN),
  ``router``, ``embeddings``, ``heads`` (an untied output projection). A group a backbone lacks selects nothing, so
  one component serves every backbone (LFM: mlp + conv + attention).
- ``layers``: ``"all"`` (default), a list of layer indices, or the least sensitive fraction of layers according to a
  pinned sensitivity probe (JSON ``{"layers": {index: score}}``, larger = more sensitive).
- Formats: ``int4`` (Q4_0, ``group``), ``ternary`` (``nested_group``), and the deployable formats of
  train/qformats.py, exact to Liquid's LFM3 releases (owner 2026-10-28): ``nvfp4`` (ModelOpt NVFP4: E2M1 weights in
  groups of 16 with E4M3 group scales and an FP32 global scale; ``shared_global`` lists module names that deploy as one
  fused GEMM and so share the global scale, ``[["w1", "w3"]]`` for the LFM gate/up pair) and ``fp8_block``
  (compressed-tensors FP8_BLOCK: E4M3 weights, 128x128 block scales, ``block``).
- ``names``: the module names (last name component) of the group a point quantizes, e.g. attention without its
  per-head ``gate_proj``. Module groups stay those of ``classify``.
- ``activations``: input-activation quantization of the selected linear modules, straight-through like the weights
  and ramped by the same λ: ``{"format": "fp8_group", "group": 128}`` (dynamic per-token group scales) or
  ``{"format": "nvfp4", "calibration": {"method": "max", "refresh": R, "batches": N}}`` (NVFP4's static per-tensor
  input scale; the group scales are dynamic). Static ranges are calibrated as ModelOpt does (max |input| over BF16
  forwards) from the current latents: before the first update and every ``R`` updates, the trainer runs
  ``Quantization.calibrate`` over its next ``N`` training batches (``calibration_due``). The ranges are trainer state
  (``state_dict``/``load_state_dict``, checkpointed with the optimizer) and are exported as ``input_scale``. Between
  refreshes they are fixed, so inputs beyond the calibrated range saturate exactly as deployed.
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
FORMATS = ('bf16', 'int4', 'ternary', 'nvfp4', 'fp8_block')
ACTIVATION_FORMATS = ('nvfp4', 'fp8_group')
CALIBRATION_SCHEMA = 'natlang.neuralese-quantization-calibration/1'
GROUPS = ('experts', 'attention', 'conv', 'mlp', 'router', 'embeddings', 'heads')
GATES = ('required', 'report', 'none')


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_spec(spec):
    """Validate a quantization component."""
    if not isinstance(spec, dict) or spec.get('schema') != SCHEMA:
        raise ValueError('quantization must be an object with schema ' + SCHEMA)
    unknown = set(spec) - {'schema', 'loss', 'stage_progress', 'points', 'description'}
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
                    set(config) - {'format', 'group', 'nested_group', 'block', 'names', 'activations', 'shared_global'}:
                raise ValueError(f'module {group} of {name} needs a format in {FORMATS}')
            for key in ('group', 'nested_group', 'block'):
                if key in config and (type(config[key]) is not int or config[key] < 1):
                    raise ValueError(f'{key} must be a positive integer')
            if 'nested_group' in config and config['format'] != 'ternary':
                raise ValueError('nested_group applies to ternary (the coarsest level of the int4 grid)')
            _validate_deploy_options(name, group, config)
        layers = point.get('layers', 'all')
        if not (layers == 'all' or (isinstance(layers, list) and all(type(i) is int and i >= 0 for i in layers))
                or (isinstance(layers, dict) and set(layers) == {'least_sensitive', 'probe'}
                    and _number(layers['least_sensitive']) and 0 < layers['least_sensitive'] <= 1
                    and isinstance(layers['probe'], dict) and set(layers['probe']) == {'path', 'sha256'})):
            raise ValueError('precision point layers is "all", a list of layer indices, or '
                             '{"least_sensitive": fraction, "probe": {"path", "sha256"}}')
    return spec


def _validate_deploy_options(point, group, config):
    """Options of the deployable formats (train/qformats.py): ``block`` (fp8_block), ``shared_global`` (nvfp4: lists
    of module names sharing one global scale, the projections a deploy engine fuses into one GEMM), ``names`` (the
    module names of the group the point quantizes; any format) and ``activations`` (input-activation quantization)."""
    fmt = config['format']
    if 'block' in config and fmt != 'fp8_block':
        raise ValueError('block applies to fp8_block')
    if 'group' in config and fmt == 'fp8_block':
        raise ValueError('fp8_block takes block, not group')
    names = config.get('names')
    if names is not None and (not isinstance(names, list) or not names
                              or not all(isinstance(n, str) and n and '.' not in n for n in names)):
        raise ValueError(f'names of {group} in {point} is a nonempty list of module names (the last name component)')
    shared = config.get('shared_global')
    if shared is not None:
        if fmt != 'nvfp4' or not isinstance(shared, list) or not all(
                isinstance(s, list) and len(s) > 1 and all(isinstance(n, str) and n for n in s) for s in shared):
            raise ValueError('shared_global applies to nvfp4: a list of module-name lists sharing one global scale')
        flat = [n for s in shared for n in s]
        if len(flat) != len(set(flat)):
            raise ValueError('a module name is in one shared_global set at most')
    activations = config.get('activations')
    if activations is None:
        return
    if fmt not in ('nvfp4', 'fp8_block'):
        raise ValueError('activations apply with the deployable weight formats (nvfp4, fp8_block)')
    if (not isinstance(activations, dict) or activations.get('format') not in ACTIVATION_FORMATS
            or set(activations) - {'format', 'group', 'calibration'}
            or ('group' in activations and (type(activations['group']) is not int or activations['group'] < 1))):
        raise ValueError(f'activations of {group} in {point} is a format in {ACTIVATION_FORMATS}, group, calibration')
    calibration = activations.get('calibration')
    if activations['format'] == 'nvfp4':
        if (not isinstance(calibration, dict) or calibration.get('method', 'max') != 'max'
                or set(calibration) - {'method', 'refresh', 'batches'}
                or type(calibration.get('refresh')) is not int or calibration['refresh'] < 1
                or type(calibration.get('batches')) is not int or calibration['batches'] < 1):
            raise ValueError('nvfp4 activations have a static scale: calibration is {method: max, refresh: updates, '
                             'batches: forwards}')
    elif calibration is not None:
        raise ValueError('fp8_group activations are dynamic and take no calibration')


def stage_component(spec, stage, *, start, end, init_mix=None):
    """The component as one stage receives it (``--quantization``): the recipe's spec plus the stage's progress range
    and update range. ``None`` when the stage is not named in ``stage_progress`` (it trains at BF16). ``end`` may be
    ``'derived'``: a stage budgeted in epochs (neuralese_direct) derives its update count from its data and sets it
    before the first update (``progress`` refuses an unbound end)."""
    if not spec or stage not in spec.get('stage_progress', {}):
        return None
    if type(start) is not int or not (end == 'derived' or type(end) is int and end > start):
        raise ValueError('a quantized stage needs an update range start < end (or end "derived")')
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
        self.inputs = {}  # module name -> InputPrecision (activation quantizer)
        self.calibration = {'step': None, 'batches': 0, 'source': None}

    # Schedule --------------------------------------------------------------------------------------------------------
    def progress(self, step):
        p0, p1 = self.stage['progress']
        start, end = self.stage['start'], self.stage['end']
        if type(end) is not int:
            raise ValueError('the stage update range end is not bound yet (a derived end is set by the trainer)')
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
        """Context manager running forwards at point ``name`` with mix ``mix`` (``bf16``: unquantized). Inline
        modules (the native trainer) get their pass operands resolved once on entry (``maple.ternary.
        inline_precision``): the ramped weights of the current latents and a device λ."""
        from ..maple.ternary import active_precision, inline_precision
        context = active_precision(self.resolve(name, mix), None if name == 'bf16' else name)
        modules = getattr(self, 'inline_modules', None)
        return inline_precision(context, modules) if modules else context

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
                'required': self.required_points(), 'gated': self.gated_points(),
                **({'calibration': {**self.calibration, 'modules': len(self.input_amax())}}
                   if self.calibration_config() else {})}

    # Installation ----------------------------------------------------------------------------------------------------
    def install(self, module, layer_count=None, *, inline=False):
        """Route every selected module's weight through the active precision point: a ``PrecisionSTE``
        parametrization on linear/embedding weights, ``precision_group`` on dense MoE experts. Weights stay the same
        trainable parameters (latents); with no active point they are untouched.

        ``inline`` (the native trainer): selected ``nn.Linear`` modules keep their plain ``weight`` Parameter (names,
        optimizer routing, checkpoints and DiLoCo surfaces unchanged); they carry their group key and peers, and every
        linear execution path (``maple.ternary.linear_precision``: the module's own forward, the native cotangent
        carriers and the shared-state direct projections) applies the active point. Only linear modules are
        supported inline."""
        from ..maple.model import DenseExperts
        from ..maple.ternary import PrecisionSTE
        if inline:
            return self._install_inline(module, layer_count)

        wanted = {}
        for name, point in self.points.items():
            for group in point['modules']:
                wanted.setdefault(group, set()).add(name)
        if layer_count is None:
            layer_count = 1 + max((layer_of(n) or 0) for n, _ in module.named_modules())
        for name, point in self.points.items():
            self.layers[name] = selected_layers(point, layer_count, root=self.root)
        configs = {}  # group -> the points' module configs for it
        for point in self.points.values():
            for group, config in point['modules'].items():
                configs.setdefault(group, []).append(config)
        seen, parametrized = set(), {}
        for name, child in module.named_modules():
            group = classify(name, child)
            if group not in wanted:
                continue
            layer = layer_of(name)
            # Keyed ``group@layer[:leaf]``: a point's layer and name selections (``options["layers"]``,
            # ``options["names"]``) decide per pass.
            if isinstance(child, DenseExperts):
                if not isinstance(child.gate_up, nn.Parameter):
                    child.make_latent(quantize=False)
                child.precision_group = self._group_key(group, layer)
            else:
                leaf = name.rsplit('.', 1)[-1]
                if all(c.get('names') is not None and leaf not in c['names'] for c in configs[group]):
                    continue  # no point quantizes this module
                weight = child.weight
                if id(weight) in seen:
                    continue  # tied weights are parametrized once
                seen.add(id(weight))
                if parametrize.is_parametrized(child, 'weight'):
                    raise ValueError('module already parametrized; precision points need plain latents: ' + name)
                key = self._group_key(group, layer, leaf)
                ste = PrecisionSTE(key)
                parametrize.register_parametrization(child, 'weight', ste, unsafe=True)
                parametrized[name] = (group, leaf, child, ste)
                if isinstance(child, nn.Linear) and any(c.get('activations') for c in configs[group]):
                    self._install_input(name, key, child, configs[group])
            self.installed[group] = self.installed.get(group, 0) + 1
        # Projections sharing one NVFP4 global scale (one fused deploy GEMM): each reads its siblings' latents.
        for name, (group, leaf, child, ste) in parametrized.items():
            parent = name.rsplit('.', 1)[0]
            shared = {n for c in configs[group] for s in c.get('shared_global', []) if leaf in s for n in s} - {leaf}
            ste.peers = [(n, parametrized[f'{parent}.{n}'][2]) for n in sorted(shared) if f'{parent}.{n}' in parametrized]
        return self.installed

    def _install_inline(self, module, layer_count=None):
        import types
        from ..maple.ternary import InputPrecision, precision_linear_forward

        wanted = {group for point in self.points.values() for group in point['modules']}
        if layer_count is None:
            layer_count = 1 + max((layer_of(n) or 0) for n, _ in module.named_modules())
        for name, point in self.points.items():
            self.layers[name] = selected_layers(point, layer_count, root=self.root)
        configs = {}
        for point in self.points.values():
            for group, config in point['modules'].items():
                configs.setdefault(group, []).append(config)
        selected, seen = {}, set()
        for name, child in module.named_modules():
            group = classify(name, child)
            if group not in wanted:
                continue
            if type(child) is not nn.Linear:
                raise ValueError(f'inline precision supports plain nn.Linear modules only: {name} is '
                                 f'{type(child).__qualname__}')
            leaf = name.rsplit('.', 1)[-1]
            if all(c.get('names') is not None and leaf not in c['names'] for c in configs[group]):
                continue
            if id(child.weight) in seen:
                continue
            if 'forward' in child.__dict__ or parametrize.is_parametrized(child, 'weight'):
                raise ValueError('inline precision needs a plain Linear (no instance forward or parametrization): '
                                 + name)
            seen.add(id(child.weight))
            key = self._group_key(group, layer_of(name), leaf)
            child.native_precision_key = key
            child.native_precision_peers = []
            child.native_precision_input = None
            if any(c.get('activations') for c in configs[group]):
                holder = InputPrecision(key)
                holder.static = any((c.get('activations') or {}).get('format') == 'nvfp4' for c in configs[group])
                child.native_precision_input = holder
                self.inputs[name] = holder
            child.forward = types.MethodType(precision_linear_forward, child)
            selected[name] = (group, leaf, child)
            self.installed[group] = self.installed.get(group, 0) + 1
        for name, (group, leaf, child) in selected.items():
            parent = name.rsplit('.', 1)[0]
            shared = {n for c in configs[group] for s in c.get('shared_global', []) if leaf in s for n in s} - {leaf}
            child.native_precision_peers = [(n, selected[f'{parent}.{n}'][2]) for n in sorted(shared)
                                            if f'{parent}.{n}' in selected]
        self.inline_modules = [child for _, _, child in selected.values()]
        return self.installed

    def _install_input(self, name, key, child, configs):
        from ..maple.ternary import InputPrecision
        holder = InputPrecision(key)
        holder.static = any((c.get('activations') or {}).get('format') == 'nvfp4' for c in configs)
        child.register_forward_pre_hook(holder)
        self.inputs[name] = holder

    @staticmethod
    def _group_key(group, layer, leaf=None):
        key = group if layer is None else f'{group}@{layer}'
        return key if leaf is None else f'{key}:{leaf}'

    # Static activation calibration -----------------------------------------------------------------------------------
    def calibration_config(self):
        """The calibration of the static (NVFP4) activation scales: {refresh, batches} (the most frequent refresh and
        the most batches any point declares), or None when no point has static activation scales."""
        found = [c['activations']['calibration'] for p in self.points.values() for c in p['modules'].values()
                 if (c.get('activations') or {}).get('format') == 'nvfp4']
        if not found:
            return None
        return {'method': 'max', 'refresh': min(c['refresh'] for c in found), 'batches': max(c['batches'] for c in found)}

    def calibration_due(self, step):
        """Whether the trainer must (re)calibrate before update ``step``: never calibrated, or ``refresh`` updates
        since the last calibration."""
        config = self.calibration_config()
        if config is None:
            return False
        last = self.calibration.get('step')
        return last is None or step - last >= config['refresh']

    def begin_calibration(self, step=None):
        """Open a calibration window at update ``step`` (BF16 forwards follow, then ``commit_calibration``). The
        window (start, batches observed, running maxima) is trainer state: ``state_dict`` saves it, so a resume
        inside the window continues it exactly instead of restarting."""
        holders = [h for h in self.inputs.values() if h.static]
        for holder in holders:
            holder.observe, holder.observed = False, None
        self._observed_batches = 0
        self._calibration_window = {'start': step}
        return bool(holders)

    @property
    def calibration_window(self):
        return getattr(self, '_calibration_window', None)

    @torch.no_grad()
    def observe_calibration(self, forward, batch):
        """One BF16 calibration forward of ``batch`` (an incremental ``calibrate``). Only this forward observes:
        the window's training passes (quantized or not) never enter the ranges."""
        holders = [h for h in self.inputs.values() if h.static]
        for holder in holders:
            holder.observe = True
        try:
            with self.context('bf16', 0.0):
                forward(batch)
        finally:
            for holder in holders:
                holder.observe = False
        self._observed_batches = getattr(self, '_observed_batches', 0) + 1

    def commit_calibration(self, step, source=None):
        """Set each static input's ``amax`` from the observed forwards since ``begin_calibration``."""
        holders = {n: h for n, h in self.inputs.items() if h.static}
        for holder in holders.values():
            holder.observe = False
        count = getattr(self, '_observed_batches', 0)
        missing = [n for n, h in holders.items() if h.observed is None]
        if count == 0 or missing:
            raise ValueError('calibration forwards reached no input of: ' + ', '.join(missing or sorted(holders)))
        for holder in holders.values():
            holder.amax, holder.observed = holder.observed.detach().float(), None
        self.calibration = {'step': int(step), 'batches': count, 'source': source}
        self._observed_batches = 0
        self._calibration_window = None
        return {n: float(h.amax) for n, h in holders.items()}

    def cancel_calibration(self):
        for holder in self.inputs.values():
            holder.observe, holder.observed = False, None
        self._observed_batches = 0
        self._calibration_window = None

    @torch.no_grad()
    def calibrate(self, forward, batches, step, source=None):
        """ModelOpt ``max`` calibration of the static activation scales from the current latents: run ``forward(batch)``
        at BF16 (unquantized, as ModelOpt calibrates) over ``batches`` (the trainer's next ``calibration.batches``
        training batches) and set each module's ``amax`` to the max |input| seen. ``source`` records which batches
        (e.g. the data cursor). The result is part of the trainer state (``state_dict``)."""
        holders = {n: h for n, h in self.inputs.items() if h.static}
        if not holders:
            return {}
        for holder in holders.values():
            holder.observe, holder.observed = True, None
        count = 0
        try:
            with self.context('bf16', 0.0):
                for batch in batches:
                    forward(batch)
                    count += 1
        finally:
            for holder in holders.values():
                holder.observe = False
        missing = [n for n, h in holders.items() if h.observed is None]
        if count == 0 or missing:
            raise ValueError('calibration forwards reached no input of: ' + ', '.join(missing or sorted(holders)))
        for holder in holders.values():
            holder.amax, holder.observed = holder.observed.detach().float(), None
        self.calibration = {'step': int(step), 'batches': count, 'source': source}
        return {n: float(h.amax) for n, h in holders.items()}

    def input_amax(self):
        """{module name: calibrated input amax} of the static activation scales (export: ``input_scale``)."""
        return {n: float(h.amax) for n, h in self.inputs.items() if h.static and h.amax is not None}

    def check_calibrated(self, name):
        """Raise when point ``name`` quantizes activations with a static scale that has not been calibrated."""
        point = self.points[name]
        static = any((c.get('activations') or {}).get('format') == 'nvfp4' for c in point['modules'].values())
        if static and any(h.static and h.amax is None for h in self.inputs.values()):
            raise ValueError(f'precision point {name} has static activation scales: calibrate before evaluating it')

    def state_dict(self):
        """The component's trainer state: the calibrated static activation ranges (FP32 values, exact in JSON) and an
        open calibration window (its start, observed batches and running maxima)."""
        state = {'schema': CALIBRATION_SCHEMA, **self.calibration, 'amax': self.input_amax()}
        window = self.calibration_window
        if window is not None:
            state['window'] = {'start': window.get('start'), 'batches': int(getattr(self, '_observed_batches', 0)),
                               'observed': {n: float(h.observed) for n, h in self.inputs.items()
                                            if h.static and h.observed is not None}}
        return state

    def load_state_dict(self, state):
        if state.get('schema') != CALIBRATION_SCHEMA:
            raise ValueError('not a quantization calibration state')
        static = {n for n, h in self.inputs.items() if h.static}
        if set(state['amax']) - static or (state['amax'] and set(state['amax']) != static):
            raise ValueError('calibration state names other modules than the installed static activation inputs')
        for name, value in state['amax'].items():
            self.inputs[name].amax = torch.tensor(value, dtype=torch.float32)
        self.calibration = {k: state.get(k) for k in ('step', 'batches', 'source')}
        self.cancel_calibration()
        window = state.get('window')
        if window is not None:
            if set(window['observed']) - static:
                raise ValueError('calibration window names other modules than the installed static inputs')
            self._calibration_window = {'start': window['start']}
            self._observed_batches = int(window['batches'])
            for name, value in window['observed'].items():
                self.inputs[name].observed = torch.tensor(value, dtype=torch.float32)



def parse_native_component(value):
    """The native trainer's ``--quantization`` (recipe stage parameter ``quantization``): None (absent), ``'off'``
    (drop a carried calibration explicitly), or (recipe spec, bind). The value is JSON text, a path, a dict, or a
    checkpoint's recorded ``natlang.native-inline-qat/1`` declaration; ``bind`` = {stage, start, end} names the stage
    of ``stage_progress`` and the global update range its progress runs over. One precision pass per update, so the
    loss mode must be ``sample``."""
    if value in (None, ''):
        return None
    if value == 'off':
        return 'off'
    if isinstance(value, dict):
        spec = json.loads(json.dumps(value))
    else:
        text = value if value.lstrip().startswith('{') else Path(value).read_text()
        spec = json.loads(text)
    if spec.get('schema') == 'natlang.native-inline-qat/1':  # a checkpoint's recorded declaration (anneal branches)
        spec = {**spec['spec'], 'bind': spec['bind']}
    bind = spec.pop('bind', None)
    if (not isinstance(bind, dict) or set(bind) != {'stage', 'start', 'end'} or type(bind['start']) is not int
            or type(bind['end']) is not int or not 0 <= bind['start'] < bind['end']):
        raise ValueError('--quantization needs "bind": {"stage": ID, "start": U0, "end": U1} with 0 <= U0 < U1')
    if (spec.get('loss') or {}).get('mode', 'sum') != 'sample':
        raise ValueError('the native trainer takes one precision pass per update: declare loss mode "sample"')
    validate_spec(spec)
    if bind['stage'] not in spec.get('stage_progress', {}):
        raise ValueError('--quantization bind.stage is not named in stage_progress: ' + str(bind['stage']))
    return spec, bind


def plain_parameter_name(name):
    """``x.parametrizations.weight.original`` -> ``x.weight`` (the latent under its unparametrized name)."""
    return name.replace('.parametrizations.weight.original', '.weight')
