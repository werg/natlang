"""The one warm-up-to-serving export transform (ARCHITECTURE_IMPROVEMENT C1).

A text warm-up keeps its authoritative state in a full-state ``checkpoint.pt`` (student parameters, heads including the
training-only input map, optimizer, RNG, schedule). What serving, diagnostics and the next lineage stage load is a
serving export ``heads.pt``. Both the trainer's own periodic export and any later diagnostic build that export here, so
the tensor conversion exists once:

* ``build_heads_export``: from the live model inside a running trainer;
* ``export_from_checkpoint``: from the exact weights of a saved full-state checkpoint, with no model loaded.

Every export names its source (``warmup.source``): the checkpoint it came from (sha256 when it came from a file), the
step of those weights and the step the checkpoint reached, so a heads file that trails its checkpoint is never taken for
current. ``heads_current`` is true only when both steps are equal.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..common.hashing import sha256_file_hex as sha

CHECKPOINT_SCHEMA = 'natlang.neuralese-text-warmup/1'
STATUS_SCHEMA = 'natlang.neuralese-text-warmup-heads-export/1'
FOUNDATION_UNQUALIFIED = {'qualified': False, 'runtime_qualified': False, 'requires_requalification': True}


def export_source(*, checkpoint_step, heads_step, checkpoint_sha256=None, checkpoint_path=None):
    """Where an export's weights came from and whether they are the checkpoint's current weights."""
    heads_step, checkpoint_step = int(heads_step), int(checkpoint_step)
    return {'checkpoint_path': None if checkpoint_path is None else str(checkpoint_path),
            'checkpoint_sha256': checkpoint_sha256,
            'checkpoint_step': checkpoint_step, 'heads_step': heads_step,
            'heads_step_known': heads_step >= 0, 'heads_current': heads_step == checkpoint_step}


def heads_export_status(*, checkpoint_step, heads_step, export_error=None, emergency=False):
    """The record beside a run (``heads-export-status.json``): does heads.pt trail checkpoint.pt?"""
    source = export_source(checkpoint_step=checkpoint_step, heads_step=heads_step)
    status = {'schema': STATUS_SCHEMA, 'checkpoint_step': source['checkpoint_step'], 'heads_step': source['heads_step'],
              'heads_step_known': source['heads_step_known'], 'heads_current': source['heads_current'],
              'checkpoint_authoritative_for_resume': True, 'emergency_export_attempted': bool(emergency)}
    if export_error is not None:
        status['export_error'] = {'type': type(export_error).__name__, 'message': str(export_error)[:1000]}
    return status


def write_heads_export_status(out, status):
    """Atomically replace ``<out>/heads-export-status.json``; False when the disk refused (never raises)."""
    out = Path(out)
    pending = out / 'heads-export-status.json.pending'
    try:
        pending.write_text(json.dumps(status, indent=2) + '\n')
        pending.replace(out / 'heads-export-status.json')
        return True
    except OSError:
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def assemble_heads_export(*, initial, heads_state, port_config, control_rows, backbone_trainables, lora, lora_layers,
                          backbone_training, rank, identity, step, report, report_path, report_sha256, source):
    """The conversion itself, over plain dictionaries: split the training-only input map from the serving heads,
    carry the explicit backbone deltas, and never inherit certification from the parent."""
    serving = dict(heads_state)
    trained_map = {k.removeprefix('input_map.'): v for k, v in serving.items() if k.startswith('input_map.')}
    serving = {k: v for k, v in serving.items() if not k.startswith('input_map.')}
    exported = {**initial, 'heads': serving, **({'neuralese_input_map': trained_map} if trained_map else {}),
                'control_rows': control_rows, 'port_config': port_config,
                'backbone_trainables': backbone_trainables,
                'backbone_training': 'lora' if backbone_training == 'adapters' else backbone_training,
                'foundation': dict(FOUNDATION_UNQUALIFIED),
                **({'maple_qat': True} if backbone_training == 'qat' else {}),
                'lora': lora, 'lora_layers': lora_layers, 'lora_rank': rank,
                'warmup': {'step': int(step), 'identity': identity,
                           'alignment_qualified': bool(report and report.get('qualified')),
                           'report_path': report_path, 'report_sha256': report_sha256, 'source': source}}
    # Parent adapters may have a different rank than the fresh-policy default.
    ranks = {v.shape[0] for n, v in exported['lora'].items() if '.lora_A.' in n}
    if len(ranks) == 1:
        exported['lora_rank'] = next(iter(ranks))
    return exported


def _plain_named(module):
    from .backbone_policy import plain_named_tensors
    return plain_named_tensors(module)


def build_heads_export(*, initial_heads_path, heads, backbone, backbone_names, backbone_training, rank, identity,
                       out, report=None, export_step, snapshot=False):
    """The export of the live model inside a running trainer. ``snapshot`` keeps tensors on their device so a
    detached copy can be taken at the update boundary and written later."""
    from .adapters import adapter_layers, lora_state
    from .trajectory_state import immutable_cpu_snapshot
    initial = torch.load(initial_heads_path, map_location='cpu', weights_only=False, mmap=True)
    control_rows = backbone.control_rows.detach()
    if not snapshot:
        control_rows = control_rows.cpu()
    out = Path(out)
    exported = assemble_heads_export(
        initial=initial, heads_state=heads.state_dict(),
        port_config={'cutoff': heads.cutoff, 'max_length': heads.max_length, **heads.port_config()},
        control_rows=control_rows,
        backbone_trainables={n: (q.detach() if snapshot else q.detach().cpu())
                             for n, q in _plain_named(backbone.hf).items() if n in backbone_names},
        lora=lora_state(backbone), lora_layers=adapter_layers(backbone), backbone_training=backbone_training,
        rank=rank, identity=identity, step=export_step, report=report,
        report_path=str((out / 'report.json').resolve()),
        report_sha256=sha(out / 'report.json') if (out / 'report.json').is_file() else None,
        source=export_source(checkpoint_step=export_step, heads_step=export_step))
    return immutable_cpu_snapshot(exported) if snapshot else exported


def load_exact_checkpoint(path):
    """A warm-up full-state checkpoint exactly as saved, with its file digest. Refuses any other file."""
    path = Path(path)
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if state.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError(f'{path} is not a text warm-up full-state checkpoint (schema {state.get("schema")!r})')
    for key in ('step', 'heads', 'student_parameters', 'identity'):
        if key not in state:
            raise ValueError(f'{path} lacks {key!r}; it is not a complete warm-up checkpoint')
    return state, sha(path)


def export_from_checkpoint(checkpoint_path, template_export, *, heads_step=None, report=None):
    """The serving export of a saved checkpoint's exact weights, with no model loaded.

    ``template_export`` is a serving export of the same run (its ``heads.pt``, ``best-heads.pt`` or an earlier export):
    it supplies what the checkpoint does not hold (port configuration, control rows, base metadata). The head tensors
    of the template and the checkpoint must agree in names and shapes, so a template from another architecture fails
    here and not at serve time. Only non-quantized adapter policies can be rebuilt from parameter names; a ternary
    (Maple QAT) checkpoint is refused rather than exported approximately.

    ``heads_step`` is the step of the weights the caller means to export; it defaults to the checkpoint's own. A
    smaller value marks the result as lagging (``warmup.source.heads_current`` false).
    """
    state, checkpoint_sha = load_exact_checkpoint(checkpoint_path)
    template = torch.load(template_export, map_location='cpu', weights_only=False, mmap=True) \
        if not isinstance(template_export, dict) else template_export
    if 'warmup' not in template or 'port_config' not in template or 'control_rows' not in template:
        raise ValueError('the template must be a serving export written by a text warm-up (needs warmup, port_config, '
                         'control_rows)')
    options = state['identity'].get('options', {})
    if options.get('backbone_training') == 'qat' or template.get('maple_qat'):
        raise ValueError('a ternary (QAT) checkpoint cannot be exported from parameter names; use the live trainer')
    checkpoint_heads = {k: v for k, v in state['heads'].items() if not k.startswith('input_map.')}
    template_heads = template['heads']
    if checkpoint_heads.keys() != template_heads.keys():
        raise ValueError('checkpoint and template serving heads differ in tensor names: '
                         f'{sorted(checkpoint_heads.keys() ^ template_heads.keys())[:6]}')
    for name, tensor in checkpoint_heads.items():
        if tuple(tensor.shape) != tuple(template_heads[name].shape):
            raise ValueError(f'checkpoint and template differ in shape for heads.{name}')
    trainables = {n.removeprefix('backbone.'): v for n, v in state['student_parameters'].items()
                  if n.startswith('backbone.')}
    lora = {n: v.detach().cpu() for n, v in trainables.items() if 'lora_' in n}
    layers = sorted({int(n.split('model.layers.')[1].split('.')[0]) for n in lora})
    report_path = template['warmup'].get('report_path')
    initial = {k: v for k, v in template.items()
               if k not in {'heads', 'neuralese_input_map', 'control_rows', 'port_config', 'backbone_trainables',
                            'backbone_training', 'foundation', 'maple_qat', 'lora', 'lora_layers', 'lora_rank', 'warmup'}}
    step = int(state['step'])
    return assemble_heads_export(
        initial=initial, heads_state={k: v.detach().cpu() for k, v in state['heads'].items()},
        port_config=dict(template['port_config']), control_rows=template['control_rows'],
        backbone_trainables={n: v.detach().cpu() for n, v in trainables.items()},
        lora=lora, lora_layers=layers, backbone_training=template.get('backbone_training', 'full'),
        rank=template.get('lora_rank', options.get('rank', 0)), identity=state['identity'], step=step,
        report=report if report is not None else state.get('qualification'), report_path=report_path,
        report_sha256=template['warmup'].get('report_sha256'),
        source=export_source(checkpoint_step=step, heads_step=step if heads_step is None else heads_step,
                             checkpoint_sha256=checkpoint_sha, checkpoint_path=Path(checkpoint_path).resolve()))
