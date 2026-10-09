"""Verify a prior trained heads checkpoint before exact-state text continuation.

This stage only binds and checks existing artifacts. It neither creates model
weights nor transfers the raw reference certificate to the learned channel.
"""
import argparse
import json
import math
from pathlib import Path

from .output_embedding_projection import sha


def verify_handoff(*, raw_heads, raw_runtime_report, warmup_checkpoint,
                   warmup_manifest, warmup_heads, warmup_runtime_report,
                   max_ce_delta=0.05, max_relative_mse=0.025, min_agreement=0.995):
    raw_heads = Path(raw_heads)
    warmup_checkpoint = Path(warmup_checkpoint)
    warmup_manifest = Path(warmup_manifest)
    warmup_heads = Path(warmup_heads)
    raw_runtime_report = Path(raw_runtime_report)
    warmup_runtime_report = Path(warmup_runtime_report)
    manifest = json.loads(warmup_manifest.read_text())
    text_runtime = json.loads(warmup_runtime_report.read_text())
    raw_runtime = json.loads(raw_runtime_report.read_text())

    if manifest.get('schema') != 'natlang.neuralese-best-warmup-checkpoint/1':
        raise ValueError('expected a shared full-state best warm-up checkpoint manifest')
    files = manifest.get('files', {})
    expected_checkpoint = files.get('best-checkpoint.pt', {}).get('sha256')
    expected_heads = files.get('best-heads.pt', {}).get('sha256')
    if not expected_checkpoint or sha(warmup_checkpoint) != expected_checkpoint:
        raise ValueError('full warm-up checkpoint does not match its exact best-checkpoint manifest')
    if not expected_heads or sha(warmup_heads) != expected_heads:
        raise ValueError('serving heads do not match their exact best-checkpoint manifest')

    warmup = manifest.get('qualification', {})
    strata = warmup.get('strata', {})
    if warmup.get('alignment_gate_passed') is not True or not strata:
        raise ValueError('best warm-up weights did not pass the declared held alignment gate')
    checked_strata = []
    for name, row in sorted(strata.items()):
        ce_delta = row.get('ce_delta')
        mse_delta = row.get('embedding_mse_delta')
        agreement = row.get('text_argmax_agreement')
        initial_delta = row.get('text_ce_delta_from_initial', 0.0)
        if (not isinstance(row.get('tokens'), int) or row['tokens'] < 1 or
                not all(isinstance(v, (int, float)) and math.isfinite(v)
                        for v in (ce_delta, mse_delta, agreement, initial_delta)) or
                ce_delta > max_ce_delta or initial_delta > max_ce_delta or
                mse_delta > max_relative_mse or agreement < min_agreement):
            raise ValueError('best warm-up checkpoint fails a declared held alignment stratum: ' + name)
        checked_strata.append(name)

    if raw_runtime.get('schema') != 'natlang.neuralese-runtime-handoff/1' or raw_runtime.get('runtime_transport_passed') is not True:
        raise ValueError('raw reference runtime transport report did not pass')
    raw_pins = raw_runtime.get('pins', {})
    if raw_pins.get('heads') != sha(raw_heads):
        raise ValueError('raw runtime report is bound to different source heads')

    if (text_runtime.get('schema') != 'natlang.latent-sketch-diagnostic/1' or
            text_runtime.get('profile') != 'latent-sketch-v2' or
            text_runtime.get('parent_sha256') != expected_heads or
            text_runtime.get('runtime_qualified') is not True or
            text_runtime.get('implementation_checks_passed') is not True or
            text_runtime.get('foundation_requalified') is not False):
        raise ValueError('trained latent-sketch runtime report is missing, mismatched, or failed')
    by_length = {row.get('length'): row for row in text_runtime.get('rows', [])}
    if not {128, 512} <= set(by_length):
        raise ValueError('trained latent-sketch runtime report must contain the declared 128 and 512 checks')
    for length in (128, 512):
        row = by_length[length]
        if (row.get('finite_gradients') is not True or
                row.get('sketch_receives_consumer_gradient') is not True):
            raise ValueError('trained latent-sketch runtime or gradient check failed at length ' + str(length))

    return {
        'schema': 'natlang.neuralese-verified-heads-handoff/1',
        'handoff_verified': True,
        'inputs': {
            'raw_heads_sha256': sha(raw_heads),
            'raw_runtime_report_sha256': sha(raw_runtime_report),
            'warmup_checkpoint_sha256': expected_checkpoint,
            'warmup_manifest_sha256': sha(warmup_manifest),
            'warmup_heads_sha256': expected_heads,
            'warmup_runtime_report_sha256': sha(warmup_runtime_report),
        },
        'warmup_step': manifest.get('step'),
        'warmup_alignment_gate_passed': True,
        'warmup_consecutive_passes': warmup.get('consecutive_passes'),
        'warmup_consecutively_qualified': warmup.get('qualified') is True,
        'warmup_checked_strata': checked_strata,
        'thresholds': {'max_ce_delta': max_ce_delta,
                       'max_relative_mse': max_relative_mse,
                       'min_agreement': min_agreement},
        'trained_latent_sketch_runtime_qualified': True,
        'trained_latent_sketch_runtime_lengths': [128, 512],
        'raw_reference_runtime_qualified': True,
        'raw_reference_certificate_inherited_by_learned_channel': False,
        'learned_channel_foundation_qualified': False,
        'weight_action': 'continue_from_exact_full_state_checkpoint',
        'scope': ('verifies exact checkpoint lineage, one held text alignment gate and trained latent-sketch '
                  'runtime/gradient evidence; continuation must earn new consecutive held gates on its declared data'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heads', type=Path, required=True, help='raw-reference heads already runtime-checked')
    parser.add_argument('--raw-runtime-report', type=Path, required=True)
    parser.add_argument('--warmup-checkpoint', type=Path, required=True)
    parser.add_argument('--warmup-manifest', type=Path, required=True)
    parser.add_argument('--warmup-heads', type=Path, required=True)
    parser.add_argument('--warmup-runtime-report', type=Path, required=True)
    parser.add_argument('--max-ce-delta', type=float, default=0.05)
    parser.add_argument('--max-relative-mse', type=float, default=0.025)
    parser.add_argument('--min-agreement', type=float, default=0.995)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError('fresh immutable verified-heads handoff output required')
    report = verify_handoff(raw_heads=args.heads, raw_runtime_report=args.raw_runtime_report,
        warmup_checkpoint=args.warmup_checkpoint, warmup_manifest=args.warmup_manifest,
        warmup_heads=args.warmup_heads, warmup_runtime_report=args.warmup_runtime_report,
        max_ce_delta=args.max_ce_delta, max_relative_mse=args.max_relative_mse,
        min_agreement=args.min_agreement)
    args.out.mkdir(parents=True)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
