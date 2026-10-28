#!/usr/bin/env python3
"""Verify and atomically publish the fresh full-state SFT seed and its immutable receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.native_state_sft import TRAINING_SCHEMA
from natlang_neuralese.common.jsonio import write_json_durable


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def publish(out, parent, stop, args_path=None):
    out, parent = Path(out), Path(parent)
    checkpoint = out / 'train/checkpoint.pt'
    progress = json.loads((out / 'train/progress.json').read_text())
    state = CheckpointPolicy.load(checkpoint)
    training = state.get('training') or {}
    step = training.get('global_step')
    if (state.get('schema') != CHECKPOINT_SCHEMA or training.get('schema') != TRAINING_SCHEMA
            or type(step) is not int or step < stop or state.get('step') != step
            or (state.get('model_config') or {}).get('source_architecture') != 'lfm3'
            or not training.get('optimizer') or not training.get('rng')
            or not training.get('source_receipts') or not training.get('exposure_counts')
            or not any(value > 0 for value in training['exposure_counts'].values())):
        raise ValueError('seed checkpoint lacks actual full optimizer/RNG/exposure or required updates')
    size, source_sha = checkpoint.stat().st_size, digest(checkpoint)
    final_save = None
    if int(progress.get('checkpoint_step') or 0) < stop:
        report_path = out / 'train/report.json'
        if not report_path.is_file():
            raise ValueError('seed progress is stale and no authenticated final report exists')
        report = json.loads(report_path.read_text())
        if (report.get('schema') != 'natlang.neuralese-native-sft-report/1'
                or report.get('checkpoint_sha256') != source_sha
                or report.get('step') != training.get('step', step)
                or Path(report.get('checkpoint', '')).resolve() != checkpoint.resolve()):
            raise ValueError('seed final report does not bind the actual checkpoint and step')
        final_save = {'report': str(report_path), 'sha256': digest(report_path),
                      'step': report['step'], 'checkpoint_sha256': source_sha,
                      'cadence_progress_stale': True}
    receipt_path = parent.with_name(parent.name + '.seed.json')
    if receipt_path.exists():
        raise ValueError('seed receipt already exists')
    parent.parent.mkdir(parents=True, exist_ok=True)
    os.link(checkpoint, parent)  # atomic, no replacement; seed trainer has exited and cannot mutate this inode
    if parent.stat().st_size != size or digest(parent) != source_sha:
        parent.unlink(missing_ok=True)
        raise ValueError('published full seed failed independent size/hash readback')
    receipt = {'schema': 'natlang.lfm3-fresh-sft-seed/1', 'checkpoint': str(parent), 'sha256': source_sha,
               'bytes': size, 'global_step': step, 'requested_stop': stop,
               'checkpoint_step_observed': progress['checkpoint_step'],
               'training_source_receipts': training['source_receipts'],
               'exposure_visits': sum(training['exposure_counts'].values()),
               'full_optimizer_rng_exposure': True, 'seed_output': str(out),
               'trainer_arguments': ({'path': str(Path(args_path).resolve()), 'sha256': digest(Path(args_path))}
                                     if args_path is not None else None)}
    if final_save is not None: receipt['final_save_evidence'] = final_save
    write_json_durable(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out'); parser.add_argument('parent'); parser.add_argument('stop', type=int); parser.add_argument('args_path')
    args = parser.parse_args()
    publish(args.out, args.parent, args.stop, args.args_path)


if __name__ == '__main__':
    main()
