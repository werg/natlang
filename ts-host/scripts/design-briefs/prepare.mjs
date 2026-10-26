#!/usr/bin/env node
/** Prepare existing teacher/RL entrypoints. This command never starts collection or training. */
import { spawnSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { REPO, json, verifyPacket } from './lib.mjs';

const args = process.argv.slice(2);
if (args.length !== 4 || args[0] !== '--packet' || args[2] !== '--out')
  throw new Error('usage: node prepare.mjs --packet FROZEN_DIRECTORY --out NEW_DIRECTORY');
const packet = resolve(args[1]), out = resolve(args[3]);
const manifest = await verifyPacket(packet);
const gate = manifest.files.find(item => item.path === 'train.gates.ir.jsonl' && item.split === 'train' && item.oracle === 'exact');
if (!gate?.rows) throw new Error('packet has no exact TRAIN gate tasks');
const workflows = manifest.files.find(item => item.path === 'train.workflows.ir.jsonl' && item.split === 'train' &&
  item.oracle === 'independent-review-required');
if (!workflows?.rows) throw new Error('packet has no review-required TRAIN workflows');
await mkdir(out, { recursive: false });
const env = { ...process.env, PYTHONPATH: [join(REPO, 'training/neuralese'), REPO,
  process.env.PYTHONPATH].filter(Boolean).join(':') };
const python = process.env.NATLANG_PYTHON ?? 'python3';
const argv = ['-m', 'natlang_neuralese.rl.task_rollouts', 'prepare', '--cases', join(packet, gate.path),
  '--out', join(out, 'rl'), '--rows', String(gate.rows), '--group', '4', '--root-mode', 'student'];
const prepared = spawnSync(python, argv, { cwd: REPO, env, encoding: 'utf8' });
if (prepared.status !== 0) throw new Error(`existing RL preparer failed: ${prepared.stderr || prepared.stdout}`);
// Read the preparer's actual artifacts, so completion does not depend on child stdout capture.
const rlPreparation = JSON.parse(await readFile(join(out, 'rl/preparation.json'), 'utf8'));
const queue = (await readFile(join(out, 'rl/queue.jsonl'), 'utf8')).trim().split('\n').map(line => JSON.parse(line));
if (rlPreparation.root_mode !== 'student' || rlPreparation.source_sha256 !== gate.sha256 || queue.length !== gate.rows * 4)
  throw new Error('existing RL preparer output does not match the frozen student-root request');
const teacherJobs = ['gates', 'workflows'].map(kind => ({
  name: `design-briefs-train-${kind}`, source: join(packet, `train.${kind}.ir.jsonl`),
  rows: kind === 'gates' ? gate.rows : workflows.rows,
  command: ['node', 'ts-host/scripts/teacher-collector.mjs', join(packet, `train.${kind}.ir.jsonl`),
    join(out, `teacher-${kind}/jobs`), join(out, `teacher-${kind}/results.jsonl`),
    '--model-id', '${TEACHER_MODEL_ID}', '--server', '${TEACHER_SERVER}',
    '--root-seed', '20261025', '--all', '--workers', '1', '--collection-role', 'teacher',
    ...(kind === 'workflows' ? ['--judge-model-id', '${JUDGE_MODEL_ID}', '--judge-server', '${JUDGE_SERVER}'] : [])],
  execution: 'prepared-only', actor: 'self-hosted-teacher',
  required_bindings: kind === 'workflows' ? ['TEACHER_MODEL_ID', 'TEACHER_SERVER', 'JUDGE_MODEL_ID', 'JUDGE_SERVER'] :
    ['TEACHER_MODEL_ID', 'TEACHER_SERVER'],
}));
await writeFile(join(out, 'teacher-jobs.json'), json({ schema: 'natlang.design-brief-teacher-jobs/1', jobs: teacherJobs,
  note: 'Resolve endpoint/model bindings against the existing self-hosted generation service; source preparation does not resume paused services.' }));
await writeFile(join(out, 'preparation.json'), json({ schema: 'natlang.design-brief-preparation/1',
  packet, release: manifest.release, rl: { ...rlPreparation, rows: gate.rows, group: 4, queue_entries: queue.length },
  teacher_jobs: teacherJobs.length,
  workflow_rl: { source: join(packet, workflows.path), rows: workflows.rows,
    status: 'independent-judge-and-live-qualification-required-before-RL-admission' },
  training_admission: false, executed_models: 0, live_qualification: 'pending',
  whole_graph: 'Preserve actual sampled/forced producer turns, consumer imports and occurrence bindings; reward task success with or without Neuralese.' }));
console.log(JSON.stringify({ output: out, exact_train_tasks: gate.rows, rl_samples: gate.rows * 4,
  teacher_jobs: teacherJobs.length, executed_models: 0 }));
