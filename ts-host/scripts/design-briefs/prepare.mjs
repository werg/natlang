#!/usr/bin/env node
/** Prepare existing teacher/RL entrypoints. This command never starts collection or training. */
import { spawnSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { REPO, json, verifyPacket } from './lib.mjs';

const args = process.argv.slice(2);
const allowed = new Set(['--packet', '--out', '--teacher-model-id', '--teacher-provider', '--teacher-server',
  '--judge-model-id', '--judge-provider', '--judge-server']);
const flags = new Map();
for (let i = 0; i < args.length; i += 2) {
  if (!allowed.has(args[i]) || flags.has(args[i]) || !args[i + 1] || args[i + 1].startsWith('--'))
    throw new Error('invalid or duplicate preparation argument');
  flags.set(args[i], args[i + 1]);
}
if (!flags.has('--packet') || !flags.has('--out'))
  throw new Error('usage: node prepare.mjs --packet FROZEN_DIRECTORY --out NEW_DIRECTORY [explicit teacher/judge bindings]');
function actor(role) {
  const model = flags.get(`--${role}-model-id`), provider = flags.get(`--${role}-provider`), server = flags.get(`--${role}-server`);
  if (model || provider || server) {
    if (!model || Number(!!provider) + Number(!!server) !== 1)
      throw new Error(`${role} requires its actual model ID and exactly one provider or server`);
    return { model_id: model, ...(provider ? { provider } : { server }) };
  }
  return null;
}
const teacher = actor('teacher'), judge = actor('judge');
if (judge && !teacher) throw new Error('bind the teacher before selecting a judge');
if (teacher && judge && teacher.model_id === judge.model_id) throw new Error('teacher and judge must be distinct actual models');
const packet = resolve(flags.get('--packet')), out = resolve(flags.get('--out'));
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
const teacherArgs = teacher ? ['--model-id', teacher.model_id,
  ...(teacher.provider ? ['--provider', teacher.provider] : ['--server', teacher.server])] :
  ['--model-id', '${TEACHER_MODEL_ID}', '--server', '${TEACHER_SERVER}'];
const judgeArgs = judge ? ['--judge-model-id', judge.model_id,
  ...(judge.provider ? ['--judge-provider', judge.provider] : ['--judge-server', judge.server])] :
  ['--judge-model-id', '${JUDGE_MODEL_ID}', '--judge-server', '${JUDGE_SERVER}'];
const teacherJobs = ['gates', 'workflows'].map(kind => ({
  name: `design-briefs-train-${kind}`, source: join(packet, `train.${kind}.ir.jsonl`),
  rows: kind === 'gates' ? gate.rows : workflows.rows,
  command: ['node', 'ts-host/scripts/teacher-collector.mjs', join(packet, `train.${kind}.ir.jsonl`),
    join(out, `teacher-${kind}/jobs`), join(out, `teacher-${kind}/results.jsonl`),
    ...teacherArgs,
    '--root-seed', '20261025', '--all', '--workers', '1', '--collection-role', 'teacher',
    ...(kind === 'workflows' ? judgeArgs : [])],
  execution: 'prepared-only', actor: teacher ?? { status: 'unbound' },
  ...(kind === 'workflows' ? { judge: judge ?? { status: 'unbound' } } : {}),
  required_bindings: [...(!teacher ? ['TEACHER_MODEL_ID', 'TEACHER_SERVER'] : []),
    ...(kind === 'workflows' && !judge ? ['JUDGE_MODEL_ID', 'JUDGE_SERVER'] : [])],
}));
await writeFile(join(out, 'teacher-jobs.json'), json({ schema: 'natlang.design-brief-teacher-jobs/1', jobs: teacherJobs,
  note: 'Use only the generation and judging actors authorized by the owner. Explicit bindings identify real models and transports; unresolved placeholders are not launchable. Preparation does not start model calls.' }));
await writeFile(join(out, 'preparation.json'), json({ schema: 'natlang.design-brief-preparation/1',
  packet, release: manifest.release, rl: { ...rlPreparation, rows: gate.rows, group: 4, queue_entries: queue.length },
  teacher_jobs: teacherJobs.length, teacher_binding: teacher, judge_binding: judge,
  workflow_rl: { source: join(packet, workflows.path), rows: workflows.rows,
    status: 'independent-judge-and-live-qualification-required-before-RL-admission' },
  training_admission: false, executed_models: 0, live_qualification: 'pending',
  whole_graph: 'Preserve actual sampled/forced producer turns, consumer imports and occurrence bindings; reward task success with or without Neuralese.' }));
console.log(JSON.stringify({ output: out, exact_train_tasks: gate.rows, rl_samples: gate.rows * 4,
  teacher_jobs: teacherJobs.length, executed_models: 0 }));
