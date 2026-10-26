/**
 * `natlang tune` setup (plans/HIERARCHICAL_LORA_TUNING.md §1.1): the student, its trainer, the authorized teacher and
 * the serving backend come from the `tune` block of the natlang configuration (`natlang setup` writes it) and are
 * reused across programs. A missing piece is an actionable error naming the setting.
 *
 * {
 *   "tune": {
 *     "student": "lfm2.5-350m",
 *     "students": { "lfm2.5-350m": { "base": { native identity }, "checkpoint": "/path", "engine": { "device": "cuda" },
 *                                     "trainer": ["python", "-m", "natlang_neuralese.tune.build"], "hfConfigDir": "/path",
 *                                     "env": { ... }, "baseGgufSha256": "...", "moduleMap": "lfm2" } },
 *     "teacher": "teacher-profile-name",
 *     "backend": "managed-local"
 *   }
 * }
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { defaultNatlangConfigDirectory } from '../package/store.js';
import type { NativeModelIdentity } from '../native/neuralese-store.js';

export type StudentSettings = {
  /** The exact base the adapters are trained on. */
  readonly base: NativeModelIdentity;
  /** The exact native checkpoint the trainer loads, and its engine options. */
  readonly checkpoint: string;
  readonly engine?: { readonly device?: string; readonly dtype?: string };
  /** argv of the trainer entry; `--plan <file> --out <dir>` are appended. */
  readonly trainer: readonly string[];
  readonly env?: Readonly<Record<string, string>>;
  readonly workdir?: string;
  /** sha256 of the base GGUF the managed backend serves; GGUF exports bind it. */
  readonly baseGgufSha256?: string;
  /** Native-to-HF module table name for exports (tune/trainer.py tables). */
  readonly moduleMap: string;
  /** HF config directory of the base, for GGUF LoRA conversion. */
  readonly hfConfigDir?: string;
};
export type TuneSettings = {
  readonly student: string;
  readonly students: Readonly<Record<string, StudentSettings>>;
  /** A model profile name: the authorized teacher that executes the program during collection. */
  readonly teacher: string;
  readonly backend: 'managed-local' | 'native';
};

export class TuneSetupError extends Error {
  constructor(message: string) { super(`tune-setup: ${message}; run natlang setup or edit the tune block of ${configPath()}`); this.name = 'TuneSetupError'; }
}
const configPath = (environment: NodeJS.ProcessEnv = process.env) => join(defaultNatlangConfigDirectory(environment), 'config.json');

export function loadTuneSettings(environment: NodeJS.ProcessEnv = process.env, overrides: { student?: string; teacher?: string } = {}):
  { settings: TuneSettings; student: StudentSettings & { name: string } } {
  const path = configPath(environment);
  const config = existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) as { tune?: Partial<TuneSettings>; profiles?: Record<string, unknown> } : {};
  const tune = config.tune;
  if (!tune) throw new TuneSetupError('no tune settings are configured');
  const name = overrides.student ?? tune.student;
  if (!name) throw new TuneSetupError('tune.student is not set');
  const student = tune.students?.[name];
  if (!student) throw new TuneSetupError(`student ${name} has no entry under tune.students`);
  for (const key of ['architecture', 'checkpoint_sha256', 'embedding_weight_sha256', 'tokenizer_fingerprint'] as const)
    if (typeof student.base?.[key] !== 'string') throw new TuneSetupError(`tune.students.${name}.base.${key} is missing`);
  if (!Array.isArray(student.trainer) || !student.trainer.length) throw new TuneSetupError(`tune.students.${name}.trainer must be an argv array`);
  if (typeof student.checkpoint !== 'string' || !student.checkpoint) throw new TuneSetupError(`tune.students.${name}.checkpoint is missing`);
  if (typeof student.moduleMap !== 'string') throw new TuneSetupError(`tune.students.${name}.moduleMap is missing`);
  const teacher = overrides.teacher ?? tune.teacher;
  if (!teacher) throw new TuneSetupError('tune.teacher (a model profile name) is not set');
  if (!config.profiles?.[teacher]) throw new TuneSetupError(`teacher profile ${teacher} does not exist`);
  const backend = tune.backend ?? 'managed-local';
  if (backend === 'managed-local' && !student.baseGgufSha256) throw new TuneSetupError(`tune.students.${name}.baseGgufSha256 is required for the managed-local backend`);
  return { settings: { student: name, students: tune.students!, teacher, backend }, student: { ...student, name } };
}
