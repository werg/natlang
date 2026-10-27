import type { TuneExecution } from './execution.js';
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
 *
 * Without a configured student, the student is the native profile natlang runs (deriveStudent): its pinned complete
 * checkpoint, this checkout's training interpreter, the native backend serving through that profile, and the
 * profile's `escalate` as the teacher. The rest of the tune block (recipe, rounds, ...) still applies.
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { defaultNatlangConfigDirectory } from '../package/store.js';
import { defaultNativePython, ensureNativeServer, nativeIdentity, type NativeModelSettings } from '../model/native-managed.js';
import type { NativeModelIdentity } from '../native/neuralese-store.js';
import type { PluggableSetting } from '../runtime/pluggable.js';
import type { RoundsSettings } from './schedule.js';
import type { IncrementalSettings } from './incremental.js';

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
  /** Several authorized teacher profiles: `natlang tune --calibrate` measures them and `routeTeacher` picks one (§4.2). */
  readonly teachers?: readonly string[];
  readonly teacherRouting: TeacherRouting;
  readonly backend: 'managed-local' | 'native' | 'vllm';
  /** Train and serve soft instruction/skill states on the managed llama.cpp backend; its server must be the llama.cpp
   * fork with `--native-state` (the native backend always serves them; vLLM cannot). */
  readonly softStates?: boolean;
  /** Invocation budgets shared by all tune execution arms; recorded in evaluation and RL receipts. */
  readonly execution?: TuneExecution;
  /** Default trainer recipe fields (natlang_neuralese.tune.trainer.Recipe), shared across programs. */
  readonly recipe?: Readonly<Record<string, unknown>>;
  /** Incremental builds (plan §8): full refit cadence and how shared-ancestor edits build. */
  readonly incremental?: IncrementalSettings;
  /** SFT/RL round schedule for `natlang tune --rounds` (plan §5.1a). */
  readonly rounds?: RoundsSettings;
  /** Cross-program families (plan §5.5): discovery threshold, minimum programs per cluster, proposal policy mode and
   * the family fit recipe (`rank`, `steps`, ...). */
  readonly families?: { readonly threshold?: number; readonly minPrograms?: number; readonly mode?: PluggableSetting;
    /** Run families automatically after this many newly activated program releases of this base (§5.5). */
    readonly every?: number;
    readonly recipe?: Readonly<Record<string, unknown>> };
  /** Placement envelope (plan §5.2). */
  readonly placement?: PlacementSettings;
  /** The model profile that serves tuned releases during tuning; default: the default profile. */
  readonly servingProfile?: string;
};
/** Teachers below `minSuccess` (development success rate) are never chosen; `mode` selects the routing policy. */
export type TeacherRouting = { readonly minSuccess: number; readonly mode?: PluggableSetting };
export const DEFAULT_TEACHER_ROUTING: TeacherRouting = { minSuccess: 0.5 };
/** Accept the cheapest measured candidate whose development success is at most `maxSuccessLoss` below the unpruned
 * stack; cost = weighted active ranks, exported bytes, distinct stacks and measured latency. */
export type PlacementSettings = { readonly maxSuccessLoss: number;
  readonly costWeights: { readonly rank: number; readonly bytes: number; readonly stack: number; readonly latencyMs: number } };
export const DEFAULT_PLACEMENT: PlacementSettings = { maxSuccessLoss: 0.02, costWeights: { rank: 1, bytes: 1e-6, stack: 4, latencyMs: 0.01 } };

export class TuneSetupError extends Error {
  constructor(message: string) { super(`tune-setup: ${message}; run natlang setup or edit the tune block of ${configPath()}`); this.name = 'TuneSetupError'; }
}
const configPath = (environment: NodeJS.ProcessEnv = process.env) => join(defaultNatlangConfigDirectory(environment), 'config.json');

type ConfigFile = { defaultProfile?: string; tune?: Partial<TuneSettings>;
  profiles?: Record<string, { native?: NativeModelSettings; local?: { modelId?: string }; escalate?: string }> };
const readConfig = (environment: NodeJS.ProcessEnv): ConfigFile => {
  const path = configPath(environment);
  return existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) as ConfigFile : {};
};

/** The native profile a student derives from: `name` when given, else the profile natlang runs (`NATLANG_PROFILE`,
 * else the default; a GGUF export of a native pin stands for its native profile), else the only native profile. */
function studentProfile(config: ConfigFile, environment: NodeJS.ProcessEnv, name?: string): [string, NativeModelSettings] | undefined {
  const profiles = config.profiles ?? {};
  const natives = Object.entries(profiles).filter(([, profile]) => profile.native).map(([key, profile]) => [key, profile.native!] as [string, NativeModelSettings]);
  if (name) return profiles[name]?.native ? [name, profiles[name]!.native!] : undefined;
  const running = environment.NATLANG_PROFILE ?? config.defaultProfile;
  const selected = running ? profiles[running] : undefined;
  if (selected?.native) return [running!, selected.native];
  const gguf = /^native-gguf-([0-9a-f]{12})-/.exec(selected?.local?.modelId ?? '')?.[1];
  if (gguf) { const found = natives.find(([, native]) => native.sha256.startsWith(gguf)); if (found) return found; }
  return natives.length === 1 ? natives[0] : undefined;
}

const derivedDevice = (native: NativeModelSettings) => native.device && native.device !== 'auto' ? native.device
  : existsSync('/dev/nvidiactl') ? 'cuda' : 'cpu';

/** Student settings from a native profile (see the module comment); undefined when no native profile applies. */
export function deriveStudent(environment: NodeJS.ProcessEnv = process.env, name?: string):
  { profile: string; native: NativeModelSettings; student?: StudentSettings; teacher?: string } | undefined {
  const config = readConfig(environment);
  const found = studentProfile(config, environment, name);
  if (!found) return undefined;
  const [profile, native] = found;
  const teacher = config.profiles?.[profile]?.escalate;
  const base = nativeIdentity(native.sha256, environment);
  if (!base) return { profile, native, ...(teacher ? { teacher } : {}) };
  const defaults = defaultNativePython(environment);
  const python = native.python ?? defaults.python ?? 'python3', pythonPath = native.pythonPath ?? (native.python ? [] : defaults.pythonPath ?? []);
  return { profile, native, ...(teacher ? { teacher } : {}), student: { base, checkpoint: native.checkpoint,
    engine: { device: derivedDevice(native), ...(native.dtype ? { dtype: native.dtype } : {}) },
    trainer: [python, '-m', 'natlang_neuralese.tune.build'], ...(pythonPath.length ? { env: { PYTHONPATH: pythonPath.join(':') } } : {}),
    moduleMap: base.architecture } };
}

/** A derived student's exact base identity is what its server reports: start the profile's server once to learn it. */
export async function prepareDerivedStudent(environment: NodeJS.ProcessEnv = process.env, overrides: { student?: string } = {},
  log: NodeJS.WritableStream = process.stderr): Promise<void> {
  const tune = readConfig(environment).tune;
  const name = overrides.student ?? tune?.student;
  if (name && tune?.students?.[name]) return;
  const derived = deriveStudent(environment, name);
  if (!derived || derived.student) return;
  log.write(`natlang tune: reading the base identity of ${derived.profile} from its server\n`);
  await ensureNativeServer(derived.native, environment, log);
  if (!nativeIdentity(derived.native.sha256, environment))
    throw new TuneSetupError(`the server of ${derived.profile} reported no native identity for ${derived.native.sha256.slice(0, 12)}`);
}

export function loadTuneSettings(environment: NodeJS.ProcessEnv = process.env, overrides: { student?: string; teacher?: string } = {}):
  { settings: TuneSettings; student: StudentSettings & { name: string } } {
  const config = readConfig(environment);
  const configured = config.tune ?? {};
  const named = overrides.student ?? configured.student;
  const derived = named && configured.students?.[named] ? undefined : deriveStudent(environment, named);
  if (derived && !derived.student)
    throw new TuneSetupError(`the base identity of native profile ${derived.profile} is not known yet (natlang tune reads it from the profile's server)`);
  if (!config.tune && !derived) throw new TuneSetupError('no tune settings are configured and no native profile can be the student');
  const tune: Partial<TuneSettings> = derived ? { ...configured, student: derived.profile, students: { ...configured.students, [derived.profile]: derived.student! },
    backend: configured.backend ?? 'native', servingProfile: derived.profile, ...(configured.teacher || !derived.teacher ? {} : { teacher: derived.teacher }) } : configured;
  const name = derived ? derived.profile : named;
  if (!name) throw new TuneSetupError('tune.student is not set');
  const student = tune.students?.[name];
  if (!student) throw new TuneSetupError(`student ${name} has no entry under tune.students`);
  for (const key of ['architecture', 'checkpoint_sha256', 'embedding_weight_sha256', 'tokenizer_fingerprint'] as const)
    if (typeof student.base?.[key] !== 'string') throw new TuneSetupError(`tune.students.${name}.base.${key} is missing`);
  if (!Array.isArray(student.trainer) || !student.trainer.length) throw new TuneSetupError(`tune.students.${name}.trainer must be an argv array`);
  if (typeof student.checkpoint !== 'string' || !student.checkpoint) throw new TuneSetupError(`tune.students.${name}.checkpoint is missing`);
  if (typeof student.moduleMap !== 'string') throw new TuneSetupError(`tune.students.${name}.moduleMap is missing`);
  const teacher = overrides.teacher ?? tune.teacher;
  if (!teacher) throw new TuneSetupError(derived ? `no teacher: give profile ${derived.profile} an "escalate" profile, or set tune.teacher`
    : 'tune.teacher (a model profile name) is not set');
  for (const profile of [teacher, ...(tune.teachers ?? [])])
    if (!config.profiles?.[profile]) throw new TuneSetupError(`teacher profile ${profile} does not exist`);
  const backend = tune.backend ?? 'managed-local';
  if (backend === 'managed-local' && !student.baseGgufSha256) throw new TuneSetupError(`tune.students.${name}.baseGgufSha256 is required for the managed-local backend`);
  return { settings: { student: name, students: tune.students!, teacher, backend, ...(tune.teachers?.length ? { teachers: tune.teachers } : {}),
    ...(tune.execution ? { execution: tune.execution } : {}),
    teacherRouting: { ...DEFAULT_TEACHER_ROUTING, ...tune.teacherRouting }, ...(tune.families ? { families: tune.families } : {}), ...(tune.recipe ? { recipe: tune.recipe } : {}), ...(tune.rounds ? { rounds: tune.rounds } : {}), ...(tune.softStates !== undefined ? { softStates: tune.softStates } : {}), ...(tune.incremental ? { incremental: tune.incremental } : {}),
    ...(tune.servingProfile ? { servingProfile: tune.servingProfile } : {}),
    placement: { ...DEFAULT_PLACEMENT, ...tune.placement, costWeights: { ...DEFAULT_PLACEMENT.costWeights, ...tune.placement?.costWeights } } },
    student: { ...student, name } };
}
