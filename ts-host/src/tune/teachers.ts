/**
 * Teacher calibration and routing (plans/HIERARCHICAL_LORA_TUNING.md §4.2).
 *
 * Every candidate teacher profile runs the same development roots without weights; the measurement (success, failures,
 * tokens and elapsed time, retries included) is kept per exact executor identity and example set, so a changed actor or
 * runtime is recalibrated rather than inheriting another's numbers. The minimum task quality is a crisp filter; choosing
 * among the teachers that meet it is the pluggable policy `routeTeacher`: the crisp frontier rule (default) or
 * applications/tuner/routeTeacher.nl. A cheap weak teacher cannot win: success is compared before cost.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { ProgramDescriptor } from '../adaptation/types.js';
import { tuneDirectory } from '../adaptation/weights/release.js';
import type { ModelDriver } from '../runtime/runtime.js';
import { pluggable, type PluggableSetting } from '../runtime/pluggable.js';
import type { SplitExample } from './collect.js';
import { evaluateRoots, rootSuccess } from './evaluate.js';

export type TeacherCalibration = { readonly schema: 'natlang.tune-teacher-calibration/v1'; readonly teacher: string; readonly profile: string;
  readonly examples: string; readonly roots: number; readonly success: number; readonly failures: number;
  readonly tokensPerRoot: number; readonly msPerRoot: number };
export type TeacherChoice = { readonly schema: 'natlang.tune-teacher-choice/v1'; readonly teacher: string | null; readonly profile: string | null;
  readonly reason: string; readonly policy: 'crisp' | 'nl'; readonly minSuccess: number; readonly calibrations: readonly TeacherCalibration[] };
export type TeacherCandidate = { readonly profile: string; readonly id: string; readonly driver: ModelDriver };

const sha = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const rate = (item: TeacherCalibration) => item.roots ? item.success / item.roots : 0;
const writeJson = (path: string, value: unknown) => { writeFileSync(path + '.tmp', JSON.stringify(value, null, 1)); renameSync(path + '.tmp', path); };

/** Measure each candidate on the same development roots; a stored measurement for the same identity and roots is reused. */
export async function calibrateTeachers(options: { root: string; program: ProgramDescriptor; candidates: readonly TeacherCandidate[];
  examples: readonly SplitExample[]; log?: (line: string) => void }): Promise<TeacherCalibration[]> {
  if (!options.examples.length) throw new Error('teacher calibration needs development root examples');
  const examples = sha(options.examples.map(example => example.id).sort());
  const directory = join(tuneDirectory(options.root), 'teachers');
  mkdirSync(directory, { recursive: true });
  const results: TeacherCalibration[] = [];
  for (const candidate of options.candidates) {
    const path = join(directory, sha([candidate.id, examples]).slice(0, 24) + '.json');
    if (existsSync(path)) { results.push(JSON.parse(readFileSync(path, 'utf8'))); continue; }
    options.log?.(`calibrating teacher ${candidate.profile} on ${options.examples.length} development roots`);
    const arm = await evaluateRoots({ root: options.root, program: options.program, driver: candidate.driver, weights: [],
      examples: options.examples, arm: 'teacher', release: null, partition: 'development' });
    const roots = arm.roots.length;
    const record: TeacherCalibration = { schema: 'natlang.tune-teacher-calibration/v1', teacher: candidate.id, profile: candidate.profile, examples, roots,
      success: arm.roots.reduce((sum, root) => sum + rootSuccess(root), 0), failures: arm.roots.filter(root => root.error).length,
      tokensPerRoot: arm.roots.reduce((sum, root) => sum + root.promptTokens + root.completionTokens, 0) / roots,
      msPerRoot: arm.roots.reduce((sum, root) => sum + root.ms, 0) / roots };
    writeJson(path, record);
    results.push(record);
  }
  return results;
}

/** The crisp frontier rule, step for step what routeTeacher.nl is asked to do. */
export function frontierTeacher(eligible: readonly TeacherCalibration[]): { teacher: string | null; reason: string } {
  if (!eligible.length) return { teacher: null, reason: 'no teacher meets the minimum success rate' };
  const best = Math.max(...eligible.map(rate));
  let kept = eligible.filter(item => rate(item) >= best - 0.05);
  const reliable = kept.filter(item => item.failures <= 0.2 * item.roots);
  if (reliable.length) kept = reliable;
  const fewest = Math.min(...kept.map(item => item.tokensPerRoot));
  const chosen = kept.filter(item => item.tokensPerRoot <= fewest * 1.1).sort((a, b) => a.msPerRoot - b.msPerRoot)[0]!;
  return { teacher: chosen.teacher, reason: `success ${chosen.success}/${chosen.roots} (best rate ${best.toFixed(2)}), ` +
    `${Math.round(chosen.tokensPerRoot)} tokens and ${Math.round(chosen.msPerRoot)} ms per root` };
}

export type RouteFunction = (candidates: string, minSuccess: number) => unknown;

/** Choose the collection teacher. Only measured candidates meeting `minSuccess` are offered to either policy; an answer
 * naming anything else is not accepted, and the crisp choice serves instead. */
export async function routeTeacher(calibrations: readonly TeacherCalibration[], settings: { minSuccess: number; mode?: PluggableSetting },
  nl?: RouteFunction): Promise<TeacherChoice> {
  const eligible = calibrations.filter(item => item.roots > 0 && rate(item) >= settings.minSuccess);
  const crisp = () => frontierTeacher(eligible);
  const route = pluggable<[], { teacher: string | null; reason: string; policy: 'crisp' | 'nl' }>({
    crisp: () => ({ ...crisp(), policy: 'crisp' }),
    nl: async () => {
      if (!nl) throw new Error('routeTeacher: the nl policy needs applications/tuner/routeTeacher.nl');
      const answer = await nl(JSON.stringify(eligible.map(({ teacher, roots, success, failures, tokensPerRoot, msPerRoot }) =>
        ({ teacher, roots, success, failures, tokensPerRoot: Math.round(tokensPerRoot), msPerRoot: Math.round(msPerRoot) }))), settings.minSuccess) as
        { teacher?: unknown; reason?: unknown } | null;
      const named = eligible.find(item => item.teacher === answer?.teacher);
      if (!named) return { ...crisp(), reason: `the routing policy named no eligible teacher; crisp rule: ${crisp().reason}`, policy: 'crisp' };
      return { teacher: named.teacher, reason: typeof answer?.reason === 'string' ? answer.reason : '(no reason given)', policy: 'nl' };
    } }, settings.mode, { name: 'routeTeacher', default: 'crisp', serve: 'crisp', same: (a, b) => a.teacher === b.teacher })();
  const chosen = await route;
  return { schema: 'natlang.tune-teacher-choice/v1', teacher: chosen.teacher, reason: chosen.reason, policy: chosen.policy,
    profile: calibrations.find(item => item.teacher === chosen.teacher)?.profile ?? null, minSuccess: settings.minSuccess, calibrations };
}

export function saveTeacherChoice(root: string, choice: TeacherChoice): void {
  mkdirSync(tuneDirectory(root), { recursive: true });
  writeJson(join(tuneDirectory(root), 'teacher.json'), choice);
}

/**
 * The routed teacher profile for collection, when several candidates are configured. The choice stands while its
 * calibration covers exactly the current candidates' identities; a changed actor or runtime needs `--calibrate`.
 */
export function routedTeacherProfile(root: string, candidates: readonly { profile: string; id: string }[]): string {
  const path = join(tuneDirectory(root), 'teacher.json');
  const fix = 'run `natlang tune <program> --calibrate` to measure the configured teachers';
  if (!existsSync(path)) throw new Error(`several teachers are configured and none is chosen yet; ${fix}`);
  const choice = JSON.parse(readFileSync(path, 'utf8')) as TeacherChoice;
  const measured = new Set(choice.calibrations.map(item => item.teacher));
  const changed = candidates.filter(item => !measured.has(item.id)).map(item => item.profile);
  if (choice.teacher && !candidates.some(item => item.id === choice.teacher)) changed.push(`${choice.profile} (chosen, no longer configured)`);
  if (changed.length) throw new Error(`teacher ${changed.join(', ')} changed or was added since calibration; ${fix}`);
  if (!choice.teacher || !choice.profile) throw new Error(`no configured teacher met the minimum success rate ${choice.minSuccess} at calibration ` +
    `(${choice.reason}); configure a stronger authorized teacher or lower tune.teacherRouting.minSuccess`);
  return choice.profile;
}
