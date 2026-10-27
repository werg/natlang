/**
 * `natlang doctor`'s model, tuning and program status: what a tester needs to know before running a program on our
 * trained models. Which checkpoint the profile pins and whether a newer one exists, whether its server is running,
 * what a tuned program runs on, how many examples wait for the next `natlang tune`, and whether tuning is configured.
 */
import { existsSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { currentRelease } from '../adaptation/weights/release.js';
import type { LoadedModelConfiguration } from '../model/config.js';
import { findNativeCheckpoint, nativeIdentity, nativeServing, newerNativeCheckpoint, type NativeModelSettings } from '../model/native-managed.js';
import { readRootExamples } from '../tune/collect.js';
import { loadTuneSettings } from '../tune/settings.js';

/** The native pin a profile runs: its own `native` block, or the pin a GGUF export of one names. */
function nativeOf(selected: LoadedModelConfiguration, profiles: Record<string, { native?: NativeModelSettings }>):
  { profile: string; native: NativeModelSettings; gguf: boolean } | undefined {
  const choice = selected.choice;
  if (choice.kind === 'external' && choice.managedNative) return { profile: selected.name, native: choice.managedNative, gguf: false };
  const sha = choice.kind === 'managed-local' ? /^native-gguf-([0-9a-f]{12})-/.exec(choice.local?.modelId ?? choice.model ?? '')?.[1] : undefined;
  const found = sha ? Object.entries(profiles).find(([, profile]) => profile.native?.sha256.startsWith(sha)) : undefined;
  return found ? { profile: found[0], native: found[1].native!, gguf: true } : undefined;
}

export async function modelStatus(selected: LoadedModelConfiguration, profiles: Record<string, { native?: NativeModelSettings; escalate?: string }>,
  programRoot: string | undefined, environment: NodeJS.ProcessEnv = process.env): Promise<Record<string, unknown>> {
  const report: Record<string, unknown> = { escalate: selected.profile.escalate ?? null };
  const found = nativeOf(selected, profiles);
  if (found) {
    const { native } = found, newer = newerNativeCheckpoint(native, selected.nativeSources);
    report.nativeModel = { profile: found.profile, ...(found.gguf ? { servedAs: 'gguf' } : {}), checkpoint: native.sha256.slice(0, 12),
      step: native.step ?? null, pinned: existsSync(native.checkpoint), actorView: !!native.actorView && existsSync(native.actorView.path),
      identity: !!nativeIdentity(native.sha256, environment), server: found.gguf ? null : await nativeServing(native),
      newer: newer ? `step ${newer.step ?? '?'}: natlang setup --native ${native.source}` : null };
  }
  try {
    const { settings, student } = loadTuneSettings(environment);
    report.tuning = { student: student.name, base: student.base.checkpoint_sha256.slice(0, 12), teacher: settings.teacher, backend: settings.backend };
  } catch (error) { report.tuning = { unavailable: (error as Error).message }; }
  if (programRoot) {
    const root = resolve(programRoot), release = currentRelease(root);
    const examples = (() => { try { return readRootExamples(root); } catch { return []; } })();
    const program: Record<string, unknown> = { root, examples: examples.length,
      withExpected: examples.filter(example => 'expected' in example).length,
      escalations: examples.filter(example => 'escalated' in example).length };
    if (release) {
      const base = release.manifest.base.checkpoint_sha256, located = findNativeCheckpoint(base, { programRoot: root, sources: selected.nativeSources }, environment);
      const current = found?.native.step;
      program.release = release.manifest.digest.slice(0, 12);
      program.base = base.slice(0, 12);
      program.baseStep = located?.step ?? null;
      program.baseAvailable = !!located;
      program.runsOn = !located ? `untuned on ${selected.name} (base missing: natlang setup --native CHECKPOINT)`
        : located.pinned ? 'its tuned release, on its pinned base' : `its tuned release once pinned (natlang setup --program ${root})`;
      if (current !== undefined && located?.step !== undefined && current > located.step)
        program.retune = `tuned at step ${located.step}; ${found!.profile} is at step ${current}: natlang tune ${root}`;
    } else program.release = null;
    if (examples.length && !release) program.next = `natlang tune ${root}`;
    report.program = { name: basename(root), ...program };
  }
  return report;
}

/** A program root to report on: the explicit one, else the current directory when it holds tuning state. */
export const programForStatus = (explicit?: string) => explicit ?? (existsSync(join(process.cwd(), 'tune.examples.jsonl'))
  || existsSync(join(process.cwd(), '.natlang', 'tune')) ? process.cwd() : undefined);
