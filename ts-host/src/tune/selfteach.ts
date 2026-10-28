/**
 * Teacher-free collection for `natlang tune`: privileged self-teaching (owner, 2026-10-28: no additional teacher
 * generation; JOLT+, plans/neuralese/TRAINING_INSIGHTS_20261028.md section 7).
 *
 * The teacher is the student itself (`tune.teacher: "self"`, the default). On a TRAINING root that has a privileged
 * reference (`tune.privileged.jsonl` beside the program, `natlang.tune-privileged/1`: the exact oracle value of each
 * stage and the root result, written by `natlang_neuralese.tune.supply privileged` or any program's own exact oracle),
 * every model turn sees that reference as privileged context; the captured turn keeps the request WITHOUT it, so a
 * training row teaches the student to produce from the plain task what it produced with the reference. Only
 * oracle-verified roots with admissible actions become rows (the ordinary evidence rules), and a reply that echoes the
 * privileged block is never a target. This is the sequence-level form of privileged self-teaching; RL rounds then
 * optimize the same oracle reward on the student's own samples.
 *
 * Custody: privileged context comes only from training worlds. A reference that names a development or confirmation
 * root is an error; development roots are not collected by the self teacher at all (they only select releases).
 * Existing verified trajectories of the program (imported pool data, earlier teacher collections) are reused before
 * the self teacher runs a root, so admitted data keeps priority over self-collected rows.
 */
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { ModelDriver } from '../runtime/runtime.js';
import type { ModelTurnRequest } from '../contracts.js';
import { canonicalJson } from '../package/archive.js';
import { tuneDirectory } from '../adaptation/weights/release.js';
import type { CollectedTurn, RootExample, SplitExample } from './collect.js';

/** The teacher setting that selects privileged self-teaching (the default when no teacher is configured). */
export const SELF_TEACHER = 'self';
export const PRIVILEGED_SCHEMA = 'natlang.tune-privileged/1';
export const PRIVILEGED_FILE = 'tune.privileged.jsonl';
/** Version of the privileged rendering below; part of the self teacher's identity, so a change recollects. */
export const PRIVILEGED_RENDERING = 'natlang.tune-privileged-rendering/1';
/** Opens every privileged block; a reply containing it echoed the reference and is not a target. */
export const PRIVILEGED_MARKER = 'Verified training reference';

export type PrivilegedReference = { readonly schema: typeof PRIVILEGED_SCHEMA; readonly function: string; readonly group?: string;
  readonly input: unknown; readonly result: unknown; readonly calls?: readonly { readonly function: string; readonly value: unknown }[] };
export type PrivilegedSet = { readonly references: ReadonlyMap<string, PrivilegedReference>; readonly digest: string | null };

const keyOf = (example: Pick<RootExample, 'function' | 'group' | 'input'>) => canonicalJson([example.function, example.group ?? null, example.input]);

/** The program's privileged references (empty without the file), keyed by root function, group and input. */
export function readPrivileged(root: string): PrivilegedSet {
  const path = join(root, PRIVILEGED_FILE);
  if (!existsSync(path)) return { references: new Map(), digest: null };
  const text = readFileSync(path, 'utf8');
  const references = new Map<string, PrivilegedReference>();
  text.split('\n').filter(line => line.trim()).forEach((line, index) => {
    const value = JSON.parse(line) as PrivilegedReference;
    if (value.schema !== PRIVILEGED_SCHEMA || typeof value.function !== 'string' || !('input' in value) || !('result' in value))
      throw new Error(`${path}:${index + 1}: not a ${PRIVILEGED_SCHEMA} reference`);
    const key = keyOf(value);
    if (references.has(key) && canonicalJson(references.get(key)) !== canonicalJson(value))
      throw new Error(`${path}:${index + 1}: two different references for one root`);
    references.set(key, value);
  });
  return { references, digest: createHash('sha256').update(text).digest('hex') };
}

/** The reference of each root that receives one; a reference naming a non-training root is a custody error. */
export function privilegedRoots(set: PrivilegedSet, examples: readonly SplitExample[]): Map<string, PrivilegedReference> {
  const out = new Map<string, PrivilegedReference>();
  for (const example of examples) {
    const reference = set.references.get(keyOf(example));
    if (!reference) continue;
    if (example.partition !== 'train')
      throw new Error(`privileged reference for ${example.partition} root ${example.id} (${example.group ?? 'no group'}): ` +
        'privileged context comes only from training worlds');
    if (example.protected) throw new Error(`privileged reference for protected root ${example.id}`);
    if ('expected' in example && canonicalJson(example.expected) !== canonicalJson(reference.result))
      throw new Error(`privileged reference for root ${example.id} disagrees with its expected value`);
    out.set(example.id, reference);
  }
  return out;
}

/** The self teacher's exact actor identity: the student's base, the privileged references and their rendering. */
export function selfTeacherId(student: { readonly name: string; readonly base: { readonly checkpoint_sha256: string } }, digest: string | null): string {
  return `self-privileged:${student.name}:${student.base.checkpoint_sha256}:${digest ?? 'none'}:${PRIVILEGED_RENDERING}`;
}
export const isSelfTeacher = (teacherId: string) => teacherId.startsWith('self-privileged:');

/** The privileged block one turn sees (model-facing; recorded as unmeasured in plans/MODEL_FACING_CHANGES.md). */
export function privilegedText(reference: PrivilegedReference): string {
  const value = (item: unknown) => JSON.stringify(item);
  const calls = (reference.calls ?? []).map(call => `${call.function} returns ${value(call.value)}`);
  return `${PRIVILEGED_MARKER}: ${[...calls, `the task's final result is ${value(reference.result)}`].join('; ')}. ` +
    'Follow the instructions step by step so that each step reaches its value.';
}

/** The request with the privileged block appended to its first user message (a new final text part for structured
 * content); without a user message it opens the conversation as a system message. Never mutates the request. */
export function withPrivileged(request: ModelTurnRequest, text: string): ModelTurnRequest {
  const messages = request.messages.map(message => message);
  const index = messages.findIndex(message => (message as { role?: unknown })?.role === 'user');
  if (index < 0) return { ...request, messages: [{ role: 'system', content: text }, ...messages] };
  const message = messages[index] as { content?: unknown };
  const content = message.content;
  messages[index] = { ...message, content: Array.isArray(content) ? [...content, { type: 'text', text: `\n\n${text}` }]
    : `${typeof content === 'string' ? content : ''}\n\n${text}` };
  return { ...request, messages };
}

/** The student driver with one root's privileged reference on every model turn (decision readouts pass through). */
export function privilegedDriver(student: ModelDriver, reference: PrivilegedReference): ModelDriver {
  const text = privilegedText(reference);
  const inner = student as (request: ModelTurnRequest, ...rest: unknown[]) => ReturnType<ModelDriver>;
  return Object.assign((request: ModelTurnRequest, ...rest: unknown[]) => inner(withPrivileged(request, text), ...rest),
    Object.fromEntries(Object.entries(student))) as unknown as ModelDriver;
}

/** Whether a captured reply echoes the privileged block (never a positive target). */
export function leaksPrivileged(turn: Pick<CollectedTurn, 'reply'>): boolean {
  return JSON.stringify([turn.reply.text ?? '', turn.reply.reasoning ?? '', turn.reply.calls ?? []]).includes(PRIVILEGED_MARKER);
}

/** Teacher identities of the program's recorded root data other than the self teacher (earlier collections),
 * whose still-valid trajectories the self route reuses first. */
export function recordedTeachers(root: string): string[] {
  const directory = join(tuneDirectory(root), 'data', 'roots');
  if (!existsSync(directory)) return [];
  const teachers = new Set<string>();
  for (const name of readdirSync(directory).filter(file => file.endsWith('.json')).sort()) {
    try {
      const teacher = (JSON.parse(readFileSync(join(directory, name), 'utf8')) as { teacher?: unknown }).teacher;
      if (typeof teacher === 'string' && !isSelfTeacher(teacher)) teachers.add(teacher);
    } catch { /* an unreadable record is not reusable data */ }
  }
  return [...teachers].sort();
}
