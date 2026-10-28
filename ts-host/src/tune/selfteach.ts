/**
 * Teacher-free collection for `natlang tune`: privileged self-teaching (owner, 2026-10-28: no additional teacher
 * generation; JOLT+, plans/neuralese/TRAINING_INSIGHTS_20261028.md section 7).
 *
 * The teacher is the student itself (`tune.teacher: "self"`, the default). The privileged facts come from the shared
 * JOLT+ mechanism (`training/neuralese/natlang_neuralese/rl/privileged_context.py`): `natlang_neuralese.tune.supply
 * privileged` (or any program's own exact oracle through `case_facts`) writes `tune.privileged.jsonl` beside the
 * program (`natlang.tune-privileged/2`): per TRAINING root, its custody receipt (`ProtectedWorlds`: train split,
 * disjoint from every registered evaluation-only packet) and its facts. While collecting, each model turn of such a
 * root sees the facts that apply to its invocation (`applicable`: the root its world-level facts, a child only its own
 * matched value; crisp kind order as `CRISP_ORDER`), rendered as the shared declaration renders them: one system turn
 * with the declared header, after the leading system turn (`PrivilegedRendering`, placement
 * `after_leading_system_turn`). The captured turn keeps the request WITHOUT it, so a training row teaches the student
 * to produce from the plain task what it produced with the reference. Only oracle-verified roots with admissible
 * actions become rows, and a root whose reply echoes the privileged header gives no row. This is the sequence-level
 * (sampling) side of privileged self-teaching; dense per-token scoring of the same graphs is the shared
 * `rl/privileged_scoring.teacher_scores`, never a second path here.
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
import type { DecisionRequest, DecisionScorer, ModelTurnRequest } from '../contracts.js';
import { currentFrame } from '../runtime/context.js';
import { canonicalJson } from '../package/archive.js';
import { tuneDirectory } from '../adaptation/weights/release.js';
import type { CollectedTurn, RootExample, SplitExample } from './collect.js';

/** The teacher setting that selects privileged self-teaching (the default when no teacher is configured). */
export const SELF_TEACHER = 'self';
export const PRIVILEGED_SCHEMA = 'natlang.tune-privileged/2';
export const PRIVILEGED_FILE = 'tune.privileged.jsonl';
/** The shared rendering declaration's header (`PrivilegedRendering.header`); a reply containing it echoed the block. */
export const PRIVILEGED_HEADER = 'Verified reference for this call (training only):';
/** Identity of the rendering below (shared declaration, chat-message form); part of the self teacher's identity. */
export const PRIVILEGED_RENDERING = 'natlang.jolt-privileged-rendering/1:system:after_leading_system_turn';
/** `privileged_context.CRISP_ORDER`: the crisp `selectPrivilegedFacts` order per invocation role. */
export const CRISP_ORDER = { root: ['expected_result', 'expected_stop', 'expected_files', 'expected_effects', 'reference_solution',
  'attempt_feedback', 'reference_failure', 'minimum_sequence', 'decisive', 'teacher_trajectory'], child: ['child_value', 'attempt_feedback'] } as const;

export type PrivilegedFact = { readonly id: string; readonly kind: string; readonly text: string; readonly match?: readonly string[];
  readonly invocation?: string | null };
export type PrivilegedReference = { readonly schema: typeof PRIVILEGED_SCHEMA; readonly function: string; readonly group?: string;
  readonly input: unknown; readonly result: unknown; readonly case_id: string;
  readonly custody: { readonly split: string; readonly protected_sha256: string }; readonly facts: readonly PrivilegedFact[] };
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
    if (value.schema !== PRIVILEGED_SCHEMA || typeof value.function !== 'string' || !('input' in value) || !('result' in value) ||
        !Array.isArray(value.facts) || value.custody?.split !== 'train' || typeof value.custody?.protected_sha256 !== 'string')
      throw new Error(`${path}:${index + 1}: not a ${PRIVILEGED_SCHEMA} reference with training-world custody`);
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

/** `privileged_context.applicable` + `select_crisp`: the facts one invocation sees, in policy order. */
export function selectFacts(facts: readonly PrivilegedFact[], invocation: { readonly role: 'root' | 'child'; readonly name?: string;
  readonly text: string }): PrivilegedFact[] {
  const applicable = facts.filter(fact => fact.kind === 'child_value' ? invocation.role === 'child' && (fact.match ?? []).length > 0 &&
      (fact.match ?? []).every(item => invocation.text.includes(item))
    : fact.kind === 'attempt_feedback' ? !fact.invocation || fact.invocation === invocation.name : invocation.role === 'root');
  const order: readonly string[] = CRISP_ORDER[invocation.role];
  let chosen = applicable.filter(fact => order.includes(fact.kind));
  // An ambiguous child match has no verified answer for this call.
  if (new Set(chosen.filter(fact => fact.kind === 'child_value').map(fact => fact.text)).size > 1)
    chosen = chosen.filter(fact => fact.kind !== 'child_value');
  return chosen.sort((a, b) => order.indexOf(a.kind) - order.indexOf(b.kind));
}

/** The block text (the shared `PrivilegedRendering.turn` body: header, then whole facts); null when nothing applies. */
export function privilegedText(facts: readonly PrivilegedFact[]): string | null {
  return facts.length ? [PRIVILEGED_HEADER, ...facts.map(fact => fact.text)].join('\n\n') : null;
}

/** The request with the block as a system turn after the leading system turn (else first). Never mutates the request. */
export function withPrivileged<T extends { messages: unknown[] }>(request: T, text: string): T {
  const messages = [...request.messages];
  const at = (messages[0] as { role?: unknown } | undefined)?.role === 'system' ? 1 : 0;
  messages.splice(at, 0, { role: 'system', content: text });
  return { ...request, messages };
}

const promptText = (messages: readonly unknown[]) => messages.map(message => {
  const content = (message as { content?: unknown })?.content;
  return typeof content === 'string' ? content : Array.isArray(content) ? content.map(part =>
    typeof (part as { text?: unknown })?.text === 'string' ? (part as { text: string }).text : '').join('') : '';
}).join('\n');

/** The student driver with one root's privileged facts on every generated or scored model turn. */
export function privilegedDriver(student: ModelDriver, reference: PrivilegedReference): ModelDriver {
  const inner = student as (request: ModelTurnRequest, ...rest: unknown[]) => ReturnType<ModelDriver>;
  const privileged = <T extends { messages: unknown[]; invocation_id?: string; invocation_role?: 'root' | 'child' }>(request: T): T => {
    const frame = currentFrame();
    const role = frame ? (frame.chain.length > 1 ? 'child' as const : 'root' as const) : (request.invocation_role ?? 'root');
    const text = privilegedText(selectFacts(reference.facts, { role, name: request.invocation_id, text: promptText(request.messages) }));
    return text ? withPrivileged(request, text) : request;
  };
  const wrapped = Object.assign((request: ModelTurnRequest, ...rest: unknown[]) => inner(privileged(request), ...rest),
    Object.fromEntries(Object.entries(student))) as unknown as ModelDriver & { decide?: DecisionScorer };
  const decide = (student as ModelDriver & { decide?: DecisionScorer }).decide;
  if (decide) {
    const scored: DecisionScorer = (request, signal) => decide(privileged(request), signal);
    if (decide.scoreMany) scored.scoreMany = (requests: DecisionRequest[], signal?: AbortSignal) =>
      decide.scoreMany!(requests.map(privileged), signal);
    wrapped.decide = scored;
  }
  return wrapped;
}

/** Whether a captured reply echoes the privileged block (never a positive target). */
export function leaksPrivileged(turn: Pick<CollectedTurn, 'reply'>): boolean {
  return JSON.stringify([turn.reply.text ?? '', turn.reply.reasoning ?? '', turn.reply.calls ?? []]).includes(PRIVILEGED_HEADER);
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
