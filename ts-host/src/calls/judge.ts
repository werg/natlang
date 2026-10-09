/**
 * Comparing a case with the agent (§6.1). Equal behavior passes without a judge. Otherwise a natural-language decision
 * run by the runtime (not by the specializer) says which behavior carries out the function's instructions better; it
 * sees the two behaviors in an order fixed by the call, not knowing which one is the case.
 */
import { hexDigest } from '../native/hash.js';
import { invokeDefinition } from '../runtime/kernel.js';
import { builtinDefinition } from '../runtime/builtin.js';
import { currentFrame } from '../runtime/context.js';
import type { NatlangRuntime } from '../runtime/runtime.js';
import { sameData, subtreeEffects, type ObservedEffect, type ValueSource } from './replay.js';
import type { CallRecord, Verdict } from './types.js';

/** What one execution did: its value or error, its service calls, the captures it wrote and the files it changed. */
export type Behavior = { value?: unknown; error?: string; effects: ObservedEffect[]; captureWrites?: Record<string, unknown>;
  files?: { path: string; kind: string; text?: string }[] };

/** The behavior a recorded call shows, its descendants' effects included. */
export function recordedBehavior(store: ValueSource, record: CallRecord): Behavior {
  const failed = record.outcome !== 'done';
  return { ...(failed ? { error: `${record.outcome}: ${record.detail}` } : { value: store.value(record.output) }),
    effects: subtreeEffects(store, record).map(effect => ({ service: effect.service, method: effect.method, args: effect.args })),
    captureWrites: Object.fromEntries(record.capture_writes.map(write => [write.name, store.value(write.after)])),
    files: record.folder?.changes.map(change => ({ path: change.path, kind: change.kind,
      ...(change.after?.complete ? { text: store.value(change.after) as string } : {}) })) ?? [] };
}

/** Whether two behaviors are the same, and how they differ. Effects compare in order. */
export function compareBehaviors(reference: Behavior, candidate: Behavior): { equal: boolean; differences: string[] } {
  const differences: string[] = [];
  if (!!reference.error !== !!candidate.error) differences.push(reference.error ? 'the reference failed; the candidate returned a value' :
    `the candidate failed: ${candidate.error}`);
  else if (!reference.error && !sameData(reference.value, candidate.value)) differences.push('the results differ');
  const key = (effect: ObservedEffect) => `${effect.service}.${effect.method}(${JSON.stringify(effect.args)})`;
  const left = reference.effects.map(key), right = candidate.effects.map(key);
  if (left.join('\n') !== right.join('\n')) {
    const sortedSame = [...left].sort().join('\n') === [...right].sort().join('\n');
    differences.push(sortedSame ? 'the same service calls in a different order' : 'the service calls differ');
  }
  if (!sameData(reference.captureWrites ?? {}, candidate.captureWrites ?? {})) differences.push('the captured variables written differ');
  const files = (behavior: Behavior) => (behavior.files ?? []).map(file => `${file.kind} ${file.path} ${file.text ?? ''}`).sort();
  if (!sameData(files(reference), files(candidate))) differences.push('the file changes differ');
  return { equal: differences.length === 0, differences };
}

/** The comparison judge is the built-in program builtin/compareBehaviors.nl. */
const judgeDefinition = () => builtinDefinition('compareBehaviors');

const describe = (behavior: Behavior): string => {
  const cap = (text: string, limit = 6000) => text.length > limit ? `${text.slice(0, limit)} …(${text.length} characters)` : text;
  const json = (value: unknown) => { try { return JSON.stringify(value, null, 1) ?? 'undefined'; } catch { return String(value); } };
  return [behavior.error ? `It failed: ${behavior.error}` : `Result: ${cap(json(behavior.value))}`,
    behavior.effects.length ? `Service calls, in order:\n${behavior.effects.map(effect =>
      `- ${effect.service}.${effect.method}(${cap(JSON.stringify(effect.args ?? []).slice(1, -1), 600)})`).join('\n')}` : 'No service calls.',
    ...(behavior.captureWrites && Object.keys(behavior.captureWrites).length ? [`Variables written: ${cap(json(behavior.captureWrites), 2000)}`] : []),
    ...(behavior.files?.length ? [`Files changed:\n${behavior.files.map(file => `- ${file.kind} ${file.path}${file.text !== undefined ?
      `:\n${cap(file.text, 2000)}` : ''}`).join('\n')}`] : [])].join('\n');
};

/**
 * The verdict for a candidate (a case) against the reference (the agent): `equal` when they behave the same, else the
 * judge's `better`, `equivalent` or `worse`. Runs the judge in a task of `runtime` with compilations off.
 */
export async function judge(runtime: NatlangRuntime, input: { instructions: string; signature: string; inputs: unknown;
  reference: Behavior; candidate: Behavior; annotations?: unknown[]; seed: string }): Promise<{ verdict: Verdict; differences: string[] }> {
  const { equal, differences } = compareBehaviors(input.reference, input.candidate);
  if (equal) return { verdict: 'equal', differences };
  const candidateFirst = parseInt(hexDigest(input.seed).slice(0, 2), 16) % 2 === 0;
  const task = [`Instructions:\n${input.instructions}`, `Signature: ${input.signature}`,
    `Inputs: ${JSON.stringify(input.inputs, null, 1)?.slice(0, 8000)}`,
    ...(input.annotations?.length ? [`Feedback recorded about this call: ${JSON.stringify(input.annotations).slice(0, 3000)}`] : [])].join('\n\n');
  const [first, second] = candidateFirst ? [input.candidate, input.reference] : [input.reference, input.candidate];
  const answer = await runtime.run(() => invokeDefinition(currentFrame()!, judgeDefinition(), [task, describe(first), describe(second)],
    { manifest: { internal: true } }), { specialization: 'off', name: 'judge' }) as 'first' | 'second' | 'equal';
  const candidateWon = (answer === 'first') === candidateFirst;
  return { verdict: answer === 'equal' ? 'equivalent' : candidateWon ? 'better' : 'worse', differences };
}
