/**
 * Running a `.nl` function's `ensures:` postconditions and its recursion `measure:` (runtime/ensures.ts parses them).
 *
 * Crisp checks run first, on read-only copies of the result and the arguments, in the call's own frame (a helper they
 * call is traced under the call). Only when every check holds are the `predicate:` clauses judged, as relational
 * refinements: one decision-readout judge pass each, with the arguments they name as context. Every failure is a
 * refinement failure at `return`, so the executor gets it as a tool error and repairs within the refinement budget.
 */
import { canonicalValue, type RefinementChecker, type RefinementFailure, type RefinementJudge } from '../native/refinement.js';
import type { NativeTraceRecorder } from '../native/trace.js';
import { runInFrame, type Frame } from './context.js';
import { expressionModule, readOnly, type EnsuresClause } from './ensures.js';
import { itemValue } from './callable.js';
import { moduleInstance, moduleInstanceAsync, moduleRealmUsesAsync } from './modules.js';
import type { ItemRecord, ModuleRecord } from './loader.js';

type Owner = { id: string; name: string; source?: string; revision?: string; types: Record<string, string>;
  params: readonly { name: string }[]; codebase: Record<string, unknown>; ensures?: readonly EnsuresClause[]; measure?: string };

// One synthetic module per definition revision: module instances are cached by record identity.
const modules = new Map<string, ModuleRecord>();
function contractsModule(owner: Owner): ModuleRecord {
  const key = JSON.stringify([owner.id, owner.revision ?? '', owner.ensures ?? [], owner.measure ?? '', Object.keys(owner.codebase)]);
  let record = modules.get(key);
  if (!record) {
    record = expressionModule({ ...owner, params: owner.params.map(parameter => parameter.name) }, owner.ensures, owner.measure, Object.keys(owner.codebase));
    if (modules.size >= 512) modules.delete(modules.keys().next().value!);
    modules.set(key, record);
  }
  return record;
}

/** Run export `name` of the owner's contract module on `args` in `frame`, with the companion folder's items in scope. */
async function runExpression(frame: Frame, owner: Owner, name: string, args: unknown[]): Promise<unknown> {
  return runInFrame(frame, async () => {
    const platform = frame.task.runtime.platform;
    const record = contractsModule(owner), level = owner.codebase as Record<string, ItemRecord>;
    const exports = await (moduleRealmUsesAsync(platform) ? moduleInstanceAsync(platform, record, level) : moduleInstance(platform, record, level));
    const items = Object.fromEntries(Object.entries(level).map(([item, child]) => [item, itemValue(child, level, frame, platform)]));
    return (exports[name] as (...given: unknown[]) => Promise<unknown>)(...args.map(arg => readOnly(arg)), items);
  });
}

/** The `measure:` of a call on `inputs`: a non-negative whole number, or a one-sentence reason it is not one. */
export async function measureOf(frame: Frame, owner: Owner, inputs: unknown[]): Promise<number | string> {
  let value: unknown;
  try { value = await runExpression(frame, owner, 'measure', owner.params.map((_, index) => inputs[index])); }
  catch (error) { return `\`${owner.name}\`'s measure ${JSON.stringify(owner.measure)} failed on these arguments: ${(error as Error).message}`; }
  if (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0) return value;
  return `\`${owner.name}\`'s measure ${JSON.stringify(owner.measure)} must be a non-negative whole number, but it is ${canonicalValue(value)} for these arguments.`;
}

export type EnsuresInput = {
  frame: Frame; owner: Owner; inputs: unknown[]; checker: RefinementChecker;
  judges: { judge?: RefinementJudge; escalation?: RefinementJudge };
  callId: string; signal?: AbortSignal; trace?: NativeTraceRecorder; emit: (kind: string, data: Record<string, unknown>) => void;
  untrusted?: { sourceOf(text: string): string | undefined };
};

/** The failures of `result` (a host value) against the owner's `ensures:` clauses; empty when all hold. */
export async function checkEnsures(input: EnsuresInput, result: unknown): Promise<RefinementFailure[]> {
  const { owner, inputs, emit } = input;
  const clauses = owner.ensures ?? [];
  const args = owner.params.map((_, index) => inputs[index]);
  const failures: RefinementFailure[] = [];
  for (const [index, clause] of clauses.entries()) {
    if (!('check' in clause)) continue;
    let verdict: unknown, thrown: string | undefined;
    try { verdict = await runExpression(input.frame, owner, `check_${index}`, [result, ...args]); }
    catch (error) { thrown = (error as Error).message; }
    const holds = verdict === true;
    emit('ensures_check', { call_id: input.callId, check: clause.check, outcome: holds ? 'pass' : 'fail',
      ...(thrown !== undefined ? { error: thrown } : holds ? {} : { value: canonicalValue(verdict) }) });
    if (holds) continue;
    const what = thrown !== undefined ? `threw: ${thrown}` : typeof verdict === 'boolean' ? 'is false' : `gave ${canonicalValue(verdict)}, not true`;
    failures.push({ code: 'refinement-unsatisfied', path: 'return', predicate: clause.message ?? clause.check,
      message: clause.message ? `Revise the result so that ${clause.message.replace(/\.$/, '')} (the postcondition \`${clause.check}\` ${what}).` :
        `Revise the result so that the postcondition \`${clause.check}\` holds (it ${what}).` });
  }
  // Crisp checks come first: a result they refuse is not worth a judge pass.
  if (failures.length) return failures;
  const obligations = clauses.flatMap(clause => 'predicate' in clause ? [{ path: 'return', predicate: clause.predicate, value: result,
    ...(clause.against?.length ? { context: Object.fromEntries(clause.against.map(name =>
      [name, inputs[owner.params.findIndex(parameter => parameter.name === name)]])) } : {}) }] : []);
  if (!obligations.length) return [];
  return input.checker.check(obligations, { phase: 'return', ...input.judges, callId: input.callId, signal: input.signal,
    trace: input.trace, emit, untrusted: input.untrusted });
}
