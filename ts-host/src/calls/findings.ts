/**
 * Findings (§5.3): what mining a function's recorded calls shows about the program or its executor, reported to the
 * program's owner instead of only shaping a decline. Crisp detectors over a study; the writer's `unstable` skips and the
 * judge's `better` verdicts for cases add to them (applications/specializer).
 */
import type { Study } from './specializer.js';
import type { CallStore, FindingInput } from './store.js';

const canonical = (value: unknown): string => JSON.stringify(value, (_key, item) =>
  item && typeof item === 'object' && !Array.isArray(item) ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => a.localeCompare(b))) : item);
const kind = (value: unknown): string => value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value;

/** Findings in a study: identical inputs handled differently, and service arguments whose type varies between calls. */
export function detectFindings(store: CallStore, subject: Study): FindingInput[] {
  const base = { definitionKey: subject.key, definitionId: subject.definition.id, definitionName: subject.definition.name,
    definitionSource: subject.definition.source ?? null };
  const findings: FindingInput[] = [];
  // The same inputs, different service calls: the instructions leave the decision open, or the executor is unreliable.
  const byInputs = new Map<string, typeof subject.examples>();
  for (const example of subject.examples) {
    const key = canonical(example.args);
    byInputs.set(key, [...(byInputs.get(key) ?? []), example]);
  }
  const split = [...byInputs.values()].filter(list => new Set(list.map(example => example.behavior)).size > 1);
  if (split.length) findings.push({ ...base, kind: 'inconsistent',
    summary: `identical inputs led to different service calls in ${split.length} case${split.length > 1 ? 's' : ''}`,
    detail: { examples: split.slice(0, 5).map(list => ({ inputs: list[0]!.args, did: [...new Set(list.map(example => example.behavior))],
      calls: list.slice(0, 6).map(example => example.callId) })) } });
  // A service argument passed as a number in some calls and a string in others (or other mixed types).
  const types = new Map<string, Map<string, string[]>>();
  for (const record of subject.records.values()) for (const effect of record.effects) {
    const args = store.value(effect.args);
    if (!Array.isArray(args)) continue;
    args.forEach((arg, position) => {
      const at = `${effect.service}.${effect.method} argument ${position + 1}`;
      const seen = types.get(at) ?? new Map<string, string[]>();
      seen.set(kind(arg), [...(seen.get(kind(arg)) ?? []), record.call_id]);
      types.set(at, seen);
    });
  }
  for (const [at, seen] of types) {
    const kinds = [...seen.keys()].filter(item => item !== 'undefined');
    if (kinds.length < 2) continue;
    findings.push({ ...base, kind: 'argument-types', summary: `${at} is passed as ${kinds.join(' or ')} in different calls`,
      detail: { argument: at, calls: Object.fromEntries([...seen].map(([type, ids]) => [type, ids.slice(0, 5)])) } });
  }
  return findings;
}

/** A case the judge found better than the executor on some calls: the executor got those calls wrong. */
export function betterFinding(subject: Study, caseLabel: string, calls: readonly string[]): FindingInput | undefined {
  if (!calls.length) return undefined;
  return { definitionKey: subject.key, definitionId: subject.definition.id, definitionName: subject.definition.name,
    definitionSource: subject.definition.source ?? null, kind: 'executor-worse',
    summary: `for calls that did ${caseLabel}, a compiled case did better than the executor (judged against the instructions)`,
    detail: { calls: calls.slice(0, 10) } };
}
