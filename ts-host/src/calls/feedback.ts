/**
 * `natlang traces good|bad|fix CALL`: a tester's verdict on a recorded call. It is annotated on the call (the one
 * feedback mechanism, `traces annotate`) and, for a root call of a program function with a known expected result,
 * also becomes a root example in that program's `tune.examples.jsonl` (docs/TUNING.md "Examples from testing").
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { hostname, userInfo } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import type { CallStore } from './store.js';
import type { CallRecord } from './types.js';

export type Verdict = 'good' | 'bad' | 'fix';
export type FeedbackOptions = { verdict: Verdict; call: string; expected?: string; group?: string; program?: string; dryRun: boolean;
  /** Models whose recorded outputs may become expected results (tuning's teachers and our own students); a `good`
   * verdict on another model's output is refused. Absent: every producer is accepted. */
  authorizedProducers?: ReadonlySet<string> };

/** Key-sorted JSON: equal inputs compare equal whatever order their keys were recorded in. */
const stable = (value: unknown): string => JSON.stringify(value, (_, v) => v && typeof v === 'object' && !Array.isArray(v) ?
  Object.fromEntries(Object.keys(v).sort().map(key => [key, (v as Record<string, unknown>)[key]])) : v);

/** `--group`, else `NATLANG_FEEDBACK_GROUP`, else one group per tester and day: `feedback-<YYYYMMDD>-<hash of host and user>`. */
export function feedbackGroup(record: CallRecord, explicit?: string, prefix = 'feedback'): string {
  if (explicit) return explicit;
  const fromEnv = process.env.NATLANG_FEEDBACK_GROUP;
  if (fromEnv) return fromEnv;
  let user = 'unknown'; try { user = userInfo().username; } catch { /* no passwd entry */ }
  const who = createHash('sha256').update(`${hostname()}\0${user}`).digest('hex').slice(0, 8);
  return `${prefix}-${record.started_at.slice(0, 10).replaceAll('-', '')}-${who}`;
}

function parseExpected(text: string): unknown {
  try { return JSON.parse(text); } catch { throw new Error(`--expected must be JSON (quote strings: '"billing"'): ${text.slice(0, 80)}`); }
}

/** The example a call gives, or the one sentence saying what is missing (the annotation is kept either way).
 * `expected: null` asks for the input alone. */
function exampleOf(store: CallStore, record: CallRecord, options: Pick<FeedbackOptions, 'program' | 'group' | 'authorizedProducers'>, expected: { value: unknown } | undefined | null,
  prefix = 'feedback'): { root: string; line: Record<string, unknown> } {
  const { definition } = record;
  if (record.parent_call_id) throw new Error(`call ${record.call_id} ran inside ${record.parent_call_id}; give feedback on the root call`);
  if (record.audit_of) throw new Error(`call ${record.call_id} is an audit of ${record.audit_of}, not a program call`);
  if (!definition.source?.endsWith('.nl')) throw new Error(`the trace of ${definition.name} records no program .nl file (source ${definition.source ?? 'none'}), so it names no tune root function`);
  const root = options.program ? resolve(options.program) : record.program_root;
  if (!root) throw new Error('the trace records no program root; pass --program DIR');
  if (!existsSync(resolve(root, definition.source))) throw new Error(`${definition.source} does not exist under ${root}; pass --program DIR for the program that holds it`);
  const input: Record<string, unknown> = {};
  for (const [name, ref] of Object.entries(record.inputs)) {
    if (!ref?.complete) throw new Error(`input ${name} of ${record.call_id} was not recorded completely (recording excludes or truncates it)`);
    const value = store.value(ref);
    if (value === undefined) throw new Error(`input ${name} of ${record.call_id} is no longer in the store`);
    input[name] = value;
  }
  if (expected === null) return { root, line: { function: definition.source, input, group: feedbackGroup(record, options.group, prefix) } };
  let result = expected?.value;
  if (!expected) {
    if (record.outcome !== 'done') throw new Error(`call ${record.call_id} ended ${record.outcome}, so its output is not an expected result; use fix --expected JSON`);
    if (!record.output?.complete) throw new Error(`the output of ${record.call_id} was not recorded completely; use fix --expected JSON`);
    const producer = record.executor.model_id;
    if (options.authorizedProducers && !(producer && options.authorizedProducers.has(producer)))
      throw new Error(`its output came from ${producer ?? `a ${record.executor.kind} executor`}, which is not a tuning teacher or one of our ` +
        `models (tune.teacher, tune.teachers, native profiles); give the right answer yourself with natlang traces fix ${record.call_id} --expected JSON`);
    result = store.value(record.output);
    return { root, line: { function: definition.source, input, expected: result, group: feedbackGroup(record, options.group),
      producer: producer ?? record.executor.kind } };
  }
  return { root, line: { function: definition.source, input, expected: result, group: feedbackGroup(record, options.group), producer: 'tester' } };
}

/** Add `line` to the examples file, or replace the expected result of the same (function, input) line; atomic rewrite. */
export function upsertExample(path: string, line: Record<string, unknown>): 'added' | 'updated' | 'unchanged' {
  const lines = existsSync(path) ? readFileSync(path, 'utf8').split('\n').filter(item => item.trim()) : [];
  const key = stable([line.function, line.input]);
  const found = { status: 'added' as 'added' | 'updated' | 'unchanged' };
  const next = lines.map(text => {
    let item: Record<string, unknown>;
    try { item = JSON.parse(text); } catch { return text; }
    if (stable([item.function, item.input]) !== key) return text;
    // Protected examples are held for final reporting; feedback never changes them.
    if (item.protected === true) throw new Error(`${path} holds this input as a protected example; it is kept unchanged`);
    if ('expected' in item && stable(item.expected) === stable(line.expected)) { found.status = 'unchanged'; return text; }
    found.status = 'updated'; return JSON.stringify({ ...item, expected: line.expected, ...('producer' in line ? { producer: line.producer } : {}) });
  });
  const { status } = found;
  if (status === 'unchanged') return status;
  if (status === 'added') next.push(JSON.stringify(line));
  mkdirSync(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.tmp`;
  writeFileSync(temporary, `${next.join('\n')}\n`); renameSync(temporary, path);
  return status;
}

/** Returns what to print; throws (after annotating) when the call cannot become an example. */
export function recordFeedback(store: CallStore, options: FeedbackOptions): string {
  const record = store.call(options.call);
  if (!record) throw new Error(`usage: natlang traces ${options.verdict} CALL${options.verdict === 'fix' ? ' --expected JSON' : ''} (no call ${options.call})`);
  if (options.verdict === 'fix' && options.expected === undefined) throw new Error('usage: natlang traces fix CALL --expected JSON');
  const expected = options.expected === undefined ? undefined : { value: parseExpected(options.expected) };
  if (!options.dryRun) store.annotate(options.call, 'feedback', { verdict: options.verdict, ...(expected ? { expected: expected.value } : {}) }, 'cli');
  const said = options.dryRun ? `dry run (nothing annotated) for ${options.call}` : `annotated ${options.call} as ${options.verdict}`;
  if (options.verdict === 'bad' && !expected) return `${said}; no example appended (add --expected JSON to give the right answer)`;
  let example: ReturnType<typeof exampleOf>;
  try { example = exampleOf(store, record, options, options.verdict === 'good' ? undefined : expected); }
  catch (error) { throw new Error(`${said}; not added as a tune example: ${(error as Error).message}`); }
  const path = join(example.root, 'tune.examples.jsonl');
  if (options.dryRun) return `${said}\n${path}: ${JSON.stringify(example.line)}`;
  const status = upsertExample(path, example.line);
  return `${said}; ${status === 'added' ? 'appended to' : status === 'updated' ? 'updated the expected result in' : 'already in'} ${path} (group ${example.line.group})`;
}

/**
 * A root program call the model failed and its escalation teacher then ran: its input joins the program's
 * `tune.examples.jsonl` without an expected result (the teacher's answer is not a verified oracle), in a group of
 * escalations per tester and day, so the next `natlang tune` collects and verifies it like any other root. An input
 * already listed is left as it is. Returns the line to print, or undefined when the call gives no example.
 */
export function recordEscalation(store: CallStore, callId: string, program?: string): string | undefined {
  const record = store.call(callId);
  if (!record) return undefined;
  let example: ReturnType<typeof exampleOf>;
  try { example = exampleOf(store, record, { ...(program ? { program } : {}) }, null, 'escalations'); } catch { return undefined; }
  const path = join(example.root, 'tune.examples.jsonl');
  const key = stable([example.line.function, example.line.input]);
  const lines = existsSync(path) ? readFileSync(path, 'utf8').split('\n').filter(item => item.trim()) : [];
  if (lines.some(text => { try { const item = JSON.parse(text); return stable([item.function, item.input]) === key; } catch { return false; } })) return undefined;
  upsertExample(path, { ...example.line, escalated: callId });
  return `${record.definition.name}: its input joins ${path} as an example (group ${example.line.group}); natlang tune ${example.root} learns it`;
}
