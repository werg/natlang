/**
 * `natlang traces good|bad|fix CALL`: a tester's verdict on a recorded call. It is annotated on the call (the one
 * feedback mechanism, `traces annotate`) and, for a root call of a program function with a known expected result,
 * also becomes an example in that program's `examples.json`, the one example store: `natlang test` checks it and
 * `natlang tune` learns from it.
 */
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { hostname, userInfo } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { saveExample, type SavedExample } from './examples.js';
import type { CallStore } from './store.js';
import type { CallRecord } from './types.js';

export type Verdict = 'good' | 'bad' | 'fix';
export type FeedbackOptions = { verdict: Verdict; call: string; expected?: string; group?: string; program?: string; dryRun: boolean;
  /** Models whose recorded outputs may become expected results (tuning's teachers and our own students); a `good`
   * verdict on another model's output is refused. Absent: every producer is accepted. */
  authorizedProducers?: ReadonlySet<string> };

/** The explicit group (the CLI's `--group`, else `NATLANG_FEEDBACK_GROUP`), else one group per tester and day: `feedback-<YYYYMMDD>-<hash of host and user>`. */
export function feedbackGroup(record: CallRecord, explicit?: string, prefix = 'feedback'): string {
  if (explicit) return explicit;
  let user = 'unknown'; try { user = userInfo().username; } catch { /* no passwd entry */ }
  const who = createHash('sha256').update(`${hostname()}\0${user}`).digest('hex').slice(0, 8);
  return `${prefix}-${record.started_at.slice(0, 10).replaceAll('-', '')}-${who}`;
}

function parseExpected(text: string): unknown {
  try { return JSON.parse(text); } catch { throw new Error(`--expected must be JSON (quote strings: '"billing"'): ${text.slice(0, 80)}`); }
}

/** The nearest directory at or above `start` holding natlang.json: the program a command run there works on. */
export function enclosingProgram(start = process.cwd()): string | undefined {
  for (let dir = resolve(start); ; dir = dirname(dir)) {
    if (existsSync(join(dir, 'natlang.json'))) return dir;
    if (dirname(dir) === dir) return undefined;
  }
}

/** A short name for an example: the function and its first input. */
function exampleName(name: string, inputs: Record<string, unknown>): string {
  const first = Object.values(inputs)[0];
  const shown = (typeof first === 'string' ? first : JSON.stringify(first) ?? '').replace(/\s+/g, ' ').trim();
  return shown ? `${name}: ${shown.length > 48 ? `${shown.slice(0, 47)}…` : shown}` : name;
}

/** The example a call gives, or the one sentence saying what is missing (the annotation is kept either way).
 * `expected: null` asks for the inputs alone. */
function exampleOf(store: CallStore, record: CallRecord, options: Pick<FeedbackOptions, 'program' | 'group' | 'authorizedProducers'>, expected: { value: unknown } | undefined | null,
  prefix = 'feedback'): { root: string; example: Example } {
  const { definition } = record;
  if (record.parent_call_id) throw new Error(`call ${record.call_id} ran inside ${record.parent_call_id}; give feedback on the root call`);
  if (record.audit_of) throw new Error(`call ${record.call_id} is an audit of ${record.audit_of}, not a program call`);
  if (!definition.source?.endsWith('.nl')) throw new Error(`the trace of ${definition.name} records no program .nl file (source ${definition.source ?? 'none'}), so it names no example function`);
  // Calls recorded without a program root belong to the program the command runs in, when that program holds the function.
  const cwdProgram = enclosingProgram();
  const root = options.program ? resolve(options.program) : record.program_root ??
    (cwdProgram && existsSync(resolve(cwdProgram, definition.source)) ? cwdProgram : undefined);
  if (!root) throw new Error(`the trace records no program root and the current directory is not in a program holding ${definition.source}; run this in the program's directory or pass --program DIR`);
  if (!existsSync(resolve(root, definition.source))) throw new Error(`${definition.source} does not exist under ${root}; pass --program DIR for the program that holds it`);
  const inputs: Record<string, unknown> = {};
  for (const [name, ref] of Object.entries(record.inputs)) {
    if (!ref?.complete) throw new Error(`input ${name} of ${record.call_id} was not recorded completely (recording excludes or truncates it)`);
    const value = store.value(ref);
    if (value === undefined) throw new Error(`input ${name} of ${record.call_id} is no longer in the store`);
    inputs[name] = value;
  }
  const base = { name: exampleName(definition.name, inputs), function: definition.source, inputs };
  if (expected === null) return { root, example: { ...base, group: feedbackGroup(record, options.group, prefix) } };
  if (!expected) {
    if (record.outcome !== 'done') throw new Error(`call ${record.call_id} ended ${record.outcome}, so its output is not an expected result; use fix --expected JSON`);
    if (!record.output?.complete) throw new Error(`the output of ${record.call_id} was not recorded completely; use fix --expected JSON`);
    const producer = record.executor.model_id;
    if (options.authorizedProducers && !(producer && options.authorizedProducers.has(producer)))
      throw new Error(`its output came from ${producer ?? `a ${record.executor.kind} executor`}, which is not a tuning teacher or one of our ` +
        `models (tune.teacher, tune.teachers, native profiles); give the right answer yourself with natlang traces fix ${record.call_id} --expected JSON`);
    return { root, example: { ...base, expected: store.value(record.output), group: feedbackGroup(record, options.group),
      producer: producer ?? record.executor.kind } };
  }
  return { root, example: { ...base, expected: expected.value, group: feedbackGroup(record, options.group), producer: 'tester' } };
}

export type Example = SavedExample;

/** Returns what to print; throws (after annotating) when the call cannot become an example. */
export function recordFeedback(store: CallStore, options: FeedbackOptions): string {
  const record = store.call(options.call);
  if (!record) throw new Error(`no recorded call ${options.call} (natlang traces list shows recent calls)`);
  if (options.verdict === 'fix' && options.expected === undefined) throw new Error('natlang traces fix needs the right answer: add --expected JSON');
  const expected = options.expected === undefined ? undefined : { value: parseExpected(options.expected) };
  if (!options.dryRun) store.annotate(options.call, 'feedback', { verdict: options.verdict, ...(expected ? { expected: expected.value } : {}) }, 'cli');
  const said = options.dryRun ? `dry run (nothing annotated) for ${options.call}` : `annotated ${options.call} as ${options.verdict}`;
  if (options.verdict === 'bad' && !expected) return `${said}; no example added (add --expected JSON to give the right answer)`;
  let found: ReturnType<typeof exampleOf>;
  try { found = exampleOf(store, record, options, options.verdict === 'good' ? undefined : expected); }
  catch (error) { throw new Error(`${said}; not added as an example: ${(error as Error).message}`); }
  const path = join(found.root, 'examples.json');
  if (options.dryRun) return `${said}\n${path}: ${JSON.stringify(found.example)}`;
  const status = saveExample(found.root, found.example);
  return `${said}; ${status === 'added' ? 'added to' : status === 'updated' ? 'updated the expected result in' : 'already in'} ${path} ` +
    `(group ${found.example.group}); natlang test checks it`;
}

/**
 * A root program call the model failed and its escalation teacher then ran: its inputs join the program's
 * `examples.json` without an expected result (the teacher's answer is not a verified oracle), in a group of
 * escalations per tester and day, so the next `natlang tune` collects and verifies it like any other root. An input
 * already listed is left as it is. Returns the line to print, or undefined when the call gives no example.
 */
export function recordEscalation(store: CallStore, callId: string, program?: string): string | undefined {
  const record = store.call(callId);
  if (!record) return undefined;
  let found: ReturnType<typeof exampleOf>;
  try { found = exampleOf(store, record, { ...(program ? { program } : {}) }, null, 'escalations'); } catch { return undefined; }
  const path = join(found.root, 'examples.json');
  if (saveExample(found.root, { ...found.example, escalated: callId }) !== 'added') return undefined;
  return `${record.definition.name}: its inputs join ${path} as an example without an expected result (group ${found.example.group}); natlang tune ${found.root} learns it`;
}
