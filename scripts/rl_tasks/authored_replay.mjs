#!/usr/bin/env node
// Replay of authored program trajectories (scripts/rl_tasks/sources/authored_programs.py): each natlang.program/2 case's
// authored trajectory (curriculum.reference: the root's actions and every other call's scripted turns or answer) runs
// through the collector's own path (curriculum.replayReference), with no model. Per case it reports what the run
// returned, which trajectory entry answered each call (`answers`, with the call's whole opening), the actions that
// failed, and the whole call tree: every invocation the runtime recorded (`.nl` and inline calls, std combinator spans,
// iterateOn loops, pluggable parts), with its parent, the actions of its turns (eval code in full), a decision readout's
// choice, and the value it returned. authored_programs.py checks consistency from it and computes the program statistics.
//
// A call no entry answers gets a placeholder of its declared type (the first option of a decision, a value shaped by the
// return type's schema) and is reported with `placeholder: true`, so one replay lists every missing entry instead of
// stopping at the first stage that lacks one (`--stop-at-missing` restores the stop).
// usage: node scripts/rl_tasks/authored_replay.mjs RUNTIME_TS_HOST CASES.jsonl OUT.json [--stop-at-missing]
import fs from 'node:fs';
const [runtime, input, out, ...flags] = process.argv.slice(2);
if (!runtime || !input || !out) throw new Error('usage: authored_replay.mjs RUNTIME_TS_HOST CASES.jsonl OUT.json [--stop-at-missing]');
const curriculum = await import(runtime + '/dist/teacher/curriculum.js');
const { replayReference, NO_PLACEHOLDER } = curriculum;
const { TOOLS_PROMPT } = await import(runtime + '/dist/native/prompt.js');
const rows = fs.readFileSync(input, 'utf8').split('\n').filter(Boolean).map(JSON.parse);
const placeholders = !flags.includes('--stop-at-missing') && NO_PLACEHOLDER !== undefined;

// A function argument is recorded as a live-value preview (`[function nl@eval:1 #46; live value, ...]`), or as a lambda
const holdsFunction = value => value && typeof value === 'object' &&
  (Object.hasOwn(value, '$lambda') || Object.hasOwn(value, '$function') ||
   /^\[function /.test(String(value.$diagnostic_preview ?? '')) || Object.values(value).some(holdsFunction));
const clip = (value, limit) => { const s = typeof value === 'string' ? value : JSON.stringify(value) ?? ''; return s.length > limit ? s.slice(0, limit) : s; };
const at = event => Date.parse(event?.observed_at ?? '') || null;

/** A value of a JSON schema's shape: the first enum option, required fields filled, empty arrays. */
function sample(schema, depth = 0) {
  if (!schema || typeof schema !== 'object' || depth > 12) return null;
  if (Array.isArray(schema.enum) && schema.enum.length) return schema.enum[0];
  if ('const' in schema) return schema.const;
  for (const key of ['anyOf', 'oneOf']) if (Array.isArray(schema[key]) && schema[key].length) {
    const branch = schema[key].find(b => b?.type !== 'null') ?? schema[key][0];
    return sample(branch, depth + 1);
  }
  const type = Array.isArray(schema.type) ? schema.type.find(t => t !== 'null') ?? 'null' : schema.type;
  switch (type) {
    case 'string': return 'placeholder';
    case 'number': case 'integer': return typeof schema.minimum === 'number' ? schema.minimum : 0;
    case 'boolean': return true;
    case 'array': return [];
    case 'null': return null;
    case 'object': return Object.fromEntries((schema.required ?? []).map(key => [key, sample(schema.properties?.[key], depth + 1)]));
    default: return schema.properties ? sample({ ...schema, type: 'object' }, depth) : null;
  }
}

/** The placeholder answer of a call no entry matches: a decision readout's first option, else a value of the return
 * type. A call whose type gives no shape (a function result) keeps the failure. */
function placeholder(request) {
  const tools = Array.isArray(request.tools) ? request.tools : [];
  if (!tools.length) {
    const last = String(request.messages?.at(-1)?.content ?? '');
    const listed = /one of these JSON values and nothing else: ([\s\S]*?)\.?\s*$/.exec(last)?.[1];
    try { const options = JSON.parse(`[${listed}]`); if (options.length) return options[0]; } catch { /* fall through */ }
    return NO_PLACEHOLDER;
  }
  const result = tools.find(tool => (tool.function?.name ?? tool.name) === 'return_result');
  const schema = (result?.function?.parameters ?? result?.parameters)?.properties?.value;
  if (!schema || /=>/.test(String(schema.description ?? '')) && !schema.type) return NO_PLACEHOLDER;
  return sample(schema);
}

function node(trace) {
  const events = trace.events ?? [];
  const initial = events.find(event => event.kind === 'state' && event.phase === 'initial');
  const lambda = initial?.value?.$lambda ?? {};
  const output = events.filter(event => event.kind === 'host_capture' && event.capture_kind === 'invocation_output' &&
    event.complete === true).at(-1);
  const returned = events.filter(event => event.kind === 'action' && event.name === 'return_result' &&
    event.outcome === 'completed').at(-1);
  const readout = events.find(event => event.kind === 'decision_readout');
  const actions = events.filter(event => event.kind === 'action').map(event => ({
    tool: event.name, outcome: event.outcome,
    ...(event.name === 'eval' ? { code: String(event.arguments?.code ?? '') } : {}),
    ...(['rejected', 'refused', 'error'].includes(String(event.outcome)) ? { result: clip(event.result_text ?? '', 4000) } : {}) }));
  // When each eval ran (from the model turn that issued it to its action event): a call started inside a window ran
  // from that eval, whichever frame its function was bound to (authored_programs.py places it under the caller).
  const windows = [];
  let turnAt = null;
  for (const event of events) {
    if (event.kind === 'model_turn') turnAt = at(event);
    else if (event.kind === 'action' && event.name === 'eval' && turnAt !== null && at(event) !== null) windows.push([turnAt, at(event)]);
  }
  const start = events.find(event => event.kind === 'invocation' && event.phase === 'start') ?? events[0];
  const end = events.filter(event => event.kind === 'invocation' && event.phase === 'end').at(-1) ?? events.at(-1);
  const extra = {};
  const iteration = events.find(event => event.kind === 'iteration');
  if (iteration) Object.assign(extra, { iteration: { predicate: iteration.predicate ?? null, step: iteration.step ?? null,
    steps: Number(/^(\d+) steps/.exec(trace.detail ?? '')?.[1] ?? 0), reviews: iteration.reviews ?? 0 } });
  const std = events.find(event => event.kind === 'std');
  if (std) extra.combinator = std.combinator;
  return { id: trace.callId, parent: trace.parentCallId ?? null, definition: String(trace.definitionId ?? ''),
    name: trace.name, outcome: trace.outcome, detail: clip(trace.detail ?? '', 2000),
    type: lambda.type ?? null, instructions: lambda.instructions ?? null, args: lambda.args ?? null,
    functional: holdsFunction(lambda.args ?? {}),
    turns: events.filter(event => event.kind === 'model_turn').length,
    decision: readout ? { options: readout.options ?? [], chosen: readout.chosen ?? null } : null,
    started: at(start), ended: at(end), eval_windows: windows,
    actions, value: output !== undefined ? output.value : returned?.arguments?.value ?? null, ...extra };
}

const results = [];
for (const record of rows) {
  const row = { id: record.id };
  try {
    const answers = [];
    const filled = new Set();
    const missing = placeholders ? (request, name) => {
      const value = placeholder(request);
      if (value !== NO_PLACEHOLDER) filled.add(answers.length - 1);
      return value;
    } : undefined;
    const { run, trajectory } = await replayReference(record, TOOLS_PROMPT, () => '',
      { onChild: ({ name, entry, opening, shown }) => answers.push({ name, entry, opening, shown }), missing });
    for (const index of filled) if (answers[index]) answers[index].placeholder = true;
    row.status = run.outcome.status;
    row.accepted = run.outcome.accepted === true;
    row.value = run.outcome.value ?? null;
    if (run.outcome.status !== 'done') row.detail = clip(run.outcome.detail ?? run.outcome.status, 4000);
    row.answers = answers;
    row.model_calls = trajectory.length;
    row.failed_actions = ((run.outcome.action_ledger ?? [])).filter(event => ['rejected', 'refused', 'error'].includes(String(event.outcome)))
      .map(event => ({ tool: event.name, call: event.call_id ?? null, result: clip(event.result_text ?? '', 4000) }));
    row.tree = (run.invocationTraces ?? []).map(node);
  } catch (e) { row.error = String(e?.stack ?? e).slice(0, 4000); }
  results.push(row);
}
fs.writeFileSync(out, JSON.stringify({ cases: rows.length, results }) + '\n');
console.log(JSON.stringify({ cases: rows.length, done: results.filter(r => r.status === 'done').length }));
