#!/usr/bin/env node
// Replay of authored program trajectories (scripts/rl_tasks/sources/authored_programs.py): each natlang.program/2 case's
// authored trajectory (curriculum.reference: the root's actions and every other call's scripted turns or answer) runs
// through the collector's own path (curriculum.replayReference), with no model. Per case it reports what the run
// returned, which trajectory entry answered each call (`answers`), the actions that failed, and the whole call tree:
// every invocation the runtime recorded (`.nl` and inline calls, std combinator spans, iterateOn loops, pluggable
// parts), with its parent, the actions of its turns (eval code in full), a decision readout's choice, and the value
// it returned. authored_programs.py checks consistency from it and computes the program statistics.
// usage: node scripts/rl_tasks/authored_replay.mjs RUNTIME_TS_HOST CASES.jsonl OUT.json
import fs from 'node:fs';
const [runtime, input, out] = process.argv.slice(2);
if (!runtime || !input || !out) throw new Error('usage: authored_replay.mjs RUNTIME_TS_HOST CASES.jsonl OUT.json');
const { replayReference } = await import(runtime + '/dist/teacher/curriculum.js');
const { TOOLS_PROMPT } = await import(runtime + '/dist/native/prompt.js');
const rows = fs.readFileSync(input, 'utf8').split('\n').filter(Boolean).map(JSON.parse);

// A function argument is recorded as a live-value preview (`[function nl@eval:1 #46; live value, ...]`), or as a lambda
const holdsFunction = value => value && typeof value === 'object' &&
  (Object.hasOwn(value, '$lambda') || Object.hasOwn(value, '$function') ||
   /^\[function /.test(String(value.$diagnostic_preview ?? '')) || Object.values(value).some(holdsFunction));
const clip = (value, limit) => { const s = typeof value === 'string' ? value : JSON.stringify(value) ?? ''; return s.length > limit ? s.slice(0, limit) : s; };

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
    ...(['rejected', 'refused', 'error'].includes(String(event.outcome)) ? { result: clip(event.result_text ?? '', 600) } : {}) }));
  const extra = {};
  const iteration = events.find(event => event.kind === 'iteration');
  if (iteration) Object.assign(extra, { iteration: { predicate: iteration.predicate ?? null, step: iteration.step ?? null,
    steps: Number(/^(\d+) steps/.exec(trace.detail ?? '')?.[1] ?? 0), reviews: iteration.reviews ?? 0 } });
  const std = events.find(event => event.kind === 'std');
  if (std) extra.combinator = std.combinator;
  return { id: trace.callId, parent: trace.parentCallId ?? null, definition: String(trace.definitionId ?? ''),
    name: trace.name, outcome: trace.outcome, detail: clip(trace.detail ?? '', 400),
    type: lambda.type ?? null, args: lambda.args ?? null, functional: holdsFunction(lambda.args ?? {}),
    turns: events.filter(event => event.kind === 'model_turn').length,
    decision: readout ? { options: readout.options ?? [], chosen: readout.chosen ?? null } : null,
    actions, value: output !== undefined ? output.value : returned?.arguments?.value ?? null, ...extra };
}

const results = [];
for (const record of rows) {
  const row = { id: record.id };
  try {
    const answers = [];
    const { run, trajectory } = await replayReference(record, TOOLS_PROMPT, () => '',
      { onChild: ({ name, entry, shown }) => answers.push({ name, entry, shown: shown.slice(0, 2000) }) });
    row.status = run.outcome.status;
    row.accepted = run.outcome.accepted === true;
    row.value = run.outcome.value ?? null;
    if (run.outcome.status !== 'done') row.detail = clip(run.outcome.detail ?? run.outcome.status, 2000);
    row.answers = answers;
    row.model_calls = trajectory.length;
    row.failed_actions = ((run.outcome.action_ledger ?? [])).filter(event => ['rejected', 'refused', 'error'].includes(String(event.outcome)))
      .map(event => ({ tool: event.name, result: clip(event.result_text ?? '', 800) }));
    row.tree = (run.invocationTraces ?? []).map(node);
  } catch (e) { row.error = String(e?.stack ?? e).slice(0, 1500); }
  results.push(row);
}
fs.writeFileSync(out, JSON.stringify({ cases: rows.length, results }) + '\n');
console.log(JSON.stringify({ cases: rows.length, done: results.filter(r => r.status === 'done').length }));
