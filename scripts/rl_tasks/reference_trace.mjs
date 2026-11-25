#!/usr/bin/env node
// The reference run of each case, call by call (scripts/rl_tasks/sources/invented.py: episode budgets and stage tasks).
// usage: node scripts/rl_tasks/reference_trace.mjs RUNTIME_TS_HOST CASES.jsonl OUT.json
// Per case: the reference replays through the collector's own path (replayReference) and the run's invocation traces
// give every model call: `calls` counts the model turns of the whole run (root and children, what the collector's
// episode budget counts), and `children` lists each `.nl` call below the root with its function path, its arguments
// as the call received them, the value it returned and its model turns. A call whose arguments hold a function value
// is listed with `functional: true` (it cannot become a task of its own). `value` is what the run returned and `answers`
// says, for each distinct child call, which `children` entry answered it (`entry`: its index, -1 for none) and how its
// arguments rendered in the call's opening (`shown`: the text match fragments are found in).
import fs from 'node:fs';
const [runtime, input, out] = process.argv.slice(2);
const { replayReference } = await import(runtime + '/dist/teacher/curriculum.js');
const { TOOLS_PROMPT } = await import(runtime + '/dist/native/prompt.js');
const rows = fs.readFileSync(input, 'utf8').split('\n').filter(Boolean).map(JSON.parse);
const holdsFunction = value => value && typeof value === 'object' &&
  (Object.hasOwn(value, '$lambda') || Object.values(value).some(holdsFunction));
const results = [];
for (const record of rows) {
  const row = { id: record.id };
  try {
    const answers = [];
    const { run, trajectory } = await replayReference(record, TOOLS_PROMPT, undefined,
      { onChild: ({ name, entry, shown }) => answers.push({ name, entry, shown: shown.slice(0, 2000) }) });
    row.status = run.outcome.status;
    row.value = run.outcome.value ?? null;
    if (run.outcome.status !== 'done') row.detail = JSON.stringify(run.outcome.detail ?? run.outcome.status).slice(0, 1500);
    row.answers = answers;
    row.accepted = run.outcome.accepted === true;
    row.calls = trajectory.length;
    row.children = [];
    for (const trace of run.invocationTraces ?? []) {
      if (!String(trace.definitionId).startsWith('nl:') || trace.parentCallId === null) continue;
      const initial = trace.events.find(event => event.kind === 'state' && event.phase === 'initial');
      const lambda = initial?.value?.$lambda ?? {};
      // The call's value as the host received it (a decision readout returns through no return_result action)
      const output = trace.events.filter(event => event.kind === 'host_capture' &&
        event.capture_kind === 'invocation_output' && event.complete === true).at(-1);
      const returned = trace.events.filter(event => event.kind === 'action' && event.name === 'return_result' &&
        event.outcome === 'completed').at(-1);
      const done = trace.outcome === 'done' && (output !== undefined || returned?.arguments?.status === 'success');
      row.children.push({ path: String(trace.definitionId).slice(3), name: trace.name, outcome: trace.outcome,
        args: lambda.args ?? null, functional: holdsFunction(lambda.args ?? {}),
        returned: done, value: output !== undefined ? output.value : returned?.arguments?.value ?? null,
        status: done ? 'success' : returned?.arguments?.status ?? null,
        turns: trace.events.filter(event => event.kind === 'model_turn').length });
    }
  } catch (e) { row.error = String(e).slice(0, 400); }
  results.push(row);
}
fs.writeFileSync(out, JSON.stringify({ cases: rows.length, results }, null, 1) + '\n');
console.log(JSON.stringify({ cases: rows.length, calls: results.map(r => r.calls ?? null) }));
