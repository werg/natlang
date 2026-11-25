#!/usr/bin/env node
// Authored trajectories rendered by the runtime (scripts/neuralese_data/authored_trajectories.py render): each
// natlang.program/2 case's authored reference (its root actions and the scripted answer of every judgment call,
// curriculum.reference) runs through the collector's own execution path (curriculum.replayReference), and every model
// request is recorded as the runtime sent it (messages, tools) with the authored reply as its target. The reply's
// reasoning is left empty: the reasoning between actions is filled later (filled_reasoning.py), never a stock note.
// usage: node scripts/rl_tasks/reference_episodes.mjs RUNTIME_TS_HOST CASES.jsonl OUT.jsonl
// One output line per case: {id, status, accepted, value, turns: [{invocation_id, function, messages, tools, target}]}.
import fs from 'node:fs';
const [runtime, input, out] = process.argv.slice(2);
if (!runtime || !input || !out) throw new Error('usage: reference_episodes.mjs RUNTIME_TS_HOST CASES.jsonl OUT.jsonl');
const { replayReference, callName } = await import(runtime + '/dist/teacher/curriculum.js');
const { TOOLS_PROMPT } = await import(runtime + '/dist/native/prompt.js');

/** The authored reply as the OpenAI-compatible message a model would have returned: a decision readout's JSON text,
 * else its tool calls (arguments as JSON text). */
function targetOf(response, index) {
  const calls = response.calls ?? [];
  if (!calls.length) return { role: 'assistant', content: response.text ?? '' };
  return { role: 'assistant', content: response.text ?? '', tool_calls: calls.map(([name, args], i) => (
    { id: `authored_${index}_${i}`, type: 'function', function: { name, arguments: JSON.stringify(args) } })) };
}

const rows = fs.readFileSync(input, 'utf8').split('\n').filter(Boolean).map(JSON.parse);
fs.writeFileSync(out, '');
let accepted = 0;
for (const record of rows) {
  const row = { id: record.id, turns: [] };
  try {
    const { run } = await replayReference(record, TOOLS_PROMPT, () => '', {
      onTurn: (request, response) => row.turns.push({ invocation_id: request.invocation_id ?? null,
        function: callName(request.messages), messages: request.messages, tools: request.tools ?? [],
        target: targetOf(response, row.turns.length) }) });
    row.status = run.outcome.status;
    row.accepted = run.outcome.accepted === true;
    row.value = run.outcome.value ?? null;
    accepted += row.accepted ? 1 : 0;
  } catch (e) { row.error = String(e).slice(0, 400); row.accepted = false; }
  fs.appendFileSync(out, JSON.stringify(row) + '\n');
}
console.log(JSON.stringify({ cases: rows.length, accepted }));
