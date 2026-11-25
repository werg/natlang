#!/usr/bin/env node
// Authored trajectories rendered by the runtime (scripts/neuralese_data/authored_trajectories.py render): each
// natlang.program/2 case's authored reference (its root actions and the scripted answer of every judgment call,
// curriculum.reference) runs through the collector's own execution path (curriculum.replayReference), and every model
// request is recorded as the runtime sent it (messages, tools) with the authored reply as its target. The reply's
// reasoning is left empty: the reasoning between actions is filled later (filled_reasoning.py), never a stock note.
// Neuralese is read as the serving engines read it by default (ts-host model/in-context-neuralese.ts): a learned
// operator's call (`<call>#operator-<n>`, function `view`) is a turn of its own whose target is the authored summary
// and whose `captures` name the state it produced; a consumer's messages show that state as a `native_state` part
// (engines splice it), and a `Neuralese<T>` value as its text in place (`--no-neuralese`: no Neuralese transport).
// usage: node scripts/rl_tasks/reference_episodes.mjs RUNTIME_TS_HOST CASES.jsonl OUT.jsonl [--no-neuralese]
// One output line per case: {id, status, accepted, value, turns: [{invocation_id, function, messages, tools, target}]}.
import fs from 'node:fs';
const [runtime, input, out, ...flags] = process.argv.slice(2);
if (!runtime || !input || !out) throw new Error('usage: reference_episodes.mjs RUNTIME_TS_HOST CASES.jsonl OUT.jsonl');
const { replayReference, callName, OPERATOR_CALL } = await import(runtime + '/dist/teacher/curriculum.js');
const { createInContextNeuralese } = await import(runtime + '/dist/model/in-context-neuralese.js');
const operator = request => /#operator-\d+$/.test(request.invocation_id ?? '');
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
    const neuralese = flags.includes('--no-neuralese') ? undefined : createInContextNeuralese();
    const seen = new Map();   // invocation -> its turns so far (a call's turns are sequential)
    const { run } = await replayReference(record, TOOLS_PROMPT, () => '', {
      ...(neuralese ? { neuralese } : {}),
      onTurn: (request, response) => {
        const turn = seen.get(request.invocation_id ?? '') ?? 0;
        seen.set(request.invocation_id ?? '', turn + 1);
        row.turns.push({ invocation_id: request.invocation_id ?? null, turn,
          function: operator(request) ? OPERATOR_CALL : callName(request.messages), messages: request.messages,
          tools: request.tools ?? [], target: targetOf(response, row.turns.length) });
      } });
    // A spliced state (an operator's summary, a compact_history note) names the turn that produced it: the learner
    // computes the state from that turn's prompt and reply (authored_trajectories.py graph_calls).
    for (const capture of neuralese?.captures() ?? []) {
      const producer = row.turns.find(t => (t.invocation_id ?? '') === capture.invocation_id && t.turn === capture.turn);
      if (!producer) throw new Error(`the producing turn of ${capture.id} was not recorded`);
      (producer.captures ??= []).push({ id: capture.id, ...(capture.select ? { select: capture.text } : {}) });
    }
    for (const t of row.turns) delete t.turn;
    row.status = run.outcome.status;
    row.accepted = run.outcome.accepted === true;
    row.value = run.outcome.value ?? null;
    accepted += row.accepted ? 1 : 0;
  } catch (e) { row.error = String(e).slice(0, 400); row.accepted = false; }
  fs.appendFileSync(out, JSON.stringify(row) + '\n');
}
console.log(JSON.stringify({ cases: rows.length, accepted }));
