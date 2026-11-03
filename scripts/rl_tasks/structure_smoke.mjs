#!/usr/bin/env node
// Structure smoke for composed Neuralese call-graph cases (scripts/rl_tasks/sources/composed.py).
// usage: node scripts/rl_tasks/structure_smoke.mjs RUNTIME_TS_HOST CASES.jsonl OUT.json [PER_SHAPE]
// The case reference replays a direct root answer (the collector replay has no Neuralese port), so it does not show
// that the intended call graph runs. This check executes, per shape, the shape's intended orchestration through the
// collector's executeProgram with the teacher-text Neuralese transport (exact text bodies, no vectors): the root runs
// the scripted eval from `curriculum.external.structure.smoke.root`, and every named child answers its call with
// the scripted value for its name (text for Neuralese<string>, a record for typed Neuralese results). It reports
// whether the run finished with the expected answer and which child calls, Neuralese writes and Neuralese
// arguments occurred. It qualifies program structure only: no model, no task outcome, no training admission.
import fs from 'node:fs';
const [runtime, input, out, perShapeArg] = process.argv.slice(2);
const { executeProgram } = await import(runtime + '/dist/teacher/collector.js');
const { callName } = await import(runtime + '/dist/teacher/curriculum.js');
const { openingText } = await import(runtime + '/dist/runtime/conversation.js');
const { TOOLS_PROMPT } = await import(runtime + '/dist/native/prompt.js');
const { createTeacherTextNeuralese } = await import(runtime + '/dist/model/teacher-text-neuralese.js');
const perShape = Number(perShapeArg || 2);
const rows = fs.readFileSync(input, 'utf8').split('\n').filter(Boolean).map(JSON.parse);
const chosen = new Map();
for (const row of rows) {
  const key = row.family + '|' + row.curriculum.shape;
  const list = chosen.get(key) ?? [];
  if (list.length < perShape && row.curriculum.external?.structure?.smoke) { list.push(row); chosen.set(key, list); }
}
const results = [];
for (const [key, list] of chosen) for (const row of list) {
  const smoke = row.curriculum.external.structure.smoke;
  const transport = createTeacherTextNeuralese();
  let step = 0;
  const childCalls = {};
  const errors = [];
  const childTurns = new Map();
  const driver = async request => {
    const context = request.messages, name = callName(context);
    const rootName = row.semantics.root.replace(/\.nl$/, '').split('/').pop();
    for (const message of context) if (message.role === 'tool' && /error|refused|rejected/i.test(String(message.content).slice(0, 200)))
      errors.push(String(message.content).slice(0, 400));
    if (name === rootName) {
      const action = smoke.root[step++];
      return { calls: [action ?? ['return_result', { status: 'failed', reason: 'smoke script ended' }]] };
    }
    let value = smoke.children[name];
    // {"$match": [{"match": "<text>" | [...], "value": v}, ...]}: per-item answers (e.g. one decision per folder file),
    // chosen by text the child's opening contains, like a reference child.
    if (value && typeof value === 'object' && Array.isArray(value.$match)) {
      const opening = openingText(context);
      const hit = value.$match.find(entry => [entry.match].flat().every(text => opening.includes(text)));
      value = hit ? hit.value : undefined;
    }
    if (value === undefined) return { calls: [['return_result', { status: 'failed', reason: 'no smoke answer for ' + name }]] };
    // {"$turns": [[tool, args], ...]}: a child that acts (calls its own children in eval) before returning; turns are
    // counted per child invocation (its opening).
    if (value && typeof value === 'object' && Array.isArray(value.$turns)) {
      const opening = openingText(context);
      const turn = childTurns.get(opening) ?? 0;
      childTurns.set(opening, turn + 1);
      if (turn === 0) childCalls[name] = (childCalls[name] ?? 0) + 1;
      return { calls: [value.$turns[turn] ?? ['return_result', { status: 'failed', reason: 'smoke turns ended' }]] };
    }
    childCalls[name] = (childCalls[name] ?? 0) + 1;
    return { calls: [['return_result', { status: 'success', value }]] };
  };
  let outcome, error;
  try {
    const run = await executeProgram(row, transport.wrap(driver), { systemPrompt: TOOLS_PROMPT, contextTokens: 16384,
      maxTurns: 60, rootSeed: 0, runId: 'structure-smoke:' + row.id, runLabel: 'structure-smoke',
      neuralese: transport.runtime });
    outcome = run.outcome;
  } catch (e) { error = String(e?.stack ?? e).slice(0, 1200); }
  const graph = JSON.stringify(outcome?.execution_graph ?? []);
  results.push({ key, id: row.id, accepted: outcome?.accepted === true, status: outcome?.status, value: outcome?.value,
    expected: row.semantics.expected, detail: outcome?.accepted ? undefined : JSON.stringify(outcome?.detail ?? null).slice(0, 600),
    child_calls: childCalls, neuralese_refs_in_graph: (graph.match(/nzt1_/g) ?? []).length, tool_errors: errors.slice(0, 3), error });
}
const summary = { shapes: chosen.size, runs: results.length, accepted: results.filter(r => r.accepted).length,
  failed: results.filter(r => !r.accepted).map(r => ({ key: r.key, id: r.id, status: r.status, detail: r.detail, error: r.error, tool_errors: r.tool_errors })) };
fs.writeFileSync(out, JSON.stringify({ schema: 'natlang.composed-structure-smoke/1', input, summary, results }, null, 1) + '\n');
console.log(JSON.stringify({ shapes: summary.shapes, runs: summary.runs, accepted: summary.accepted }));
