#!/usr/bin/env node
/**
 * Rewrite collected result rows from the retired blocked and failed tools to return_result with a status.
 *
 *   node scripts/inline-curriculum/migrate-status.mjs IN.results.jsonl OUT.results.jsonl
 *
 * Tool calls (model responses, assistant records, and the calls replayed in later contexts), offered tool schemas,
 * turn notices, the system prompt's finishing sentences, and blocked(...)/failed(...) inside eval code are rewritten.
 * The model's reasoning text is left as it was. Each rewritten row is marked in provenance.finish_surface_migration.
 */
import { readFileSync, writeFileSync } from 'node:fs';

const MIGRATION = 'return_result-status/1';
const RETURN_TOOL = value => ({ type: 'function', function: { name: 'return_result',
  description: 'Finish the call. With status "success", value is the result and must have the declared return type. ' +
    'With status "blocked" (required information is missing; do not guess) or "failed" (the instructions require an invalid ' +
    'or contradictory operation), give the reason instead of a value.',
  parameters: { type: 'object', properties: { status: { type: 'string', enum: ['success', 'blocked', 'failed'] }, value,
    reason: { type: 'string', description: 'For "blocked": what is missing. For "failed": why it cannot be done.' } },
  required: ['status'], additionalProperties: false } } });

const TEXT = [
  ['call return_result with the result, or the blocked tool with what is missing, or the failed tool with why.]',
    'call return_result with status "success" and the result, or status "blocked" with what is missing, or status "failed" with why.]'],
  ['If the task cannot be finished, call the blocked tool with what is missing, or the failed tool with why.]',
    'If the task cannot be finished, call return_result with status "blocked" and what is missing, or status "failed" and why.]'],
  ['Call return_result with a ', 'Call return_result with status "success" and a '],
  ['You can send back a response with return_result, blocked or failed. The return_result tool takes a value of your intended target type.',
    'You send back a response with return_result and a status. With status "success" it takes a value of your intended target type; with status "blocked" or "failed" it takes a reason instead.'],
  ['when it is missing, call blocked instead of substituting something else, and call failed when the instructions cannot be carried out.',
    'when it is missing, return with status "blocked" instead of substituting something else, and with status "failed" when the instructions cannot be carried out.'],
  ['stop and call blocked with what is missing.', 'stop and return with status "blocked" and what is missing.'],
];
const rewriteText = text => TEXT.reduce((out, [from, to]) => out.split(from).join(to), text);
// Finisher calls in eval code: blocked(x) and failed(x) become return_result(undefined, status, x).
const rewriteCode = code => code.replace(/(^|[^.\w$])(blocked|failed)\s*\(/g, (_, lead, name) => `${lead}return_result(undefined, '${name}', `);

/** [name, args] of one call in the new surface. */
function rewriteCall(name, args = {}) {
  if (name === 'blocked') return ['return_result', { status: 'blocked', reason: args.missing ?? args.reason ?? '' }];
  if (name === 'failed') return ['return_result', { status: 'failed', reason: args.message ?? args.reason ?? '' }];
  if (name === 'return_result' && args.status === undefined) return ['return_result', { status: 'success', ...args }];
  if (name === 'eval' && typeof args.code === 'string') return ['eval', { ...args, code: rewriteCode(args.code) }];
  return [name, args];
}
const parse = text => { try { return JSON.parse(text); } catch { return undefined; } };
function rewriteFunctionCall(call) {
  const args = parse(call.function?.arguments ?? '');
  if (!args || typeof args !== 'object') return call;
  const [name, next] = rewriteCall(call.function.name, args);
  return { ...call, function: { ...call.function, name, arguments: JSON.stringify(next) } };
}
function rewriteTools(tools) {
  if (!Array.isArray(tools)) return tools;
  const old = tools.find(tool => tool.function?.name === 'return_result');
  return tools.filter(tool => !['blocked', 'failed'].includes(tool.function?.name))
    .map(tool => tool === old ? RETURN_TOOL(old.function.parameters?.properties?.value ?? {}) : tool);
}
function rewriteMessage(message) {
  const next = { ...message };
  if (typeof next.content === 'string') next.content = rewriteText(next.content);
  if (Array.isArray(next.tool_calls)) next.tool_calls = next.tool_calls.map(rewriteFunctionCall);
  return next;
}
function rewriteTurn(turn) {
  const next = { ...turn };
  if (Array.isArray(next.context)) next.context = next.context.map(rewriteMessage);
  next.tools_offered = rewriteTools(next.tools_offered);
  if (next.model_response) {
    next.model_response = { ...next.model_response };
    if (Array.isArray(next.model_response.calls)) next.model_response.calls = next.model_response.calls.map(([name, args]) => rewriteCall(name, args));
    if (Array.isArray(next.model_response.raw_calls)) next.model_response.raw_calls = next.model_response.raw_calls.map(rewriteFunctionCall);
  }
  if (next.assistant && Array.isArray(next.assistant.calls)) next.assistant = { ...next.assistant, calls: next.assistant.calls.map(call => {
    const [tool, args] = rewriteCall(call.tool, call.arguments);
    return { ...call, tool, source_tool: tool, arguments: args };
  }) };
  return next;
}

export function migrateRow(row) {
  if (row.provenance?.finish_surface_migration === MIGRATION) return row;
  return { ...row, trajectory: (row.trajectory ?? []).map(rewriteTurn),
    provenance: { ...row.provenance, finish_surface_migration: MIGRATION } };
}

const [input, output] = process.argv.slice(2);
if (input && output) {
  const rows = readFileSync(input, 'utf8').split('\n').filter(Boolean).map(line => migrateRow(JSON.parse(line)));
  writeFileSync(output, rows.map(row => JSON.stringify(row)).join('\n') + '\n');
  const left = rows.filter(row => /"name":"(?:blocked|failed)"|"tool":"(?:blocked|failed)"/.test(JSON.stringify(row.trajectory))).length;
  console.log(`${rows.length} rows -> ${output}${left ? ` (${left} still name an old tool)` : ''}`);
}
