#!/usr/bin/env node
/**
 * Collector result rows -> `natlang.tune-turn/v1` rows of soft states that stood in the calls' prompts (memetic
 * guidance refinement: training/neuralese/natlang_neuralese/rl/memetic_workspace.py `refine_guidance`).
 *
 * stdin: one JSON object {rows: [{row, reward}], states: [{text, binding, function}], component}
 * stdout: one tune turn per line.
 *
 * Each action step whose opening carries one guidance state (the state inside `<natlang_program_guidance>`, whatever
 * its ID: the runtime primes guidance in the call that first uses it) becomes a turn of the guidance the call reads
 * (its function's, else the program-wide one, as `guidanceForCall` picks): the prompt with the state replaced by the text it was written from (so `tune.build` primes
 * the binding's soft state from that text at that position, `tune/soft.py`), the step's own sampled reply as the target
 * (teacher `rl`: every sampled token), and the action evidence the tune pool import computes
 * (`tune/pool.ts` `actionEvidenceForTrajectory`). Only rows whose reward is 1 are passed in: verified successes.
 */
import { actionEvidenceForTrajectory } from '../dist/tune/pool.js';
import { nativeStateSentinel, nativeStateSentinelIds } from '../dist/native/neuralese.js';

let input = '';
for await (const chunk of process.stdin) input += chunk;
const { rows, states, component } = JSON.parse(input);
/** The message with `state` replaced by its text, and how many times it occurred. */
function replaced(message, state) {
  const sentinel = nativeStateSentinel(state.id);
  if (typeof message.content === 'string') {
    const count = message.content.split(sentinel).length - 1;
    return [count ? { ...message, content: message.content.split(sentinel).join(state.text) } : message, count];
  }
  if (!Array.isArray(message.content)) return [message, 0];
  let count = 0;
  const parts = message.content.map(part => {
    if (part?.type === 'native_state' && part.id === state.id) { count++; return { type: 'text', text: state.text }; }
    if (part?.type === 'text' && part.text.includes(sentinel)) { count += part.text.split(sentinel).length - 1; return { type: 'text', text: part.text.split(sentinel).join(state.text) }; }
    return part;
  });
  if (!count) return [message, 0];
  const text = parts.every(part => part?.type === 'text') ? parts.map(part => part.text).join('') : undefined;
  return [{ ...message, content: text ?? parts }, count];
}

const flat = message => typeof message.content === 'string' ? message.content : Array.isArray(message.content) ?
  message.content.map(part => part?.type === 'text' ? part.text : part?.type === 'native_state' ? nativeStateSentinel(part.id) : '').join('') : '';
const OPEN = '<natlang_program_guidance>\n', CLOSE = '\n</natlang_program_guidance>';
/** The guidance state of a call and the guidance it stands for: [{id, text, binding}] or undefined. */
function guidanceOf(context) {
  const found = [];
  for (const message of context) {
    const text = flat(message);
    for (let at = text.indexOf(OPEN); at >= 0; at = text.indexOf(OPEN, at + 1)) {
      const end = text.indexOf(CLOSE, at);
      if (end >= 0) found.push(...nativeStateSentinelIds(text.slice(at + OPEN.length, end)));
    }
  }
  if (found.length !== 1) return undefined;
  const fn = context.map(flat).join('\n').match(/You are inside this call: ([A-Za-z_$][\w$]*)\(/)?.[1];
  const entry = states.find(state => state.function === fn) ?? states.find(state => state.function == null);
  return entry ? { ...entry, id: found[0] } : undefined;
}

for (const [index, { row, reward }] of rows.entries()) {
  if (reward !== 1) continue;
  const steps = row.trajectory ?? [];
  const evidence = actionEvidenceForTrajectory(row, steps, false);
  const oracle = row.outcome?.oracle?.accepted === true;
  const counters = new Map();
  for (const [at, step] of steps.entries()) {
    if (step.phase !== 'action') continue;
    const call = step.invocation_id ?? null, turn = (counters.get(call ?? '') ?? 0) + 1;
    counters.set(call ?? '', turn);
    const state = guidanceOf(step.context ?? []);
    if (!state) continue;
    let total = 0;
    const messages = (step.context ?? []).map(message => { const [out, count] = replaced(message, state); total += count; return out; });
    if (total !== 1) continue;
    const response = step.model_response ?? {}, action = evidence[at];
    // Only verified actions are positive targets (tune/rows.py refuses others).
    if (action.actionDisposition !== 'admissible' || response.protocol_failure) continue;
    process.stdout.write(JSON.stringify({ schema: 'natlang.tune-turn/v1', root: `${row.id ?? 'row'}:${index}`, component,
      partition: 'train', binding: state.binding, call, turn, messages, tools: step.tools_offered ?? [],
      reply: { ...(Array.isArray(response.calls) && response.calls.length ? { calls: response.calls } : {}),
        ...(response.text !== undefined ? { text: response.text } : {}),
        ...(typeof step.assistant?.reasoning === 'string' && step.assistant.reasoning ? { reasoning: step.assistant.reasoning } : {}),
        ...(response.protocol_failure ? { truncated: true } : {}) },
      teacher: 'rl', source: 'memetic-support', texts: { instructions: state.text },
      evidence: { rootOracle: oracle, enclosingSuccess: true, callOutcome: null, actionPolicy: action.actionPolicy,
        actionDisposition: action.actionDisposition, failedHistory: action.failedHistory, actionEvents: action.actionEvents } }) + '\n');
  }
}
