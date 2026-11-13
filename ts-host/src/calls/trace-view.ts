/**
 * `natlang traces show CALL`: what the model did in one call, readably. Per turn the model's reasoning, its text, the
 * tool calls it made with their results or rejections and the child calls they started, then the outcome. Long texts
 * are cut at a sensible size unless `full`; `--events` and `--json` stay the raw views.
 */
import type { CallStore, CallSummary } from './store.js';
import type { CallRecord } from './types.js';

type Event = Record<string, unknown>;
export type TraceViewOptions = { full?: boolean };

const json = (value: unknown) => { const shown = JSON.stringify(value); return shown === undefined ? 'undefined' : shown; };

/** `text` cut to `chars` characters and `lines` lines, saying how much was left out. */
function cut(text: string, chars: number, lines: number, full: boolean): string {
  if (full) return text;
  let shown = text.split('\n').slice(0, lines).join('\n');
  if (shown.length > chars) shown = shown.slice(0, chars);
  const left = text.length - shown.length;
  return left > 0 ? `${shown.trimEnd()} … (${left} more characters; --full shows all)` : shown;
}

const indent = (text: string, prefix: string) => text.split('\n').map(line => `${prefix}${line}`.trimEnd()).join('\n');
const seconds = (ms: unknown) => `${(Number(ms) / 1000).toFixed(1)} s`;

/** The arguments of a tool call: an eval's code as a block, anything else as JSON. */
function callArguments(name: string, args: unknown, full: boolean): string {
  const code = (args as { code?: unknown } | null)?.code;
  if (name === 'eval' && typeof code === 'string') return `eval\n${indent(cut(code, 2000, 30, full), '    | ')}`;
  return `${name} ${cut(json(args), 400, 1, full)}`;
}

function childLine(store: CallStore, child: CallSummary, full: boolean): string {
  const record = store.call(child.call_id);
  const output = record?.output?.complete ? ` → ${cut(json(store.value(record.output)), 200, 1, full)}` : '';
  return `↳ child ${child.definition_name}${output} (${child.executor}, ${child.outcome}; ${child.call_id})`;
}

/** Child call lines; past ten, the first eight and a count. */
function childLines(store: CallStore, children: CallSummary[], full: boolean): string[] {
  const shown = full || children.length <= 10 ? children : children.slice(0, 8);
  return [...shown.map(child => childLine(store, child, full)),
    ...(shown.length < children.length ? [`… ${children.length - shown.length} more child calls (--full shows all)`] : [])];
}

/** The readable view of a call: `record` when it finished, else the row of a running or interrupted call. */
export function renderTrace(store: CallStore, call: { record?: CallRecord; running?: Record<string, unknown> }, options: TraceViewOptions = {}): string {
  const full = !!options.full;
  const record = call.record;
  const callId = record?.call_id ?? String(call.running!.call_id);
  const events = store.events(callId);
  const children = store.children(callId);
  const lines: string[] = [];
  if (record) {
    const value = (ref: Parameters<CallStore['value']>[0]) => ref?.complete ? store.value(ref) : ref ? `(not recorded: ${ref.reason ?? 'incomplete'})` : undefined;
    lines.push(`${record.definition.name} (${record.definition.source ?? record.definition.site}): ${record.outcome} in ${seconds(record.cost.wall_ms)}, ` +
      `${record.cost.model_requests} model request${record.cost.model_requests === 1 ? '' : 's'}, ${record.cost.tokens_in}+${record.cost.tokens_out} tokens`);
    lines.push(`call ${record.call_id}${record.parent_call_id ? ` (inside ${record.parent_call_id})` : ''}; ${record.executor.kind}` +
      `${record.executor.model_id ? ` ${record.executor.model_id}` : ''}${record.executor.case_hash ? ` case ${record.executor.case_hash}` : ''}` +
      `${record.program_root ? `; program ${record.program_root}` : ''}; started ${record.started_at}`);
    for (const [name, ref] of Object.entries(record.inputs)) lines.push(`input ${name}: ${cut(json(value(ref)), 600, 1, full)}`);
    for (const [name, ref] of Object.entries(record.captures)) lines.push(`capture ${name}: ${cut(json(value(ref)), 300, 1, full)}`);
  } else {
    const running = call.running!;
    lines.push(`${String(running.definition_name)} (${String(running.definition_source ?? running.site ?? 'inline')}): ` +
      `${running.outcome === 'running' ? 'running' : 'interrupted (its process ended)'} since ${String(running.started_at)}`, `call ${callId}`);
  }

  const attached = new Set<string>();
  let actionIndex = 0;
  if (events?.length) {
    const requests = new Map(events.filter(event => event.kind === 'model_request' && event.phase === 'end').map(event => [event.turn, event]));
    for (const event of events) {
      const kind = event.kind, phase = event.phase;
      if (kind === 'model_turn') {
        const request = requests.get(event.turn), tokens = event.completion_tokens ?? request?.completion_tokens;
        const facts = [...(typeof tokens === 'number' ? [`${tokens} tokens`] : []), ...(request?.duration_ms !== undefined ? [seconds(request.duration_ms)] : [])];
        lines.push('', `turn ${String(event.turn)}${facts.length ? ` (${facts.join(', ')})` : ''}`);
      } else if (kind === 'model_request' && phase === 'error') {
        lines.push('', `turn ${String(event.turn)}: the model request failed: ${cut(String(event.error), 400, 4, full)}`);
      } else if (kind === 'proposal' && phase === 'generated') {
        if (typeof event.reasoning === 'string' && event.reasoning.trim()) lines.push(indent(`thinking: ${cut(event.reasoning.trim(), 600, 8, full)}`, '  '));
        if (typeof event.text === 'string' && event.text.trim()) lines.push(indent(`says: ${cut(event.text.trim(), 600, 8, full)}`, '  '));
      } else if (kind === 'action') {
        const name = String(event.name);
        lines.push(indent(`→ ${callArguments(name, event.arguments, full)}`, '  '));
        const outcome = String(event.outcome ?? '');
        const result = typeof event.result_text === 'string' ? event.result_text : '';
        const failed = ['error', 'rejected', 'blocked'].includes(outcome);
        lines.push(indent(`${failed ? '✗' : '='} ${outcome}${result ? `: ${cut(result.trim(), failed ? 800 : 400, failed ? 10 : 4, full)}` : ''}`, '    '));
        const started = children.filter(item => item.parent_action_index === actionIndex);
        started.forEach(child => attached.add(child.call_id));
        lines.push(...childLines(store, started, full).map(line => indent(line, '    ')));
        actionIndex++;
      } else if (kind === 'model_incomplete') {
        lines.push(indent(`the reply was incomplete (${String(event.reason)})`, '  '));
      } else if (kind === 'model_protocol_failure') {
        lines.push(indent(`protocol failure ${String(event.code ?? '')}: ${cut(json(event.failure), 400, 4, full)}`, '  '));
      } else if (kind === 'compaction') {
        lines.push(indent(`context compacted (${String(event.elided ?? '?')} elided)${event.note ? `: ${String(event.note)}` : ''}`, '  '));
      } else if (kind === 'decision_readout' && phase === 'scored') {
        const options = event.options as unknown[] | undefined, chosen = Number(event.chosen);
        const probability = (event.probabilities as number[] | undefined)?.[chosen];
        lines.push('', `decision: ${json(options?.[chosen])}${probability !== undefined ? ` (p=${probability.toFixed(2)})` : ''} of ${(options ?? []).map(json).join(', ')}`);
      } else if (kind === 'refinement_check') {
        lines.push(indent(`check ${String(event.predicate ?? '')}${event.path ? ` on ${json(event.path)}` : ''}: ${String(event.outcome)}`, '  '));
      }
    }
  } else if (record?.approach.evals.length) {
    lines.push('', 'the trace events were evicted; the evals it ran:', ...record.approach.evals.map((code, index) => indent(cut(code, 600, 10, full), `  ${index + 1}. `)));
  } else if (events === undefined) lines.push('', 'no trace events are stored for this call');

  const rest = children.filter(child => !attached.has(child.call_id));
  if (rest.length) lines.push('', 'child calls:', ...childLines(store, rest, full).map(line => indent(line, '  ')));
  if (record?.effects.length) {
    lines.push('', 'service calls:');
    for (const effect of record.effects) {
      const args = effect.args?.complete ? store.value(effect.args) : effect.args;
      const result = effect.result?.complete ? store.value(effect.result) : effect.result;
      lines.push(`  ${effect.service}.${effect.method}(${cut(json(args), 160, 1, full).replace(/^\[|\]$/g, '')}) ${effect.error ? `failed: ${effect.error}` : `→ ${cut(json(result), 160, 1, full)}`}`);
    }
  }
  for (const item of store.annotations(callId)) lines.push(`annotation ${item.kind}: ${cut(json(item.value), 200, 1, full)}`);
  for (const item of store.callCases(callId)) lines.push(`case ${item.case_hash}: ${item.role}${item.verdict ? ` ${item.verdict}` : ''}`);
  if (record) {
    const output = record.output?.complete ? store.value(record.output) : undefined;
    lines.push('', record.outcome === 'done' ? `result: ${cut(json(output), 1200, 1, full)}` :
      `outcome: ${record.outcome}${record.detail ? `: ${cut(record.detail, 1200, 12, full)}` : ''}`);
    if (record.executor.case_error) lines.push(`the compiled case stopped (${record.executor.case_error}); the model finished the call`);
  }
  return lines.join('\n');
}
