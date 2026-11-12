import { createHash } from 'node:crypto';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value as Record<string, unknown>)
    .sort(([a], [b]) => a.localeCompare(b)).map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(',')}}`;
  return JSON.stringify(value);
}
const sha256 = (value: string | Uint8Array): string => createHash('sha256').update(value).digest('hex');

/** The trajectory record of one model turn: the request's context and the response, as training reads them. */
export function trajectoryTurn(request: ModelTurnRequest, response: ModelTurn): Record<string, unknown> {
  const raw = response.raw_response as Record<string, unknown> | undefined;
  const message = ((raw?.choices as Record<string, unknown>[] | undefined)?.[0]?.message ?? {}) as Record<string, unknown>;
  const planned = Object.hasOwn(response, 'execution_plan');
  const retainedReasoning = planned ? response.execution_plan :
    response.reasoning ?? message.reasoning_content ?? message.reasoning ?? message.thinking ?? null;
  return { phase: 'action', ...(request.invocation_id ? { invocation_id: request.invocation_id } : {}), context: structuredClone(request.messages),
    request_sha256: sha256(canonical(Object.fromEntries(Object.entries(request).filter(([key]) => key !== "invocation_id")))),
    model_response: { calls: structuredClone(response.calls ?? []), text: response.text ?? '',
      raw_calls: structuredClone(response.raw_calls ?? []),
      ...(response.tool_parse ? { tool_parse: structuredClone(response.tool_parse),
        ...(raw ? { raw_response: structuredClone(raw), raw_response_canonical: canonical(raw) } : {}) } : {}),
      ...(raw?.x_natlang_rl_capture && typeof raw.x_natlang_rl_capture === 'object' ?
        { rl_sampling: structuredClone(raw.x_natlang_rl_capture) } : {}),
      ...(Array.isArray((raw?.native_state as {imports?:unknown}|undefined)?.imports) ?
        { native_state_imports: structuredClone((raw?.native_state as {imports:unknown[]}).imports) } : {}),
      ...(raw?.pi_stream_observation && typeof raw.pi_stream_observation === 'object' ?
        { pi_stream_observation: structuredClone(raw.pi_stream_observation) } : {}),
      ...(raw?.pi_reply_diagnostic && typeof raw.pi_reply_diagnostic === 'object' ?
        { provider_reply_diagnostic: structuredClone(raw.pi_reply_diagnostic) } : {}),
      ...(planned ? { execution_plan: response.execution_plan } : {}),
      ...(response.completion_tokens === undefined ? {} : { completion_tokens: response.completion_tokens }),
      ...(response.prompt_tokens === undefined ? {} : { prompt_tokens: response.prompt_tokens }),
      ...(response.protocol_failure ? {protocol_failure:structuredClone(response.protocol_failure)} : {}),
      ...(response.transport_provenance ? { transport_provenance: structuredClone(response.transport_provenance) } : {}) },
    tools_offered: structuredClone(request.tools),
    ...(request.response_tools ? { response_tools: structuredClone(request.response_tools),
      model_request: structuredClone(request), model_request_canonical: canonical(Object.fromEntries(Object.entries(request).filter(([key]) => key !== 'invocation_id'))) } : {}), assistant: { content: response.text ?? '',
      ...(planned ? { execution_plan: response.execution_plan } : {}),
      reasoning: retainedReasoning,
      calls: (response.calls ?? []).map(([tool, args]) => ({ tool, source_tool: tool, arguments: args, call_id: null })) },
    ...(raw?.natlang_action_provenance && typeof raw.natlang_action_provenance === 'object' ?
      { action_provenance: structuredClone(raw.natlang_action_provenance) } : {}),
    raw_response_sha256: raw ? sha256(canonical(raw)) : null };
}

type PromptSegment = { kind: string; token_ids?: number[]; producing?: boolean[] };
type Sample = Record<string, unknown>;
const PROMPT_DELTA_MIN = 64;
const SAMPLE_KEYS = 'index,input_role,logp,query_own_position,terminal,token_id';

/** Ordinary sampled tokens as columns (`samples_columns`), when the capture has exactly the regular form; else null. */
function sampleColumns(samples: unknown, inputs: unknown): Record<string, unknown> | null {
  if (!Array.isArray(samples) || !samples.length || !Array.isArray(inputs) || inputs.length > samples.length) return null;
  const start = (samples[0] as Sample)?.query_own_position;
  if (typeof start !== 'number') return null;
  const ids: number[] = [], logps: number[] = [], terminal: number[] = [];
  for (const [index, sample] of (samples as Sample[]).entries()) {
    if (!sample || Object.keys(sample).sort().join(',') !== SAMPLE_KEYS || sample.index !== index ||
        sample.input_role !== 'ordinary' || sample.query_own_position !== start + index ||
        typeof sample.token_id !== 'number' || typeof sample.logp !== 'number' || typeof sample.terminal !== 'boolean') return null;
    ids.push(sample.token_id); logps.push(sample.logp);
    if (sample.terminal) terminal.push(index);
  }
  for (const [index, input] of (inputs as Sample[]).entries())
    if (!input || Object.keys(input).length !== 3 || input.sample_index !== index || input.token_id !== ids[index] ||
        input.input_role !== 'ordinary') return null;
  return { token_id: ids, logp: logps, query_own_start: start, terminal_at: terminal, generated_inputs: inputs.length };
}

/**
 * Disk: the RL capture of each call (`model_response.rl_sampling`) is stored compactly; readers expand it exactly
 * (`task_rollouts.expand_captures`).
 * - A call's prompt mostly repeats an earlier call's (the previous call of the invocation plus its reply: about 83% of
 *   prompt tokens in RL episodes). A prompt that is one `tokens` segment becomes `{kind: 'tokens_delta', turn, keep,
 *   token_ids, producing_at}`: the first `keep` tokens and `producing` flags of trajectory turn `turn`'s prompt, then
 *   `token_ids`, producing exactly at `producing_at`. The base is the latest call of an invocation sharing the longest prefix.
 * - Ordinary samples (`{index, token_id, logp, input_role: 'ordinary', terminal, query_own_position}` in sequence, and
 *   `generated_inputs` repeating their first n tokens) become `samples_columns: {token_id[], logp[], query_own_start,
 *   terminal_at[], generated_inputs: n}`. Any other form (top-k receipts, forced or producing tokens) stays as it is.
 * Returns a new trajectory; the input is not modified.
 */
export function compactCaptures(trajectory: unknown): unknown {
  if (!Array.isArray(trajectory)) return trajectory;
  const latest = new Map<unknown, { turn: number; ids: number[]; producing: boolean[] }>();
  return trajectory.map((step, turn) => {
    const sampling = (step as { model_response?: { rl_sampling?: Record<string, unknown> } })?.model_response?.rl_sampling;
    if (!sampling || typeof sampling !== 'object') return step;
    let compact: Record<string, unknown> = sampling;
    const columns = sampleColumns(sampling.samples, sampling.generated_inputs);
    if (columns) {
      const { samples: _samples, generated_inputs: _inputs, ...rest } = compact;
      compact = { ...rest, samples_columns: columns };
    }
    const prompt = sampling.prompt as PromptSegment[] | undefined;
    const segment = Array.isArray(prompt) && prompt.length === 1 ? prompt[0] : undefined;
    if (segment && segment.kind === 'tokens' && Array.isArray(segment.token_ids) && Array.isArray(segment.producing) &&
        segment.producing.length === segment.token_ids.length && Object.keys(segment).length === 3) {
      const ids = segment.token_ids, producing = segment.producing;
      let best: { turn: number; keep: number } | undefined;
      for (const base of latest.values()) {
        const limit = Math.min(base.ids.length, ids.length);
        let keep = 0;
        while (keep < limit && base.ids[keep] === ids[keep] && base.producing[keep] === producing[keep]) keep++;
        if (!best || keep > best.keep) best = { turn: base.turn, keep };
      }
      latest.set((step as { invocation_id?: unknown }).invocation_id ?? null, { turn, ids, producing });
      if (best && best.keep >= PROMPT_DELTA_MIN) compact = { ...compact, prompt: [{ kind: 'tokens_delta', turn: best.turn,
        keep: best.keep, token_ids: ids.slice(best.keep),
        producing_at: producing.slice(best.keep).flatMap((flag, index) => (flag ? [index] : [])) }] };
    }
    if (compact === sampling) return step;
    const s = step as { model_response: Record<string, unknown> };
    return { ...s, model_response: { ...s.model_response, rl_sampling: compact } };
  });
}
