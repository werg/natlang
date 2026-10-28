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
    tools_offered: structuredClone(request.tools), assistant: { content: response.text ?? '',
      ...(planned ? { execution_plan: response.execution_plan } : {}),
      reasoning: retainedReasoning,
      calls: (response.calls ?? []).map(([tool, args]) => ({ tool, source_tool: tool, arguments: args, call_id: null })) },
    ...(raw?.natlang_action_provenance && typeof raw.natlang_action_provenance === 'object' ?
      { action_provenance: structuredClone(raw.natlang_action_provenance) } : {}),
    raw_response_sha256: raw ? sha256(canonical(raw)) : null };
}
