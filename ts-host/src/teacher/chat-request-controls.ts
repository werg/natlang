import {validateCoefficientRuntime} from '../model/native-coefficient-policy.js';

/** Collection-specific OpenAI-compatible wire settings. No messages, tools or credentials. */
export function chatRequestControls(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('chat request controls must be an object');
  const controls = value as Record<string, unknown>;
  const allowed = ['max_tokens', 'top_p', 'top_k', 'min_p', 'chat_template_kwargs', 'repetition_penalty', 'reasoning_effort',
    'x_natlang_adapters', 'x_natlang_rl_capture', 'x_natlang_rl_top_k', 'x_natlang_privileged_block', 'x_natlang_native_protocol_recovery', 'guidance',
    'x_natlang_native_coefficient_policy'];
  if (Object.keys(controls).some(key => !allowed.includes(key))) throw new Error('unsupported chat request control');
  if (controls.x_natlang_rl_capture !== undefined && typeof controls.x_natlang_rl_capture !== 'boolean')
    throw new Error('x_natlang_rl_capture must be boolean');
  // Score-centered RL (serve/http.py): the native sampler's top-k log-probs per sampled position, inside the capture.
  // Captured rollouts record top-k by default on the native server; 0 opts out.
  if (controls.x_natlang_rl_top_k !== undefined && (!Number.isSafeInteger(controls.x_natlang_rl_top_k) ||
      Number(controls.x_natlang_rl_top_k) < 0 || Number(controls.x_natlang_rl_top_k) > 4096 || controls.x_natlang_rl_capture !== true))
    throw new Error('x_natlang_rl_top_k must be an integer in [0, 4096] alongside x_natlang_rl_capture: true');
  // JOLT+ teacher attempts: a rl/privileged_context block {insert_at, token_ids} (serve/http.py), verified by the learner.
  const block = controls.x_natlang_privileged_block as { insert_at?: unknown; token_ids?: unknown } | undefined;
  if (block !== undefined && (!block || typeof block !== 'object' || !Number.isSafeInteger(block.insert_at) ||
      !Array.isArray(block.token_ids) || !block.token_ids.length || !block.token_ids.every(Number.isSafeInteger) ||
      Object.keys(block).length !== 2 || controls.x_natlang_rl_capture !== true))
    throw new Error('x_natlang_privileged_block must be {insert_at, token_ids} alongside x_natlang_rl_capture: true');
  if (controls.x_natlang_native_protocol_recovery !== undefined && typeof controls.x_natlang_native_protocol_recovery !== 'boolean')
    throw new Error('x_natlang_native_protocol_recovery must be boolean');
  if (controls.guidance !== undefined && controls.guidance !== false)
    throw new Error('collection guidance control may only disable guidance');
  if (controls.max_tokens !== undefined && (!Number.isSafeInteger(controls.max_tokens) || Number(controls.max_tokens) < 1))
    throw new Error('max_tokens must be a positive integer');
  if (controls.reasoning_effort !== undefined && (typeof controls.reasoning_effort !== 'string' ||
      !['none', 'minimal', 'low', 'medium', 'high'].includes(controls.reasoning_effort)))
    throw new Error('reasoning_effort must be none, minimal, low, medium or high');
  // S7 RL rollouts sample an adapter policy held by the Neuralese server: [{id, scale}] (serve/http.py).
  if (controls.x_natlang_adapters !== undefined && (!Array.isArray(controls.x_natlang_adapters) ||
      !controls.x_natlang_adapters.every(item => item && typeof item === 'object' && !Array.isArray(item) &&
        typeof (item as { id?: unknown }).id === 'string' && Number.isFinite((item as { scale?: unknown }).scale) &&
        Object.keys(item).every(key => key === 'id' || key === 'scale'))))
    throw new Error('x_natlang_adapters must be a list of {id, scale} adapter references');
  if(controls.x_natlang_native_coefficient_policy!==undefined)validateCoefficientRuntime(controls.x_natlang_native_coefficient_policy);
  const serialized = JSON.stringify(controls);
  if (/"(?:apiKey|api_key|authorization|headers|messages|tools|endpoint)"\s*:/i.test(serialized)) throw new Error('credentials and conversation content are forbidden in chat controls');
  return JSON.parse(serialized);
}
