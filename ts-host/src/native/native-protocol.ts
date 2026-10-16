/** Exact host-observed native marker failure contract; model text cannot assert it. */
import type { NativeProtocolFailure } from '../contracts.js';
import { canonical } from './canonical-data.js';
type RecordValue = Record<string, unknown>;
const record = (value: unknown): value is RecordValue => !!value && typeof value === 'object' && !Array.isArray(value);
const integer = (value: unknown): value is number => Number.isSafeInteger(value) && Number(value) >= 0;
const codes = new Set(['neuralese-native-state-open', 'neuralese-native-state-close',
  'neuralese-native-state-consume-marker', 'neuralese-native-state-nested']);
const keys = ['schema', 'code', 'message', 'trigger', 'sample_index', 'fed', 'own_input_count',
  'generated_input_count', 'max_tokens', 'stop_token_ids'].sort();

export function nativeProtocolFailure(response: unknown): NativeProtocolFailure | undefined {
  if (!record(response)) return undefined;
  const failure = response.x_natlang_native_failure;
  const capture = response.x_natlang_rl_capture;
  if (failure === undefined) {
    if (record(capture) && (capture.protocol_failure !== undefined ||
        ['natlang.native-rl-sampling/4','natlang.native-structural-capture/2'].includes(String(capture.schema))))
      throw new Error('neuralese-native-protocol-integrity: failed capture lacks its typed failed turn');
    return undefined;
  }
  const choice = Array.isArray(response.choices) ? response.choices[0] : undefined;
  const message = record(choice) ? choice.message : undefined;
  if (!record(failure) || canonical(Object.keys(failure).sort()) !== canonical(keys) ||
      failure.schema !== 'natlang.native-marker-failure/1' || typeof failure.code !== 'string' || !codes.has(failure.code) ||
      typeof failure.message !== 'string' || !failure.message.startsWith(String(failure.code) + ': ') ||
      typeof failure.fed !== 'boolean' || !integer(failure.own_input_count) ||
      !integer(failure.generated_input_count) || !integer(failure.max_tokens) || failure.max_tokens < 1 ||
      !Array.isArray(failure.stop_token_ids) || !failure.stop_token_ids.length ||
      failure.stop_token_ids.some(value => !integer(value)) ||
      canonical(failure.stop_token_ids) !== canonical([...new Set(failure.stop_token_ids)].sort((a, b) => Number(a) - Number(b))) ||
      !record(choice) || choice.finish_reason !== 'error' || !record(message) ||
      typeof message.content !== 'string' || (message.tool_calls !== undefined &&
        (!Array.isArray(message.tool_calls) || message.tool_calls.length !== 0)) ||
      !record(capture) || !['natlang.native-rl-sampling/4', 'natlang.native-structural-capture/2'].includes(String(capture.schema)) ||
      capture.finish_reason !== 'error' || !record(capture.protocol_failure) || canonical(capture.protocol_failure) !== canonical(failure) ||
      !Array.isArray(capture.samples) || !Array.isArray(capture.generated_inputs) || !Array.isArray(capture.prompt) ||
      !record(capture.policy) || ['checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture'].some(key =>
        typeof (capture.policy as RecordValue)[key] !== 'string') ||
      capture.generated_inputs.length !== failure.generated_input_count)
    throw new Error('neuralese-native-protocol-integrity: malformed native failed-turn receipt');
  const inputs = capture.generated_inputs as RecordValue[], samples = capture.samples as RecordValue[];
  let prompt = 0;
  for (const segment of capture.prompt) {
    if (!record(segment)) throw new Error('neuralese-native-protocol-integrity: malformed prompt receipt');
    if (segment.kind === 'tokens') {
      if (!Array.isArray(segment.token_ids) || segment.token_ids.some(value => !integer(value)))
        throw new Error('neuralese-native-protocol-integrity: malformed prompt tokens');
      prompt += segment.token_ids.length;
    } else if (segment.kind !== 'native_state') throw new Error('neuralese-native-protocol-integrity: unknown prompt segment');
  }
  if (prompt + inputs.length !== failure.own_input_count || inputs.some(value => !record(value)) ||
      samples.some((value, index) => !record(value) || value.index !== index || !integer(value.token_id) ||
        typeof value.terminal !== 'boolean' || value.terminal !== (failure.stop_token_ids as number[]).includes(value.token_id as number)) ||
      (failure.trigger === 'sample' ? !integer(failure.sample_index) || failure.sample_index !== samples.length - 1 ||
        inputs.some(value => value.sample_index === failure.sample_index) !== failure.fed :
        failure.trigger !== 'length' || failure.sample_index !== null || failure.fed ||
        failure.code !== 'neuralese-native-state-open' || inputs.length < failure.max_tokens))
    throw new Error('neuralese-native-protocol-integrity: failed turn changed its sampled/fed boundary');
  return structuredClone(failure) as NativeProtocolFailure;
}

export function nativeProtocolFeedback(failure: NativeProtocolFailure): string {
  return `${failure.code}: Return a complete native value or the declared ordinary result; this reply has an incomplete or misplaced native-state delimiter.`;
}
