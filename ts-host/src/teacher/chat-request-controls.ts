/** Collection-specific OpenAI-compatible wire settings. No messages, tools or credentials. */
export function chatRequestControls(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('chat request controls must be an object');
  const controls = value as Record<string, unknown>;
  const allowed = ['max_tokens', 'top_p', 'top_k', 'min_p', 'chat_template_kwargs', 'repetition_penalty'];
  if (Object.keys(controls).some(key => !allowed.includes(key))) throw new Error('unsupported chat request control');
  if (controls.max_tokens !== undefined && (!Number.isSafeInteger(controls.max_tokens) || Number(controls.max_tokens) < 1))
    throw new Error('max_tokens must be a positive integer');
  const serialized = JSON.stringify(controls);
  if (/"(?:apiKey|api_key|authorization|headers|messages|tools|endpoint)"\s*:/i.test(serialized)) throw new Error('credentials and conversation content are forbidden in chat controls');
  return JSON.parse(serialized);
}
