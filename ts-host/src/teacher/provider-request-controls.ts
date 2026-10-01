/** JSON-only transport controls, captured in teacher provenance. Credentials belong in the environment. */
export type ProviderRequestControls = {
  piOptions?: Record<string, unknown>;
  modelOptions?: Record<string, unknown>;
  piPayload?: Record<string, unknown>;
  omitPayloadKeys?: string[];
};
export function providerRequestControls(value: unknown): ProviderRequestControls {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('provider request controls must be an object');
  const input = value as Record<string, unknown>;
  const allowed = ['piOptions', 'piPayload', 'modelOptions', 'omitPayloadKeys'];
  if (Object.keys(input).some(key => !allowed.includes(key))) throw new Error('unknown provider request control');
  for (const [name, keys] of [['piOptions', ['maxTokens', 'reasoningEffort']],
    ['piPayload', ['provider', 'reasoning', 'tool_choice']],
    ['modelOptions', ['inherit', 'name', 'contextWindow', 'maxTokens', 'reasoning', 'cost']]] as const) {
    const item = input[name];
    if (item !== undefined && (!item || typeof item !== 'object' || Array.isArray(item) ||
      Object.keys(item).some(key => !(keys as readonly string[]).includes(key)))) throw new Error(`invalid ${name} controls`);
  }
  const omitted = input.omitPayloadKeys;
  if (omitted !== undefined && (!Array.isArray(omitted) || omitted.some(key => key !== 'seed'))) throw new Error('only seed may be omitted');
  const serialized = JSON.stringify(value);
  if (/"(?:apiKey|api_key|authorization|headers|messages|tools|endpoint)"\s*:/i.test(serialized)) throw new Error('credentials and request content are forbidden in transport controls');
  return JSON.parse(serialized) as ProviderRequestControls;
}
export function controlledProviderProfile(provider: string, model: string,
  piOptions?: Record<string, unknown>, controls?: ProviderRequestControls) {
  const reviewed = controls ? providerRequestControls(controls) : undefined;
  return { provider, model, piOptions: { ...reviewed?.piOptions, ...piOptions },
    ...(reviewed ? { modelOptions: reviewed.modelOptions, piPayload: { ...reviewed.piPayload,
      ...Object.fromEntries((reviewed.omitPayloadKeys ?? []).map(key => [key, undefined])) } } : {}) };
}
