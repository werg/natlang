function exactRecordKeys(value, keys) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value) &&
    Object.keys(value).sort().join(',') === [...keys].sort().join(',');
}

/**
 * Match OpenCode v1.18.35's no-effect unavailable-tool result, never arbitrary
 * native tool events. At pinned commit 53d1eabb61e21162157817bf677da0a4ad3332e3,
 * session/llm.ts routes missing provider tools to tool/invalid.ts, whose executor
 * returns only `title`, `output`, and empty metadata; it performs no I/O:
 * https://github.com/anomalyco/opencode/blob/53d1eabb61e21162157817bf677da0a4ad3332e3/packages/opencode/src/session/llm.ts
 * https://github.com/anomalyco/opencode/blob/53d1eabb61e21162157817bf677da0a4ad3332e3/packages/opencode/src/tool/invalid.ts
 * The CLI may wrap that empty metadata as exactly `{ truncated: false }`.
 */
export function rejectedNatlangToolAttempt(part, allowedNames) {
  const state = part?.state;
  const input = state?.input;
  const exactHandlerMetadata = exactRecordKeys(state?.metadata, []) ||
    (exactRecordKeys(state?.metadata, ['truncated']) && state.metadata.truncated === false);
  if (part?.tool !== 'invalid' || state?.status !== 'completed' || state.title !== 'Invalid Tool' ||
      !exactRecordKeys(input, ['tool', 'error']) || typeof input.tool !== 'string' || !allowedNames.has(input.tool) ||
      typeof input.error !== 'string' || !input.error.startsWith(`Model tried to call unavailable tool '${input.tool}'.`) ||
      state.output !== `The arguments provided to the tool are invalid: ${input.error}` ||
      !exactHandlerMetadata) return undefined;
  return { rejected_tool_name: input.tool, rejection: input.error, handler: 'OpenCode InvalidTool', status: 'completed',
    protocol_record: part };
}
