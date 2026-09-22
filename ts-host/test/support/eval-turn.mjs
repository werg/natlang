export function evalTurn(request, code, completion_tokens = 1) {
  const message = [...(request.messages ?? [])].reverse()
    .find(item => item.role === 'user' && typeof item.content === 'string');
  const program = message?.content?.split('Program:\n')[1]?.split('\nScope:')[0] ?? '';
  const lines = [...program.matchAll(/^\s*(\d+) \[ \]/gm)].map(match => Number(match[1]));
  const end = Math.max(1, ...lines);
  return { calls: [['eval', { code }], ['mark_lines', { start: 1, end }]], completion_tokens };
}
