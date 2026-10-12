// Text the model actually received before this child decision. The native
// materializer keeps the current action in `turn.target`, separately from
// the durable opening and prior messages in `turn.messages`; preceding
// assistant tool calls are part of the actual conversation context.
export function visibleTurnText(turn) {
  const visibleToolCallText = call => {
    const raw = call.function?.arguments ?? '';
    if (typeof raw !== 'string' || !raw) return '';
    // Tool arguments are transported as JSON. Inspect both the exact wire
    // text and decoded argument values so captured source values embedded in
    // earlier eval code are checked as the model saw them.
    let decoded = '';
    try {
      const value = JSON.parse(raw);
      const leaves = [];
      const visit = child => {
        if (typeof child === 'string') leaves.push(child);
        else if (Array.isArray(child)) child.forEach(visit);
        else if (child && typeof child === 'object') Object.values(child).forEach(visit);
      };
      visit(value);
      decoded = leaves.join('\n');
    } catch {}
    return `${raw}\n${decoded}`;
  };
  return (turn.messages ?? []).map(message => `${message.content ?? ''}\n${(message.tool_calls ?? [])
    .map(visibleToolCallText).join('\n')}`).join('\n');
}

function compactJson(text) {
  let out = '', quoted = false, escaped = false;
  for (const char of text) {
    if (quoted) {
      out += char;
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') quoted = false;
    } else if (char === '"') {
      quoted = true;
      out += char;
    } else if (!/\s/.test(char)) out += char;
  }
  return out;
}

// Structured source fields are visible only when every value appears in the
// exact serialized structure. A compact display placeholder is insufficient.
export function visibleValue(value, text) {
  if (typeof text !== 'string') return false;
  if (typeof value === 'string') return text.includes(value) || text.includes(JSON.stringify(value));
  const serialized = JSON.stringify(value);
  return compactJson(text).includes(compactJson(serialized));
}
