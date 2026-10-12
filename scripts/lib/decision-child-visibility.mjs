// Text the model actually received before this child decision. The native
// materializer keeps the current action in `turn.target`, separately from
// the durable opening and prior messages in `turn.messages`.
export function visibleTurnText(turn) {
  const visibleToolCallText = call => {
    const raw = call.function?.arguments ?? '';
    if (typeof raw !== 'string' || !raw) return '';
    // Tool arguments are transported as JSON. Inspect both that exact wire
    // text and decoded argument values: code passed to eval is visible to the
    // model after JSON decoding, even though quotes in the wire text are
    // escaped and would otherwise defeat exact JSON-value matching.
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
