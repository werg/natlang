/** Scripted source references read only the exact string exposed by the existing text transport. */
export function readTextFromRenderedOpening(request) {
  const strings = [];
  const visit = value => {
    if (typeof value === 'string') strings.push(value);
    else if (Array.isArray(value)) value.forEach(visit);
    else if (value && typeof value === 'object') Object.values(value).forEach(visit);
  };
  visit(request.messages ?? []);
  const openings = new Map();
  for (const rendered of strings) {
    let search = 0;
    while (true) {
      const start = rendered.indexOf('[[Neuralese text block id=', search);
      if (start < 0) break;
      search = start + 2;
      const bodyStart = rendered.indexOf('; exact JSON string body=', start);
      if (bodyStart < 0) continue;
      const header = rendered.slice(start, bodyStart);
      if (header.slice(2).includes('[[') || !/(?:; | )type=Neuralese<string>(?:;|$)/.test(header)) continue;
      const valueStart = bodyStart + '; exact JSON string body='.length;
      for (let close = rendered.indexOf(']]', valueStart); close >= 0; close = rendered.indexOf(']]', close + 2)) {
        try {
          const body = JSON.parse(rendered.slice(valueStart, close));
          if (typeof body !== 'string') break;
          openings.set(rendered.slice(start, close + 2), body);
          search = close + 2;
          break;
        } catch { /* JSON strings can contain the display terminator. */ }
      }
    }
  }
  if (openings.size !== 1)
    throw new Error('scripted typed read requires exactly one visible exact JSON string opening');
  return [...openings.values()][0];
}

export function scriptedTextReferenceDriver(reference) {
  return async request => {
    if (JSON.stringify(request.messages ?? []).includes('This is a typed Neuralese read call.'))
      return { calls: [['return_result', { status: 'success', value: readTextFromRenderedOpening(request) }]] };
    return reference(request);
  };
}
