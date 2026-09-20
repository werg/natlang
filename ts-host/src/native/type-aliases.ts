/** Read the supported `type Name = ...;` declarations without truncating record fields. */
export function readTypeAliases(source: string): Record<string, string> {
  // Preserve quoted literals while removing comments before finding declaration headers.
  const clean = source.replace(/"(?:[^"\\]|\\.)*"|\/\/[^\n]*|\/\*[\s\S]*?\*\//g,
    value => value.startsWith('/') ? ' '.repeat(value.length) : value);
  const header = /\b(?:export\s+)?type\s+([A-Za-z_]\w*)\s*=\s*/g;
  const result: Record<string, string> = {};
  let match: RegExpExecArray | null;
  while ((match = header.exec(clean))) {
    const start = header.lastIndex;
    let depth = 0, quoted = false, escaped = false, end = start;
    for (; end < clean.length; end++) {
      const char = clean[end]!;
      if (quoted) {
        if (escaped) escaped = false;
        else if (char === '\\') escaped = true;
        else if (char === '"') quoted = false;
      } else if (char === '"') quoted = true;
      else if ('{[(<'.includes(char)) depth++;
      else if ('}])>'.includes(char)) depth--;
      else if (char === ';' && depth === 0) break;
      if (depth < 0) throw new Error(`unbalanced type alias ${match[1]}`);
    }
    if (end === clean.length || quoted || depth !== 0)
      throw new Error(`unterminated type alias ${match[1]}`);
    const name = match[1]!;
    if (Object.hasOwn(result, name)) throw new Error(`duplicate type alias ${name}`);
    result[name] = clean.slice(start, end).trim();
    header.lastIndex = end + 1;
  }
  return result;
}
