/**
 * The edit tool's argument repair (pi `prepareEditArguments`), run by the tool task before validation: `edits` sent as
 * a JSON string or as one edit object, and a top-level oldText/newText pair, become an edits list. Works on a copy.
 */

function isSingleEditInput(value: unknown): boolean {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const edit = value as Record<string, unknown>;
  return typeof edit.oldText === 'string' && typeof edit.newText === 'string';
}

export default function prepare(input: unknown): unknown {
  if (!input || typeof input !== 'object' || Array.isArray(input)) return input;
  const args: Record<string, unknown> = { ...(input as Record<string, unknown>) };
  if (typeof args.edits === 'string') {
    try {
      const parsed: unknown = JSON.parse(args.edits);
      if (Array.isArray(parsed)) args.edits = parsed;
      else if (isSingleEditInput(parsed)) args.edits = [parsed];
    } catch { /* left as sent: validation reports it */ }
  } else if (isSingleEditInput(args.edits)) args.edits = [args.edits];
  if (typeof args.oldText !== 'string' || typeof args.newText !== 'string') return args;
  const edits = Array.isArray(args.edits) ? [...args.edits] : [];
  edits.push({ oldText: args.oldText, newText: args.newText });
  const { oldText: _oldText, newText: _newText, ...rest } = args;
  return { ...rest, edits };
}
