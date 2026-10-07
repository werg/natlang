/** Match a declared read_file call against the exact source FileHandle bytes. */
export function matchDeclaredSourceRead({ readPath, expectedPath, resultText, folderFiles }) {
  if (typeof readPath !== 'string' || typeof expectedPath !== 'string' || readPath !== expectedPath)
    return null;
  const sourceText = folderFiles?.[readPath];
  if (typeof sourceText !== 'string' || resultText !== sourceText) return null;
  return { source_path: readPath, source_text: sourceText };
}
