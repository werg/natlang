import ts from 'typescript';

/**
 * The `type Name = ...` declarations of a TypeScript source, as type text. Generic aliases are left out:
 * natlang types have no type parameters.
 */
export function readTypeAliases(source: string): Record<string, string> {
  const file = ts.createSourceFile('types.ts', source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  const errors = (file as unknown as { parseDiagnostics?: readonly ts.Diagnostic[] }).parseDiagnostics ?? [];
  if (errors.length) throw new Error(`type alias syntax: ${errors.map(item => ts.flattenDiagnosticMessageText(item.messageText, '\n')).join('; ')}`);
  const result: Record<string, string> = {};
  for (const statement of file.statements) {
    if (!ts.isTypeAliasDeclaration(statement) || statement.typeParameters?.length) continue;
    const name = statement.name.text;
    if (Object.hasOwn(result, name)) throw new Error(`duplicate type alias ${name}`);
    result[name] = statement.type.getText(file);
  }
  return result;
}
