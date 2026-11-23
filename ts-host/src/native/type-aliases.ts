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
    // The alias's own doc comment says what values of it mean (a data format, a unit); it is kept at the head of the
    // definition, where the type grammar reads it as a comment and the call opening shows it above the alias.
    const doc = ts.getJSDocCommentsAndTags(statement).filter(ts.isJSDoc)
      .map(item => ts.getTextOfJSDocComment(item.comment) ?? '').join(' ').replace(/\s+/g, ' ').trim();
    result[name] = (doc ? `/** ${doc.replace(/\*\//g, '* /')} */ ` : '') + statement.type.getText(file);
  }
  return result;
}

/** `type name = definition;` as a call is shown it: an alias's doc comment (kept by readTypeAliases) goes above it. */
export function aliasDeclaration(name: string, definition: string): string {
  const doc = /^(\/\*\*[^]*?\*\/)\s*/.exec(definition);
  return doc ? `${doc[1]}\ntype ${name} = ${definition.slice(doc[0].length)};` : `type ${name} = ${definition};`;
}
