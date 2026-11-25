import ts from 'typescript';

/**
 * The `type Name = ...` declarations of a TypeScript source, as type text, and its `interface Name { ... }` records (an
 * interface of properties is the record type of its body; `extends A, B` intersects A and B with it). Generic aliases
 * and interfaces are left out: natlang types have no type parameters. An interface with methods is left to a
 * callable-folder module, which reads it as a live value (runtime/loader.ts declaredTypes).
 */
export function readTypeAliases(source: string): Record<string, string> {
  const file = ts.createSourceFile('types.ts', source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  const errors = (file as unknown as { parseDiagnostics?: readonly ts.Diagnostic[] }).parseDiagnostics ?? [];
  if (errors.length) throw new Error(`type alias syntax: ${errors.map(item => ts.flattenDiagnosticMessageText(item.messageText, '\n')).join('; ')}`);
  const result: Record<string, string> = {};
  for (const statement of file.statements) {
    const body = ts.isTypeAliasDeclaration(statement) && !statement.typeParameters?.length ? statement.type.getText(file) :
      ts.isInterfaceDeclaration(statement) ? interfaceRecord(statement, file) : undefined;
    if (body === undefined) continue;
    const name = (statement as ts.TypeAliasDeclaration | ts.InterfaceDeclaration).name.text;
    if (Object.hasOwn(result, name)) throw new Error(`duplicate type alias ${name}`);
    // The alias's own doc comment says what values of it mean (a data format, a unit); it is kept at the head of the
    // definition, where the type grammar reads it as a comment and the call opening shows it above the alias.
    const doc = ts.getJSDocCommentsAndTags(statement).filter(ts.isJSDoc)
      .map(item => ts.getTextOfJSDocComment(item.comment) ?? '').join(' ').replace(/\s+/g, ' ').trim();
    result[name] = (doc ? `/** ${doc.replace(/\*\//g, '* /')} */ ` : '') + body;
  }
  return result;
}

/** `type name = definition;` as a call is shown it: an alias's doc comment (kept by readTypeAliases) goes above it. */
export function aliasDeclaration(name: string, definition: string): string {
  const doc = /^(\/\*\*[^]*?\*\/)\s*/.exec(definition);
  return doc ? `${doc[1]}\ntype ${name} = ${definition.slice(doc[0].length)};` : `type ${name} = ${definition};`;
}

/** An interface of properties as record type text (its body, with the field comments), or undefined for any other. */
function interfaceRecord(statement: ts.InterfaceDeclaration, file: ts.SourceFile): string | undefined {
  if (statement.typeParameters?.length || !statement.members.every(member => ts.isPropertySignature(member) && !!member.type &&
      (ts.isIdentifier(member.name) || ts.isStringLiteral(member.name)))) return undefined;
  const bases: string[] = [];
  for (const clause of statement.heritageClauses ?? []) for (const base of clause.types) {
    if (!ts.isIdentifier(base.expression) || base.typeArguments?.length) return undefined;
    bases.push(base.expression.text);
  }
  const text = file.text, open = statement.members.pos - 1;
  if (text[open] !== '{') return undefined;
  const record = text.slice(open, statement.end).trim();
  return [...bases, record].join(' & ');
}
