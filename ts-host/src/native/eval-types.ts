/**
 * Types an eval declares for itself (`type State = …`, `interface Row { … }`), in the runtime's type language, so a
 * local annotated with one is checked and stored like any other.
 */
import ts from 'typescript';
import { parseType, type Type } from './types.js';

/**
 * The top-level type aliases and interfaces (without type parameters or heritage) that `code` declares. One the type
 * language cannot express is `unknown`: a value annotated with it is kept as it is instead of refused.
 */
export function evalTypeDeclarations(code: string): Record<string, Type> {
  const file = ts.createSourceFile('eval.ts', code, ts.ScriptTarget.ES2022, true);
  const declared: Record<string, Type> = {};
  for (const statement of file.statements) {
    let text: string | undefined;
    if (ts.isTypeAliasDeclaration(statement) && !statement.typeParameters) text = statement.type.getText(file);
    else if (ts.isInterfaceDeclaration(statement) && !statement.typeParameters && !statement.heritageClauses)
      text = `{ ${statement.members.map(member => member.getText(file).replace(/[;,]\s*$/, '')).join('; ')} }`;
    else continue;
    try { declared[statement.name.text] = parseType(text); } catch { declared[statement.name.text] = parseType('unknown'); }
  }
  return declared;
}

/** `type` with the names `declared` defines replaced by their definitions, through declarations that use each other. */
export function inlineDeclaredTypes(type: Type, declared: Record<string, Type>, seen: ReadonlySet<string> = new Set()): Type {
  const inline = (inner: Type) => inlineDeclaredTypes(inner, declared, seen);
  switch (type.kind) {
    case 'name': {
      const definition = declared[type.name];
      if (!definition || seen.has(type.name)) return type;
      return inlineDeclaredTypes(definition, declared, new Set([...seen, type.name]));
    }
    case 'record': return { ...type, fields: type.fields.map(field => ({ ...field, type: inline(field.type) })) };
    case 'list': case 'dict': return { ...type, element: inline(type.element) };
    case 'union': return { ...type, members: type.members.map(inline) };
    case 'lambda': return { ...type, params: inline(type.params) as typeof type.params, returns: inline(type.returns) };
    default: return type;
  }
}
