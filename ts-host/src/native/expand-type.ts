import { type Type, type TypeEnv } from './types.js';

/** A type expanded for an editor: aliases inlined (named by `alias`), record fields with their `types.ts` doc comments,
 * and a recursive or unknown name left as `{ kind: 'name' }` for the general value editor. */
export type ExpandedType = Type & { alias?: string };

/** `type` with aliases inlined for an editor; a name seen again on its own path (recursion) stays a name. */
export function expandType(type: Type, env: TypeEnv, docs: Record<string, Record<string, string>> = {}, path: string[] = []): ExpandedType {
  if (type.kind === 'name') {
    const found = env.lookup(type.name);
    if (!found || path.includes(type.name) || path.length > 12) return type;
    const inner = expandType(found, env, docs, [...path, type.name]);
    if (inner.kind === 'record' && docs[type.name])
      return { ...inner, alias: type.name, fields: inner.fields.map(field => docs[type.name]![field.name] ? { ...field, doc: docs[type.name]![field.name] } : field) } as ExpandedType;
    return { ...inner, alias: inner.alias ?? type.name };
  }
  const child = (inner: Type) => expandType(inner, env, docs, path);
  switch (type.kind) {
    case 'record': return { ...type, fields: type.fields.map(field => ({ ...field, type: child(field.type) })) };
    case 'list': case 'dict': return { ...type, element: child(type.element) };
    case 'union': return { ...type, members: type.members.map(child) };
    case 'refined': case 'untrusted': return { ...type, base: child(type.base) };
    case 'intersection': case 'index': { try { return child(env.resolve(type)); } catch { return type; } }
    default: return type;
  }
}
