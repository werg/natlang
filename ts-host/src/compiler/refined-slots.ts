/** The refined slots (`Is<T, "predicate">`) of a function signature, for `natlang check` (plans/REFINEMENT_TYPES.md section 6). */
import { TypeEnv, parseType, refinementChain, type Type } from '../native/types.js';

export type RefinedSlot = { source: string; function: string; slot: string; predicate: string };

/** Every predicate a value of `type` must satisfy, by slot path (`return`, `body`, `return.items[].quote`). */
export function refinedPredicates(type: Type, env: TypeEnv, path: string, seen = new Set<string>()): { slot: string; predicate: string }[] {
  const out: { slot: string; predicate: string }[] = [];
  const walk = (current: Type, at: string, depth: number): void => {
    if (depth > 24) return;
    if (current.kind === 'name') {
      const key = `${at}\0${current.name}`;
      const found = env.lookup(current.name);
      if (!found || seen.has(key)) return;
      seen.add(key);
      return walk(found, at, depth + 1);
    }
    switch (current.kind) {
      case 'refined': {
        const chain = refinementChain(current, env);
        for (const predicate of chain.predicates) out.push({ slot: at, predicate });
        return walk(chain.base, at, depth + 1);
      }
      case 'untrusted': return walk(current.base, at, depth + 1);
      case 'list': return walk(current.element, `${at}[]`, depth + 1);
      case 'dict': return walk(current.element, `${at}{}`, depth + 1);
      case 'record': for (const field of current.fields) walk(field.type, `${at}.${field.name}`, depth + 1); return;
      case 'union': for (const member of current.members) walk(member, at, depth + 1); return;
      default: return;
    }
  };
  walk(type, path, 0);
  return out;
}

/** The refined slots of a function given its argument and result type text and its type aliases. */
export function refinedSlotsOf(source: string, name: string, args: Record<string, string>, returns: string,
  types: Record<string, string>): RefinedSlot[] {
  try {
    const env = new TypeEnv(Object.fromEntries(Object.entries(types).map(([alias, text]) => [alias, parseType(text)])));
    const slots = [...Object.entries(args).map(([raw, text]) => ({ path: raw.replace(/\?$/, ''), text })), { path: 'return', text: returns }]
      .flatMap(({ path, text }) => refinedPredicates(parseType(text), env, path));
    return slots.map(slot => ({ source, function: name, ...slot }));
  } catch { return []; }
}
