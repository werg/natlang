/**
 * Representation-generic results (plans/neuralese/DECISIONS.md 2026-10-09, "representation chosen by use").
 *
 * A natlang function may declare its result as a type parameter constrained to a crisp type and its soft form,
 * `generic: { R: string | Neuralese<string> }` with `returns: R`. Each call site instantiates `R` from the type its
 * consumer expects; the compiler records that choice on the call, and the runtime runs one body at the chosen
 * representation: the crisp instance as ordinary execution, the Neuralese instance as a template write
 * (DECISIONS.md 44). A function whose result is declared without a type parameter keeps that one representation.
 */
import { DEFAULT_DIALECT, formatType, parseType, type Type } from './types.js';

/** Which instance of a representation-generic result a call runs. `dialect` absent: the declared (or default) one. */
export type Representation = { kind: 'crisp' } | { kind: 'neuralese'; dialect?: string };
export const CRISP: Representation = Object.freeze({ kind: 'crisp' });

/** A declared representation-generic result: `name` stands for `crisp` or `neuralese` (= `Neuralese<crisp, dialect>`). */
export type GenericResult = { name: string; constraint: string; crisp: string; neuralese: string; dialect: string };

const IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;
const example = (name: string) => `generic: { ${name}: string | Neuralese<string> } with returns: ${name}`;

/**
 * The generic result `generic` declares for `returns`; undefined when the function declares none. Throws an Error
 * whose message says what to write instead.
 */
export function genericResult(generic: Record<string, string> | undefined, returns: string): GenericResult | undefined {
  if (!generic) return undefined;
  const entries = Object.entries(generic);
  if (entries.length !== 1)
    throw new Error(`generic declares the one type parameter a result is generic in, as ${example('R')}`);
  const [name, constraint] = entries[0]!;
  if (!IDENTIFIER.test(name) || name.endsWith('?')) throw new Error(`generic: ${JSON.stringify(name)} is not a type parameter name; name it R`);
  if (returns.trim() !== name) throw new Error(`generic declares ${name} for the result, so the result is returns: ${name}`);
  let type: Type;
  try { type = parseType(constraint); } catch (error) { throw new Error(`generic: ${name}: ${(error as Error).message}`); }
  const members = type.kind === 'union' ? type.members : [type];
  const soft = members.filter(member => member.kind === 'neuralese') as Extract<Type, { kind: 'neuralese' }>[];
  const crisp = members.filter(member => member.kind !== 'neuralese');
  const crispText = crisp.length ? formatType(crisp.length === 1 ? crisp[0]! : { kind: 'union', members: crisp }) : '';
  if (soft.length !== 1 || !crisp.length || formatType(soft[0]!.element) !== crispText)
    throw new Error(`generic: ${name} is a crisp type and its Neuralese form, as ${name}: ${crispText || 'string'} | ` +
      `Neuralese<${crispText || 'string'}>; each call then runs the one its result is used as`);
  return { name, constraint: formatType(type), crisp: crispText, neuralese: formatType(soft[0]!), dialect: soft[0]!.dialect };
}

/** How a representation reads in messages: `string`, `Neuralese<string>`, `Neuralese<string, "nd:x@1">`. */
export function representationType(crisp: string, representation: Representation): string {
  if (representation.kind === 'crisp') return crisp;
  return representation.dialect && representation.dialect !== DEFAULT_DIALECT ?
    `Neuralese<${crisp}, ${JSON.stringify(representation.dialect)}>` : `Neuralese<${crisp}>`;
}

/** A representation given by a host (`invokeAt`), checked. */
export function checkRepresentation(value: unknown): Representation {
  const record = value as { kind?: unknown; dialect?: unknown } | null;
  if (record && typeof record === 'object' && record.kind === 'crisp' && Object.keys(record).length === 1) return CRISP;
  if (record && typeof record === 'object' && record.kind === 'neuralese' &&
      Object.keys(record).every(key => key === 'kind' || key === 'dialect') &&
      (record.dialect === undefined || (typeof record.dialect === 'string' && record.dialect)))
    return record.dialect === undefined ? { kind: 'neuralese' } : { kind: 'neuralese', dialect: record.dialect as string };
  throw new TypeError(`a representation is { kind: "crisp" } or { kind: "neuralese", dialect?: string }, got ${JSON.stringify(value)}`);
}
