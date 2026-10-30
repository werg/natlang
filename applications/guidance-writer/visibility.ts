/**
 * Which facts a consumer may see at all: an exact data rule, not a preference, so it is crisp. The runtime filters by it
 * before `selectGuidanceFacts` runs. `rl/privileged_context.applicable` (training/neuralese/natlang_neuralese) is the
 * verifier of the same rule: the engine re-checks every guidance request with it and the learner re-checks every
 * captured block, so a change here is a change there.
 *
 * - `child_value`: only a child call, and only when every `match` string occurs in its prompt text, or, when the
 *   reference names the inputs it is about (identifier match strings: a quoted value such as `"M1"` or a token with a
 *   digit such as `d81028_evidence_94_0`), when every such identifier occurs: a policy's own child call is matched by
 *   what it is called on, not by the reference's wording. Several children that match with different answers give
 *   none (crispSelection). An answer value applies only when it fits the child's declared result type (`answerFits`).
 * - `attempt_feedback`, `fit_hindsight`: a call-specific fact (`invocation` set) only for that call; otherwise for any.
 * - every other kind: the root call and learned-optimizer bindings.
 */
import type { GuidanceConsumer, GuidanceFact } from './types.js';

// ---------------------------------------------------------------- child answer types
// A reference child's verified answer may guide a child call only when it fits that call's declared result type, so a
// policy's yes/no child is never pulled toward a catalog key. Mirrors rl/privileged_context.answer_fits: a small reader
// of the TypeScript type text natlang prints in a call's opening ("You are inside this call: f(...): T" and
// "type X = ..." lines). Anything it does not read, an untyped (unknown/any) result included, admits no answer.

export const ANSWER_PREFIX = 'The verified answer of this call is ';
type TypeNode = { kind: 'prim'; name: string } | { kind: 'literal'; value: string | number | boolean }
  | { kind: 'union'; options: TypeNode[] } | { kind: 'array'; item: TypeNode } | { kind: 'tuple'; items: TypeNode[] }
  | { kind: 'record'; value: TypeNode } | { kind: 'object'; fields: Map<string, { type: TypeNode; optional: boolean }> }
  | { kind: 'alias'; name: string };
type Token = ['str', string] | ['num', number] | ['name', string] | ['p', string];
const PRIMS = new Set(['string', 'number', 'boolean', 'null', 'undefined', 'unknown', 'any']);

function tokens(text: string): Token[] | null {
  const out: Token[] = [];
  let consumed = 0;
  for (const found of text.matchAll(/\s*(?:("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(-?\d+(?:\.\d+)?)|([A-Za-z_$][\w$]*)|(\S))/g)) {
    if (found.index !== consumed) return null;
    consumed += found[0].length;
    const [, string, number, name, punct] = found;
    if (string === undefined && number === undefined && name === undefined && punct === undefined) continue;
    out.push(string !== undefined ? ['str', string.startsWith('"') ? JSON.parse(string) as string : string.slice(1, -1)]
      : number !== undefined ? ['num', Number(number)] : name !== undefined ? ['name', name] : ['p', punct!]);
  }
  return text.slice(consumed).trim() ? null : out;
}

export function parseTypeText(text: string): TypeNode | null {
  const list = tokens(text);
  if (!list) return null;
  let at = 0;
  const peek = (value?: string) => {
    const token = list[at];
    return value === undefined ? token : token && token[0] === 'p' && token[1] === value ? token : undefined;
  };
  const take = (value: string) => { if (!peek(value)) throw new Error(value); at++; };
  const union = (): TypeNode => {
    if (peek('|')) at++;
    const options = [postfix()];
    for (let guard = 0; guard < list.length && peek('|'); guard++) { at++; options.push(postfix()); }
    return options.length === 1 ? options[0]! : { kind: 'union', options };
  };
  const postfix = (): TypeNode => {
    let node = primary();
    const arraySuffix = () => !!peek('[') && list[at + 1]?.[0] === 'p' && list[at + 1]?.[1] === ']';
    for (let guard = 0; guard < list.length && arraySuffix(); guard++) {
      at += 2; node = { kind: 'array', item: node };
    }
    return node;
  };
  const primary = (): TypeNode => {
    const token = peek();
    if (!token) throw new Error('end');
    at++;
    const [kind, value] = token;
    if (kind === 'str' || kind === 'num') return { kind: 'literal', value };
    if (kind === 'p' && value === '(') { const node = union(); take(')'); return node; }
    if (kind === 'p' && value === '[') {
      const items: TypeNode[] = [];
      for (let guard = 0; guard < list.length && !peek(']'); guard++) { items.push(union()); if (peek(',')) at++; }
      take(']'); return { kind: 'tuple', items };
    }
    if (kind === 'p' && value === '{') {
      const fields = new Map<string, { type: TypeNode; optional: boolean }>();
      for (let guard = 0; guard < list.length && !peek('}'); guard++) {
        const name = peek();
        if (!name || (name[0] !== 'name' && name[0] !== 'str')) throw new Error('field');
        at++;
        const optional = !!peek('?');
        if (optional) at++;
        take(':');
        fields.set(String(name[1]), { type: union(), optional });
        if (peek(';') || peek(',')) at++;
      }
      take('}'); return { kind: 'object', fields };
    }
    if (kind === 'name') {
      if (value === 'true' || value === 'false') return { kind: 'literal', value: value === 'true' };
      if (PRIMS.has(value)) return { kind: 'prim', name: value };
      if (peek('<')) {
        at++;
        const args = [union()];
        for (let guard = 0; guard < list.length && peek(','); guard++) { at++; args.push(union()); }
        take('>');
        if (value === 'Array' && args.length === 1) return { kind: 'array', item: args[0]! };
        if (value === 'Record' && args.length === 2) return { kind: 'record', value: args[1]! };
        if (value === 'Neuralese' && args.length === 1) return args[0]!;
        throw new Error(value);
      }
      return { kind: 'alias', name: value };
    }
    throw new Error(String(value));
  };
  try {
    const node = union();
    return at === list.length ? node : null;
  } catch { return null; }
}

const isNumber = (value: unknown) => typeof value === 'number';
function fits(value: unknown, node: TypeNode, aliases: ReadonlyMap<string, TypeNode>, depth = 0): boolean {
  if (depth > 32) return false;
  switch (node.kind) {
    case 'prim':
      return node.name === 'unknown' || node.name === 'any' || (node.name === 'string' && typeof value === 'string') ||
        (node.name === 'number' && isNumber(value)) || (node.name === 'boolean' && typeof value === 'boolean') ||
        ((node.name === 'null' || node.name === 'undefined') && value === null);
    case 'literal': return typeof node.value === 'number' ? isNumber(value) && value === node.value : value === node.value;
    case 'union': return node.options.some(option => fits(value, option, aliases, depth + 1));
    case 'array': return Array.isArray(value) && value.every(item => fits(item, node.item, aliases, depth + 1));
    case 'tuple': return Array.isArray(value) && value.length === node.items.length &&
      value.every((item, index) => fits(item, node.items[index]!, aliases, depth + 1));
    case 'record': return !!value && typeof value === 'object' && !Array.isArray(value) &&
      Object.values(value).every(item => fits(item, node.value, aliases, depth + 1));
    case 'object': {
      if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
      const record = value as Record<string, unknown>;
      if (Object.keys(record).some(key => !node.fields.has(key))) return false;
      return [...node.fields].every(([name, field]) => name in record ? fits(record[name], field.type, aliases, depth + 1) : field.optional);
    }
    case 'alias': { const target = aliases.get(node.name); return !!target && fits(value, target, aliases, depth + 1); }
  }
}

/** The result type and type aliases a call's opening text declares; result null when not stated or not read. */
export function declaredResult(text: string): { result: TypeNode | null; aliases: Map<string, TypeNode> } {
  const aliases = new Map<string, TypeNode>();
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    const found = /^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*=\s*(.+?);?\s*$/.exec(line);
    const parsed = found ? parseTypeText(found[2]!) : null;
    if (found && parsed) aliases.set(found[1]!, parsed);
  }
  const opening = lines.find(line => line.startsWith('You are inside this call: '));
  if (opening) {
    const signature = opening.slice('You are inside this call: '.length);
    let depth = 0;
    for (let index = 0; index < signature.length; index++) {
      const char = signature[index]!;
      if ('([{<'.includes(char)) depth++;
      else if (')]}>'.includes(char)) {
        depth--;
        if (depth === 0 && char === ')' && signature.slice(index + 1, index + 3) === ': ')
          return { result: parseTypeText(signature.slice(index + 3).trim()), aliases };
      }
    }
  }
  return { result: null, aliases: new Map() };
}

/** Whether a child_value fact may guide this child: a status outcome (no answer value) always may; an answer value
 * only when the child declares a read result type that is not unknown/any and the value fits it. */
export function answerFits(factText: string, childText: string): boolean {
  const first = factText.split('\n', 1)[0]!;
  if (!first.startsWith(ANSWER_PREFIX)) return true;
  let value: unknown;
  try { value = JSON.parse(first.slice(ANSWER_PREFIX.length).replace(/\.$/, '')); } catch { return false; }
  const { result, aliases } = declaredResult(childText);
  if (!result || (result.kind === 'prim' && (result.name === 'unknown' || result.name === 'any'))) return false;
  return fits(value, result, aliases);
}

/** A match string naming an input value rather than wording: a quoted value or a token containing a digit. */
export const identifierMatch = (item: string): boolean => /^"[^"\s]+"$/.test(item) || (/^[A-Za-z0-9][\w.:-]*$/.test(item) && /\d/.test(item));

/** Whether a child reference's match strings select this prompt: all of them, or all of its identifier strings. */
export function childMatches(match: readonly string[], text: string): boolean {
  if (!match.length) return false;
  if (match.every(item => text.includes(item))) return true;
  const ids = match.filter(identifierMatch);
  return ids.length > 0 && ids.every(item => text.includes(item));
}

export function visible(fact: GuidanceFact, consumer: GuidanceConsumer): boolean {
  if (fact.kind === 'child_value')
    return consumer.role === 'child' && childMatches(fact.match ?? [], consumer.text) && answerFits(fact.text, consumer.text);
  if (fact.kind === 'attempt_feedback' || fact.kind === 'fit_hindsight')
    return fact.invocation === undefined || fact.invocation === null || fact.invocation === consumer.name;
  return consumer.role === 'root' || consumer.role === 'binding';
}

/** The facts `consumer` may see, in their given order. */
export function visibleFacts(consumer: GuidanceConsumer, facts: readonly GuidanceFact[]): GuidanceFact[] {
  return facts.filter(fact => visible(fact, consumer));
}
