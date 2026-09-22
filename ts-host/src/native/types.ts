/** Native port of natlang's structural type grammar and fit relation. */
export type Type =
  | { kind: 'prim'; name: 'Text' | 'Num' | 'Bool' | 'Null' | 'Blob' | 'Folder' | 'File' }
  | { kind: 'lit'; value: string | number }
  | { kind: 'record'; fields: { name: string; type: Type; optional: boolean }[] }
  | { kind: 'list'; element: Type }
  | { kind: 'dict'; element: Type }
  | { kind: 'union'; members: Type[] }
  | { kind: 'name'; name: string }
  | { kind: 'lambda'; params: Extract<Type, { kind: 'record' }>; returns: Type }
  | { kind: 'map'; a: Type; b: Type }
  | { kind: 'fold'; a: Type; s: Type }
  | { kind: 'iterate'; s: Type };

export class TypeSyntaxError extends Error {}
type Token = { kind: 'str' | 'num' | 'id' | 'p'; value: string };
const TOKEN = /\s*(?:"((?:[^"\\]|\\.)*)"|(-?\d+(?:\.\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(\[\])|([{}<>|,;:?()]))/y;
const PRIMS = new Set(['Text', 'Num', 'Bool', 'Null', 'Blob', 'Folder', 'File']);
const pendingKinds = new Set(['lambda', 'map', 'fold', 'iterate']);

function tokenize(source: string): Token[] {
  const text = source.trim(), result: Token[] = [];
  let position = 0;
  while (position < text.length) {
    TOKEN.lastIndex = position;
    const match = TOKEN.exec(text);
    if (!match) throw new TypeSyntaxError(`bad character at ${position}: ${JSON.stringify(text.slice(position, position + 12))}`);
    const kind = match[1] !== undefined ? 'str' : match[2] !== undefined ? 'num' :
      match[3] !== undefined ? 'id' : 'p';
    result.push({ kind, value: match[1] ?? match[2] ?? match[3] ?? match[4] ?? match[5]! });
    position = TOKEN.lastIndex;
  }
  return result;
}

class Parser {
  private index = 0;
  constructor(private tokens: Token[]) {}
  private peek(): Token | undefined { return this.tokens[this.index]; }
  private eat(value?: string): Token {
    const token = this.tokens[this.index];
    if (!token || (value !== undefined && token.value !== value))
      throw new TypeSyntaxError(`expected ${JSON.stringify(value ?? 'token')}, got ${JSON.stringify(token?.value)}`);
    this.index++;
    return token;
  }
  parse(): Type {
    const result = this.union();
    if (this.peek()) throw new TypeSyntaxError(`trailing input at ${JSON.stringify(this.peek()!.value)}`);
    return result;
  }
  private union(): Type {
    const members = [this.postfix()];
    while (this.peek()?.value === '|') { this.eat('|'); members.push(this.postfix()); }
    if (members.length === 1) return members[0]!;
    return { kind: 'union', members: members.flatMap(m => m.kind === 'union' ? m.members : [m]) };
  }
  private postfix(): Type {
    let type = this.atom();
    while (this.peek()?.value === '[]') { this.eat('[]'); type = { kind: 'list', element: type }; }
    return type;
  }
  private args(count: number): Type[] {
    this.eat('<');
    const result = [this.union()];
    while (this.peek()?.value === ',') { this.eat(','); result.push(this.union()); }
    this.eat('>');
    if (result.length !== count) throw new TypeSyntaxError(`expected ${count} type arguments, got ${result.length}`);
    return result;
  }
  private record(): Extract<Type, { kind: 'record' }> {
    this.eat('{');
    const fields: Extract<Type, { kind: 'record' }>['fields'] = [];
    while (this.peek()?.value !== '}') {
      const name = this.eat();
      if (name.kind !== 'id') throw new TypeSyntaxError(`bad field name ${JSON.stringify(name.value)}`);
      const optional = this.peek()?.value === '?';
      if (optional) this.eat('?');
      this.eat(':');
      fields.push({ name: name.value, type: this.union(), optional });
      if (this.peek()?.value === ',' || this.peek()?.value === ';') this.eat();
    }
    this.eat('}');
    if (new Set(fields.map(f => f.name)).size !== fields.length) throw new TypeSyntaxError('duplicate field name');
    return { kind: 'record', fields };
  }
  private atom(): Type {
    const token = this.peek();
    if (!token) throw new TypeSyntaxError('unexpected end of type');
    if (token.value === '(') { this.eat('('); const inner = this.union(); this.eat(')'); return inner; }
    if (token.value === '{') return this.record();
    this.eat();
    if (token.kind === 'str') return { kind: 'lit', value: token.value };
    if (token.kind === 'num') return { kind: 'lit', value: Number(token.value) };
    if (token.kind !== 'id') throw new TypeSyntaxError(`unexpected ${JSON.stringify(token.value)}`);
    if (PRIMS.has(token.value)) return { kind: 'prim', name: token.value as Extract<Type, { kind: 'prim' }>['name'] };
    if (token.value === 'Dict') return { kind: 'dict', element: this.args(1)[0]! };
    if (token.value === 'Lambda') {
      const [params, returns] = this.args(2);
      if (params?.kind !== 'record') throw new TypeSyntaxError('Lambda params must be a record type');
      return { kind: 'lambda', params, returns: returns! };
    }
    if (token.value === 'Map') { const [a, b] = this.args(2); return { kind: 'map', a: a!, b: b! }; }
    if (token.value === 'Fold') { const [a, s] = this.args(2); return { kind: 'fold', a: a!, s: s! }; }
    if (token.value === 'Iterate') return { kind: 'iterate', s: this.args(1)[0]! };
    return { kind: 'name', name: token.value };
  }
}

export function parseType(text: string): Type {
  if (typeof text !== 'string') throw new TypeSyntaxError('type must be a string');
  return new Parser(tokenize(text)).parse();
}

export function formatType(type: Type): string {
  switch (type.kind) {
    case 'prim': case 'name': return type.name;
    case 'lit': return typeof type.value === 'string' ? `"${type.value}"` : String(type.value);
    case 'record': return type.fields.length ? `{ ${type.fields.map(f => `${f.name}${f.optional ? '?' : ''}: ${formatType(f.type)}`).join(', ')} }` : '{}';
    case 'list': return (type.element.kind === 'union' ? `(${formatType(type.element)})` : formatType(type.element)) + '[]';
    case 'dict': return `Dict<${formatType(type.element)}>`;
    case 'union': return type.members.map(formatType).join(' | ');
    case 'lambda': return `Lambda<${formatType(type.params)}, ${formatType(type.returns)}>`;
    case 'map': return `Map<${formatType(type.a)}, ${formatType(type.b)}>`;
    case 'fold': return `Fold<${formatType(type.a)}, ${formatType(type.s)}>`;
    case 'iterate': return `Iterate<${formatType(type.s)}>`;
  }
}

export const LOOP_VERDICT = parseType('{ reason: Text, verdict: "continue" | "done" | "degenerate" }');

export class TypeEnv {
  constructor(readonly names: Record<string, Type> = {}, readonly parent?: TypeEnv) {}
  child(names: Record<string, Type>): TypeEnv { return Object.keys(names).length ? new TypeEnv({ ...names }, this) : this; }
  lookup(name: string): Type | undefined { return this.names[name] ?? this.parent?.lookup(name) ?? (name === 'LoopVerdict' ? LOOP_VERDICT : undefined); }
  resolve(type: Type): Type {
    const seen = new Set<string>();
    while (type.kind === 'name') {
      if (seen.has(type.name)) throw new TypeSyntaxError(`type ${type.name} is defined only in terms of itself`);
      seen.add(type.name);
      const found = this.lookup(type.name);
      if (!found) throw new TypeSyntaxError(`unknown type name ${type.name}`);
      type = found;
    }
    return type;
  }
  checkNames(type: Type): void {
    if (type.kind === 'name') { if (!this.lookup(type.name)) throw new TypeSyntaxError(`unknown type name ${type.name}`); return; }
    if (type.kind === 'record') for (const field of type.fields) this.checkNames(field.type);
    if (type.kind === 'list' || type.kind === 'dict') this.checkNames(type.element);
    if (type.kind === 'union') for (const member of type.members) this.checkNames(member);
    if (type.kind === 'lambda') { this.checkNames(type.params); this.checkNames(type.returns); }
    if (type.kind === 'map') { this.checkNames(type.a); this.checkNames(type.b); }
    if (type.kind === 'fold') { this.checkNames(type.a); this.checkNames(type.s); }
    if (type.kind === 'iterate') this.checkNames(type.s);
  }
}

export function resultType(type: Type): Type {
  if (type.kind === 'lambda') return type.returns;
  if (type.kind === 'map') return { kind: 'list', element: type.b };
  if (type.kind === 'fold' || type.kind === 'iterate') return type.s;
  throw new TypeError('not a pending type');
}

export function fitsType(source: Type, target: Type, env = new TypeEnv(), seen = new Set<string>()): boolean {
  if (source.kind === 'name' && target.kind === 'name') {
    const key = `${source.name}\0${target.name}`;
    if (source.name === target.name || seen.has(key)) return true;
    seen.add(key);
  }
  const a = env.resolve(source), b = env.resolve(target);
  if (JSON.stringify(a) === JSON.stringify(b)) return true;
  if (pendingKinds.has(a.kind) && !pendingKinds.has(b.kind)) return fitsType(resultType(a), b, env, seen);
  if (a.kind === 'union') return a.members.every(member => fitsType(member, b, env, seen));
  if (b.kind === 'union') return b.members.some(member => fitsType(a, member, env, seen));
  if (a.kind === 'lit' && b.kind === 'prim') return b.name === (typeof a.value === 'string' ? 'Text' : 'Num');
  if ((a.kind === 'list' || a.kind === 'dict') && a.kind === b.kind) return fitsType(a.element, b.element, env, seen);
  if (a.kind === 'record' && b.kind === 'record') {
    for (const field of b.fields) {
      const got = a.fields.find(f => f.name === field.name);
      if (!got) { if (!field.optional) return false; continue; }
      if (!fitsType(got.type, field.type, env, seen)) return false;
    }
    return a.fields.every(field => b.fields.some(other => other.name === field.name));
  }
  if (a.kind === 'lambda' && b.kind === 'lambda')
    return fitsType(a.returns, b.returns, env, seen) && fitsType(b.params, a.params, env, seen);
  return false;
}
