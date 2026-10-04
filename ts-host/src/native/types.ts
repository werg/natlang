/** Native port of natlang's structural type grammar and fit relation. */
export type Type =
  | { kind: 'prim'; name: 'string' | 'number' | 'boolean' | 'null' | 'Blob' | 'Folder' | 'FileHandle' | 'unknown' }
  | { kind: 'lit'; value: string | number }
  | { kind: 'record'; fields: { name: string; type: Type; optional: boolean }[] }
  | { kind: 'list'; element: Type }
  | { kind: 'dict'; element: Type }
  | { kind: 'union'; members: Type[] }
  | { kind: 'name'; name: string }
  | { kind: 'lambda'; params: Extract<Type, { kind: 'record' }>; returns: Type }
  /** A live host object checked by contract rather than copied as data. `name` is its TypeScript text. */
  | { kind: 'host'; name: string; contract: HostCheck }
  /** A soft value of `element` in a Neuralese dialect: an opaque reference to a stored block of vectors. */
  | { kind: 'neuralese'; element: Type; dialect: string };

/** The dialect a `Neuralese<T>` without a second argument names; a program's configuration binds it. */
export const DEFAULT_DIALECT = 'DefaultDialect';

/** `Adapter`: a weight adapter value (a stored coefficient block whose dialect is its spec; model/tiny_adapters.py).
 * It is a soft value of a marker element, so it fits `Adapter` and `Neuralese<unknown>` slots and no other. */
export const ADAPTER_TYPE: Type = { kind: 'neuralese', dialect: DEFAULT_DIALECT,
  element: { kind: 'record', fields: [{ name: '$adapter', type: { kind: 'lit', value: 'adapter' }, optional: false }] } };
const isAdapterType = (type: Type) => type.kind === 'neuralese' && type.dialect === DEFAULT_DIALECT &&
  type.element.kind === 'record' && type.element.fields.length === 1 && type.element.fields[0]!.name === '$adapter';

/** Runtime check for a live host value. */
export type HostCheck =
  | { kind: 'tag'; tag: string }
  | { kind: 'class'; name: string }
  | { kind: 'shape'; members: string[] }
  | { kind: 'function' }
  | { kind: 'any' };

const BUILTIN_HOST_TAGS = new Set(['Date', 'Map', 'Set', 'WeakMap', 'WeakSet', 'RegExp', 'Error', 'URL',
  'URLSearchParams', 'Uint8Array', 'ArrayBuffer', 'Headers', 'Response', 'Request']);

/** Check a live value against a host contract. Built-ins use their internal tag, which works across realms. */
export function checkHost(value: unknown, contract: HostCheck, classes?: ReadonlyMap<string, Function>): boolean {
  if (contract.kind === 'any') return value !== undefined;
  if (contract.kind === 'function') return typeof value === 'function';
  if (value === null || (typeof value !== 'object' && typeof value !== 'function')) return false;
  if (contract.kind === 'tag') return Object.prototype.toString.call(value) === `[object ${contract.tag}]`;
  if (contract.kind === 'class') {
    const constructor = classes?.get(contract.name);
    return constructor ? value instanceof (constructor as new (...args: never[]) => unknown) :
      (value as object).constructor?.name === contract.name;
  }
  return contract.members.every(member => member in (value as object));
}

export class TypeSyntaxError extends Error {}
type Token = { kind: 'str' | 'num' | 'id' | 'p'; value: string };
const TOKEN = /\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|(-?\d+(?:\.\d+)?)|([A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*)|(\[\]|=>)|([{}<>|,;:?()]))/y;
const PRIMS = new Set(['string', 'number', 'boolean', 'null', 'Blob', 'Folder', 'FileHandle', 'unknown']);

function tokenize(source: string): Token[] {
  const text = source.trim(), result: Token[] = [];
  let position = 0;
  while (position < text.length) {
    TOKEN.lastIndex = position;
    const match = TOKEN.exec(text);
    if (!match) throw new TypeSyntaxError(`bad character at ${position}: ${JSON.stringify(text.slice(position, position + 12))}`);
    // TypeScript spells string literals with either quote; a single-quoted one is the same literal.
    const single = match[2]?.replace(/\\'/g, "'");
    const kind = match[1] !== undefined || single !== undefined ? 'str' : match[3] !== undefined ? 'num' :
      match[4] !== undefined ? 'id' : 'p';
    // Double-quoted names/literals use JSON escaping, matching generated type
    // signatures and compiler targets. Decode before duplicate-field checks.
    const quoted = match[1] !== undefined ? JSON.parse(`"${match[1]}"`) as string : undefined;
    result.push({ kind, value: quoted ?? single ?? match[3] ?? match[4] ?? match[5] ?? match[6]! });
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
      if (!['id', 'str'].includes(name.kind)) throw new TypeSyntaxError(`bad field name ${JSON.stringify(name.value)}`);
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
    if (token.value === '(') {
      const start = this.index;
      this.eat('(');
      const fields: Extract<Type, { kind: 'record' }>['fields'] = [];
      let callable = this.peek()?.value === ')' ||
        (this.peek()?.kind === 'id' && [':', '?'].includes(this.tokens[this.index + 1]?.value ?? ''));
      if (callable) {
        while (this.peek()?.value !== ')') {
          const name = this.eat();
          if (name.kind !== 'id') { callable = false; break; }
          const optional = this.peek()?.value === '?'; if (optional) this.eat('?');
          this.eat(':'); fields.push({ name: name.value, optional, type: this.union() });
          if (this.peek()?.value === ',') this.eat(','); else break;
        }
        if (callable && this.peek()?.value === ')') {
          this.eat(')');
          if (this.peek()?.value === '=>') {
            this.eat('=>');
            return { kind: 'lambda', params: { kind: 'record', fields }, returns: this.union() };
          }
        }
      }
      this.index = start; this.eat('('); const inner = this.union(); this.eat(')'); return inner;
    }
    if (token.value === '{') return this.record();
    this.eat();
    if (token.kind === 'str') return { kind: 'lit', value: token.value };
    if (token.kind === 'num') return { kind: 'lit', value: Number(token.value) };
    if (token.kind !== 'id') throw new TypeSyntaxError(`unexpected ${JSON.stringify(token.value)}`);
    if (PRIMS.has(token.value)) return { kind: 'prim', name: token.value as Extract<Type, { kind: 'prim' }>['name'] };
    if (token.value === 'Record') {
      const [key, value] = this.args(2);
      if (key?.kind !== 'prim' || key.name !== 'string')
        throw new TypeSyntaxError('Record keys must be string');
      return { kind: 'dict', element: value! };
    }
    if (token.value === 'Adapter') return ADAPTER_TYPE;
    if (token.value === 'Neuralese' && this.peek()?.value === '<') {
      // Neuralese<T> or Neuralese<T, D>; D is a dialect name or string literal.
      this.eat('<');
      const element = this.union();
      let dialect = DEFAULT_DIALECT;
      if (this.peek()?.value === ',') {
        this.eat(',');
        const item = this.eat();
        if (item.kind !== 'str' && item.kind !== 'id') throw new TypeSyntaxError('Neuralese<T, D> takes a dialect name or string');
        dialect = item.value;
      }
      this.eat('>');
      if (element.kind === 'neuralese') throw new TypeSyntaxError('neuralese-nested: Neuralese<Neuralese<T>> is not a type');
      return { kind: 'neuralese', element, dialect };
    }
    if (token.value === 'Live' && this.peek()?.value === '<') {
      // Live<"TypeScript text", "tag" | "class" | "shape" | "function" | "any", "detail">
      this.eat('<');
      const parts: string[] = [];
      while (true) {
        const item = this.eat();
        if (item.kind !== 'str') throw new TypeSyntaxError('Live<...> takes string arguments');
        parts.push(item.value);
        if (this.peek()?.value === ',') { this.eat(','); continue; }
        break;
      }
      this.eat('>');
      const [name = 'object', kind = 'any', detail = ''] = parts;
      const contract: HostCheck = kind === 'tag' ? { kind, tag: detail } : kind === 'class' ? { kind, name: detail } :
        kind === 'shape' ? { kind, members: detail ? detail.split(',') : [] } : kind === 'function' ? { kind } : { kind: 'any' };
      return { kind: 'host', name, contract };
    }
    if (BUILTIN_HOST_TAGS.has(token.value)) {
      let text = token.value;
      if (this.peek()?.value === '<') {
        // Type arguments of a built-in are displayed but not checked element by element.
        const start = this.index;
        let depth = 0;
        do {
          const item = this.eat();
          if (item.value === '<') depth++; else if (item.value === '>') depth--;
        } while (depth > 0);
        text += this.tokens.slice(start, this.index).map(item => item.kind === 'str' ? JSON.stringify(item.value) : item.value)
          .join('').replace(/,/g, ', ');
      }
      return { kind: 'host', name: text, contract: { kind: 'tag', tag: token.value } };
    }
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
    case 'lit': return typeof type.value === 'string' ? JSON.stringify(type.value) : String(type.value);
    case 'record': return type.fields.length ? `{ ${type.fields.map(f => `${/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(f.name) ? f.name : JSON.stringify(f.name)}${f.optional ? '?' : ''}: ${formatType(f.type)}`).join(', ')} }` : '{}';
    case 'list': return (type.element.kind === 'union' ? `(${formatType(type.element)})` : formatType(type.element)) + '[]';
    case 'dict': return `Record<string, ${formatType(type.element)}>`;
    case 'union': return type.members.map(formatType).join(' | ');
    case 'lambda': return `(${type.params.fields.map(field => `${field.name}${field.optional ? '?' : ''}: ` +
      formatType(field.type)).join(', ')}) => ${formatType(type.returns)}`;
    case 'host': return type.name;
    case 'neuralese': return isAdapterType(type) ? 'Adapter' : `Neuralese<${formatType(type.element)}${type.dialect === DEFAULT_DIALECT ? '' :
      `, ${JSON.stringify(type.dialect)}`}>`;
  }
}

/** Child types of a type, for structural walks. */
function children(type: Type): Type[] {
  switch (type.kind) {
    case 'record': return type.fields.map(field => field.type);
    case 'list': case 'dict': case 'neuralese': return [type.element];
    case 'union': return type.members;
    case 'lambda': return [type.params, type.returns];
    default: return [];
  }
}

export class TypeEnv {
  /** Constructors used to check `class` host contracts; inherited by child environments. */
  classes?: ReadonlyMap<string, Function>;
  constructor(readonly names: Record<string, Type> = {}, readonly parent?: TypeEnv) {
    this.classes = parent?.classes;
    for (const name of Object.keys(names)) this.checkRecursiveFunction(name);
  }
  /**
   * Reject an alias that reaches itself through a function type (`type-recursive-function`): self-application
   * would make recursion expressible without a recursive definition. Recursive data types stay allowed.
   */
  private checkRecursiveFunction(alias: string): void {
    const visiting = new Set<string>();
    const walk = (type: Type, throughFunction: boolean): void => {
      if (type.kind === 'name') {
        if (type.name === alias && throughFunction)
          throw new TypeSyntaxError(`type-recursive-function: type ${alias} refers to itself through a function type`);
        const key = `${type.name}\0${throughFunction}`;
        if (visiting.has(key)) return;
        visiting.add(key);
        const found = this.lookup(type.name);
        if (found) walk(found, throughFunction);
        return;
      }
      for (const child of children(type)) walk(child, throughFunction || type.kind === 'lambda');
    };
    const own = this.names[alias];
    if (own) walk(own, false);
  }
  child(names: Record<string, Type>): TypeEnv { return Object.keys(names).length ? new TypeEnv({ ...names }, this) : this; }
  lookup(name: string): Type | undefined { return this.names[name] ?? this.parent?.lookup(name); }
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
    if (type.kind === 'neuralese') {
      this.checkNames(type.element);
      // A view of a view has no meaning distinct from a view.
      if (this.resolveQuiet(type.element)?.kind === 'neuralese')
        throw new TypeSyntaxError(`neuralese-nested: ${formatType(type)} nests a Neuralese type`);
    }
  }
  private resolveQuiet(type: Type): Type | undefined { try { return this.resolve(type); } catch { return undefined; } }
}


export function fitsType(source: Type, target: Type, env = new TypeEnv(), seen = new Set<string>()): boolean {
  if (target.kind === 'prim' && target.name === 'unknown') return true;
  if (source.kind === 'name' && target.kind === 'name') {
    const key = `${source.name}\0${target.name}`;
    if (source.name === target.name || seen.has(key)) return true;
    seen.add(key);
  }
  const a = env.resolve(source), b = env.resolve(target);
  if (JSON.stringify(a) === JSON.stringify(b)) return true;
  if (a.kind === 'union') return a.members.every(member => fitsType(member, b, env, seen));
  if (b.kind === 'union') return b.members.some(member => fitsType(a, member, env, seen));
  if (a.kind === 'lit' && b.kind === 'prim') return b.name === (typeof a.value === 'string' ? 'string' : 'number');
  if ((a.kind === 'list' || a.kind === 'dict') && a.kind === b.kind) return fitsType(a.element, b.element, env, seen);
  if (a.kind === 'record' && b.kind === 'record') {
    for (const field of b.fields) {
      const got = a.fields.find(f => f.name === field.name);
      if (!got) { if (!field.optional) return false; continue; }
      if (!fitsType(got.type, field.type, env, seen)) return false;
    }
    return a.fields.every(field => b.fields.some(other => other.name === field.name));
  }
  if (a.kind === 'host' && b.kind === 'host')
    return b.contract.kind === 'any' || JSON.stringify(a.contract) === JSON.stringify(b.contract);
  if (b.kind === 'host' && b.contract.kind === 'any') return true;
  if (a.kind === 'neuralese' && b.kind === 'neuralese')
    return a.dialect === b.dialect && fitsType(a.element, b.element, env, seen);
  if (a.kind === 'lambda' && b.kind === 'lambda')
    return fitsType(a.returns, b.returns, env, seen) && fitsType(b.params, a.params, env, seen);
  return false;
}
