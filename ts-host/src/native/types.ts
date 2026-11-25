/** Native port of natlang's structural type grammar and fit relation. */
export type Type =
  | { kind: 'prim'; name: 'string' | 'number' | 'boolean' | 'null' | 'Blob' | 'Folder' | 'FileHandle' | 'unknown' }
  | { kind: 'lit'; value: string | number }
  | { kind: 'record'; fields: { name: string; type: Type; optional: boolean }[] }
  | { kind: 'list'; element: Type }
  /** A fixed-length (optionally trailing-optional) tuple `[A, B]` or `[A, B?]`. `minLength`, when set, is the count of required leading elements. */
  | { kind: 'tuple'; elements: Type[]; minLength?: number }
  /** A dictionary from string keys. `key` / `keyType` names the key of `Record<Kind, T>`: when it resolves to string literals,
   * the type is the record with exactly those fields (TypeEnv.resolve), as in TypeScript. */
  | { kind: 'dict'; element: Type; key?: string; keyType?: Type }
  | { kind: 'union'; members: Type[] }
  /** An alias. `unparsed` marks a definition the type grammar cannot read (parseAliases): using it reports why. */
  | { kind: 'name'; name: string; unparsed?: string }
  | { kind: 'lambda'; params: Extract<Type, { kind: 'record' }>; returns: Type }
  /** A live host object checked by contract rather than copied as data. `name` is its TypeScript text. */
  | { kind: 'host'; name: string; contract: HostCheck }
  /** A soft value of `element` in a Neuralese dialect: an opaque reference to a stored block of vectors. */
  | { kind: 'neuralese'; element: Type; dialect: string }
  /** `Is<T, "predicate">`: values of `base` that a reader can judge to satisfy the natural-language `predicate`
   * (plans/REFINEMENT_TYPES.md). Structurally it is `base`; the predicate is checked where a value enters the slot. */
  | { kind: 'refined'; base: Type; predicate: string }
  /** `A & B`: values admitted by both members. Resolution distributes unions and narrows shared fields;
   * incompatible alternatives are the empty union (`never`). Record fields stay lazy for recursive aliases. */
  | { kind: 'intersection'; members: Type[] }
  /** `T['field']`: a record field, a dictionary element, or the union of the selected members. */
  | { kind: 'index'; base: Type; key: string }
  /** `Partial<R>`, `Required<R>`, `Pick<R, K>`, `Omit<R, K>`: record transformations resolved in `TypeEnv.resolve`. */
  | { kind: 'utility'; op: 'Partial' | 'Required'; target: Type }
  | { kind: 'utility'; op: 'Pick' | 'Omit'; target: Type; keys: Type }
  /** `Untrusted<T>`: a `T` that came from outside the program. Structurally `base`; the model-facing renderer always
   * shows it as a quoted data block (native/untrusted.ts) and it never enters instruction text (plans/REFINEMENT_TYPES.md). */
  | { kind: 'untrusted'; base: Type };

/** The dialect a `Neuralese<T>` without a second argument names; a program's configuration binds it. */
export const DEFAULT_DIALECT = 'DefaultDialect';

/**
 * How a runtime's type checks see dialects (DECISIONS.md 2026-10-09, "dialects are static facts"). `dialect` is the
 * runtime's reader dialect, which `DefaultDialect` stands for; null for a text-only runtime, where `DefaultDialect`
 * stays unbound. `blockDialect` is the dialect a stored block was written in, when the store knows it without I/O.
 */
export type DialectBinding = { readonly dialect: string | null; blockDialect?(id: string): string | undefined;
  nativeDefaultCompatible?(id: string, source: string, target: string): boolean };

/** `NativeAdapter` is an opaque reference to an archived nnt1 coefficient artifact. */
export const NATIVE_ADAPTER_TYPE: Type = { kind: 'host', name: 'NativeAdapter',
  contract: { kind: 'shape', members: ['$nativeAdapter'] } };
/** `NativeTensor` is an opaque ref to one archived non-coefficient nnt1 tensor artifact. */
export const NATIVE_TENSOR_TYPE: Type = { kind: 'host', name: 'NativeTensor',
  contract: { kind: 'shape', members: ['$nativeTensor'] } };

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

/** A predicate with its whitespace collapsed: two predicates are the same when their normalized texts are equal. */
export const normalizePredicate = (predicate: string): string => predicate.replace(/\s+/g, ' ').trim();
const refinementInvalid = (got: string) => `refinement-predicate-invalid: Is<T, P> takes a nonempty string literal as P, ` +
  `for example Is<string, "one line of at most 60 characters">; it got ${got}.`;

type Token = { kind: 'str' | 'num' | 'id' | 'p'; value: string };
// Whitespace and comments (field doc comments in types.ts aliases) separate tokens.
const TOKEN = /(?:\s|\/\*[^]*?\*\/|\/\/[^\n]*)*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|(-?\d+(?:\.\d+)?)|([A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*)|(\[\]|=>)|([{}<>|&,;:?()\[\]]))/y;
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
    // TypeScript allows a leading `|`, as multi-line unions are often written.
    if (this.peek()?.value === '|') this.eat('|');
    const members = [this.intersection()];
    while (this.peek()?.value === '|') { this.eat('|'); members.push(this.intersection()); }
    if (members.length === 1) return members[0]!;
    return { kind: 'union', members: members.flatMap(m => m.kind === 'union' ? m.members : [m]) };
  }
  private intersection(): Type {
    if (this.peek()?.value === '&') this.eat('&');
    const members = [this.postfix()];
    while (this.peek()?.value === '&') { this.eat('&'); members.push(this.postfix()); }
    if (members.length === 1) return members[0]!;
    const flat = members.flatMap(m => m.kind === 'intersection' ? m.members : [m]);
    // Object types written inline merge now; a member that is an alias merges when the alias is resolved.
    return flat.every(m => m.kind === 'record') ? mergeIntersection(flat, type => type) : { kind: 'intersection', members: flat };
  }
  private postfix(): Type {
    let type = this.atom();
    for (;;) {
      if (this.peek()?.value === '[]') { this.eat('[]'); type = { kind: 'list', element: type }; }
      else if (this.peek()?.value === '[' && this.tokens[this.index + 1]?.kind === 'str') {
        this.eat('['); const key = this.eat().value; this.eat(']');
        type = type.kind === 'record' ? indexedType(type, key, item => item) : { kind: 'index', base: type, key };
      } else break;
    }
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
  private record(): Type {
    this.eat('{');
    const fields: Extract<Type, { kind: 'record' }>['fields'] = [];
    if (this.peek()?.value === '[') return this.indexSignature();
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
  /** `{ [key: string]: T }` is a dictionary; the opening brace is already read. */
  private indexSignature(): Type {
    this.eat('[');
    const name = this.eat();
    if (name.kind !== 'id') throw new TypeSyntaxError(`bad index signature key ${JSON.stringify(name.value)}`);
    this.eat(':');
    const keyType = this.eat();
    if (keyType.value !== 'string') throw new TypeSyntaxError('an index signature takes string keys: write { [key: string]: T }');
    this.eat(']'); this.eat(':');
    const element = this.union();
    if (this.peek()?.value === ',' || this.peek()?.value === ';') this.eat();
    if (this.peek()?.value !== '}') throw new TypeSyntaxError('an index signature stands alone in its braces: write { [key: string]: T }, or combine fields with `&`');
    this.eat('}');
    return { kind: 'dict', element };
  }
  /** `[A, B]`, `[]`, `[A, B?]`, `[first: A, second?: B]` (trailing comma allowed). */
  private tuple(): Type {
    this.eat('[');
    const elements: Type[] = [];
    let minLength: number | undefined;
    while (this.peek()?.value !== ']') {
      let labeledOptional = false;
      if (this.peek()?.kind === 'id') {
        const next = this.tokens[this.index + 1]?.value;
        const after = this.tokens[this.index + 2]?.value;
        if (next === ':') { this.eat(); this.eat(':'); }
        else if (next === '?' && after === ':') { this.eat(); this.eat('?'); this.eat(':'); labeledOptional = true; }
      }
      const element = this.union();
      const optional = labeledOptional || this.peek()?.value === '?';
      if (!labeledOptional && this.peek()?.value === '?') this.eat('?');
      if (optional) {
        if (minLength === undefined) minLength = elements.length;
      } else if (minLength !== undefined) {
        throw new TypeSyntaxError('a required tuple element cannot follow an optional element');
      }
      elements.push(element);
      if (this.peek()?.value === ',') this.eat(',');
      else break;
    }
    this.eat(']');
    return minLength === undefined ? { kind: 'tuple', elements } : { kind: 'tuple', elements, minLength };
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
            // A natlang function is awaited: `=> Promise<R>` (the TypeScript spelling) is the same type as `=> R`.
            const promised = this.peek()?.value === 'Promise' && this.tokens[this.index + 1]?.value === '<';
            if (promised) this.eat();
            return { kind: 'lambda', params: { kind: 'record', fields }, returns: promised ? this.args(1)[0]! : this.union() };
          }
        }
      }
      this.index = start; this.eat('('); const inner = this.union(); this.eat(')'); return inner;
    }
    if (token.value === '{') return this.record();
    if (token.value === '[]') { this.eat('[]'); return { kind: 'tuple', elements: [] }; }
    if (token.value === '[') return this.tuple();
    this.eat();
    if (token.kind === 'str') return { kind: 'lit', value: token.value };
    if (token.kind === 'num') return { kind: 'lit', value: Number(token.value) };
    if (token.kind !== 'id') throw new TypeSyntaxError(`unexpected ${JSON.stringify(token.value)}`);
    if (token.value === 'never') return { kind: 'union', members: [] };
    if (PRIMS.has(token.value)) return { kind: 'prim', name: token.value as Extract<Type, { kind: 'prim' }>['name'] };
    if (token.value === 'Record') {
      const [key, value] = this.args(2);
      // Record<'a' | 'b', T> is the record with exactly those fields, as in TypeScript. Record<string, T>, and a key
      // named by an alias (Record<Kind, T>, resolved only later), is a dictionary from string keys.
      const keys = key?.kind === 'lit' ? [key] : key?.kind === 'union' ? key.members : undefined;
      if (keys?.length && keys.every(item => item.kind === 'lit' && typeof item.value === 'string')) {
        const names = [...new Set(keys.map(item => String((item as { value: string }).value)))];
        return { kind: 'record', fields: names.map(name => ({ name, type: value!, optional: false })) };
      }
      if (key?.kind === 'name') return { kind: 'dict', element: value!, key: key.name };
      if (key?.kind === 'prim' && key.name === 'string') return { kind: 'dict', element: value! };
      if (key?.kind === 'index' || (key?.kind === 'union' && key.members.every(item =>
          item.kind === 'name' || item.kind === 'index' || (item.kind === 'lit' && typeof item.value === 'string') ||
          (item.kind === 'prim' && item.name === 'string'))))
        return { kind: 'dict', element: value!, keyType: key };
      throw new TypeSyntaxError('Record keys are strings: write Record<string, T>, a union of string literals, or a type name for one');
    }
    if ((token.value === 'Partial' || token.value === 'Required') && this.peek()?.value === '<') {
      const [target] = this.args(1);
      return { kind: 'utility', op: token.value, target: target! };
    }
    if ((token.value === 'Pick' || token.value === 'Omit') && this.peek()?.value === '<') {
      const [target, keys] = this.args(2);
      return { kind: 'utility', op: token.value, target: target!, keys: keys! };
    }
    if (token.value === 'NativeAdapter') return NATIVE_ADAPTER_TYPE;
    if (token.value === 'NativeTensor') return NATIVE_TENSOR_TYPE;
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
    if (token.value === 'Is' && this.peek()?.value === '<') {
      // Is<T, "predicate">: one nonempty string literal; nesting means both predicates.
      this.eat('<');
      const base = this.union();
      if (this.peek()?.value !== ',') throw new TypeSyntaxError(refinementInvalid('a second type argument'));
      this.eat(',');
      const item = this.peek();
      if (item?.kind !== 'str') throw new TypeSyntaxError(refinementInvalid(item ? `${JSON.stringify(item.value)}, which is not a string literal` : 'nothing'));
      this.eat();
      const predicate = normalizePredicate(item.value);
      if (!predicate) throw new TypeSyntaxError(refinementInvalid('an empty string'));
      this.eat('>');
      return { kind: 'refined', base, predicate };
    }
    if (token.value === 'Untrusted' && this.peek()?.value === '<') return { kind: 'untrusted', base: this.args(1)[0]! };
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

// Cap both many small declarations and a few large ones. Source characters bound the
// number of tokens and objects in the tree, so changing a program cannot grow the cache unbounded.
const MAX_TYPE_CACHE_ENTRIES = 4096;
const MAX_TYPE_CACHE_CHARACTERS = 512 * 1024;
const MAX_CACHEABLE_TYPE_CHARACTERS = 16 * 1024;
const typeCacheEnabled = typeof process === 'undefined' || process.env?.NATLANG_TYPE_PARSE_CACHE !== 'off';
const typeCache = new Map<string, Type>();
let typeCacheCharacters = 0;

function freezeTypeTree(node: object): void {
  if (Object.isFrozen(node)) return;
  for (const value of Object.values(node))
    if (value !== null && typeof value === 'object') freezeTypeTree(value);
  Object.freeze(node);
}

/** Parse a static declaration for internal read-only use. The public parser still creates a mutable tree. */
/** The message for using an alias whose definition the type grammar cannot read. */
const unreadableAlias = (alias: { name: string; unparsed?: string }) => new UnreadableAliasError(
  `type ${alias.name} is not in the natlang type grammar (${alias.unparsed}); a signature that uses it cannot be checked, so ` +
  `rewrite ${alias.name} with records, unions, literals, arrays and Record, or leave it out of natlang signatures`);
/** A signature names an alias whose definition the type grammar cannot read. */
export class UnreadableAliasError extends TypeSyntaxError {}

/**
 * Type alias definitions by name, parsed. A definition the grammar cannot read (a TypeScript-only type in `types.ts`)
 * becomes a marker that reports why where it is used, so it breaks only the signatures that name it.
 */
export function parseAliases(texts: Readonly<Record<string, string>>, cached = true): Record<string, Type> {
  const names: Record<string, Type> = {};
  for (const [name, text] of Object.entries(texts)) {
    try { names[name] = cached ? parseTypeCached(text) : parseType(text); }
    catch (error) {
      if (!(error instanceof TypeSyntaxError)) throw error;
      names[name] = { kind: 'name', name, unparsed: error.message };
    }
  }
  return names;
}

export function parseTypeCached(text: string): Type {
  if (!typeCacheEnabled || typeof text !== 'string' || text.length > MAX_CACHEABLE_TYPE_CHARACTERS)
    return parseType(text);
  const hit = typeCache.get(text);
  if (hit) return hit;

  const type = parseType(text);
  freezeTypeTree(type);
  while (typeCache.size >= MAX_TYPE_CACHE_ENTRIES || typeCacheCharacters + text.length > MAX_TYPE_CACHE_CHARACTERS) {
    const oldest = typeCache.keys().next().value!;
    typeCacheCharacters -= oldest.length;
    typeCache.delete(oldest);
  }
  typeCache.set(text, type);
  typeCacheCharacters += text.length;
  return type;
}

export function formatType(type: Type): string {
  switch (type.kind) {
    case 'prim': case 'name': return type.name;
    case 'lit': return typeof type.value === 'string' ? JSON.stringify(type.value) : String(type.value);
    case 'record': return type.fields.length ? `{ ${type.fields.map(f => `${/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(f.name) ? f.name : JSON.stringify(f.name)}${f.optional ? '?' : ''}: ${formatType(f.type)}`).join(', ')} }` : '{}';
    case 'list': return (type.element.kind === 'union' ? `(${formatType(type.element)})` : formatType(type.element)) + '[]';
    case 'tuple': {
      const required = type.minLength ?? type.elements.length;
      return `[${type.elements.map((element, index) => {
        const text = formatType(element);
        return index >= required ?
          `${element.kind === 'union' || element.kind === 'intersection' || element.kind === 'lambda' ? `(${text})` : text}?` : text;
      }).join(', ')}]`;
    }
    case 'dict': return `Record<${type.keyType ? formatType(type.keyType) : type.key ?? 'string'}, ${formatType(type.element)}>`;
    case 'union': return type.members.length ? type.members.map(formatType).join(' | ') : 'never';
    case 'lambda': return `(${type.params.fields.map(field => `${field.name}${field.optional ? '?' : ''}: ` +
      formatType(field.type)).join(', ')}) => ${formatType(type.returns)}`;
    case 'host': return type.name;
    case 'intersection': return type.members.map(m => m.kind === 'union' ? `(${formatType(m)})` : formatType(m)).join(' & ');
    case 'index': return `${type.base.kind === 'union' || type.base.kind === 'intersection' ? `(${formatType(type.base)})` : formatType(type.base)}[${JSON.stringify(type.key)}]`;
    case 'utility': return type.op === 'Pick' || type.op === 'Omit' ?
      `${type.op}<${formatType(type.target)}, ${formatType(type.keys)}>` : `${type.op}<${formatType(type.target)}>`;
    case 'refined': return `Is<${formatType(type.base)}, ${JSON.stringify(type.predicate)}>`;
    case 'untrusted': return `Untrusted<${formatType(type.base)}>`;
    case 'neuralese': return `Neuralese<${formatType(type.element)}${type.dialect === DEFAULT_DIALECT ? '' :
      `, ${JSON.stringify(type.dialect)}`}>`;
  }
}

/** Child types of a type, for structural walks. */
function children(type: Type): Type[] {
  switch (type.kind) {
    case 'record': return type.fields.map(field => field.type);
    case 'list': case 'neuralese': return [type.element];
    case 'tuple': return type.elements;
    case 'dict': return type.keyType ? [type.keyType, type.element] : [type.element];
    case 'refined': case 'untrusted': return [type.base];
    case 'union': case 'intersection': return type.members;
    case 'index': return [type.base];
    case 'utility': return type.op === 'Pick' || type.op === 'Omit' ? [type.target, type.keys] : [type.target];
    case 'lambda': return [type.params, type.returns];
    default: return [];
  }
}

/** A name that can be redefined by the lexical type environment. */
export function typeAliasName(type: Type): string | undefined {
  if (type.kind === 'name' || type.kind === 'host' && type.contract.kind === 'tag' && type.contract.tag === type.name)
    return type.name;
}

/** Names whose definitions participate in this portable type. */
export function typeAliasNames(type: Type): string[] {
  const name = typeAliasName(type);
  if (name !== undefined) return [name];
  return [...(type.kind === 'dict' && type.key ? [type.key] : []), ...children(type).flatMap(typeAliasNames)];
}

export class TypeEnv {
  /** Constructors used to check `class` host contracts; inherited by child environments. */
  classes?: ReadonlyMap<string, Function>;
  /** The runtime's dialect binding; inherited by child environments. Without one, `DefaultDialect` is unbound. */
  dialects?: DialectBinding;
  constructor(readonly names: Record<string, Type> = {}, readonly parent?: TypeEnv) {
    this.classes = parent?.classes;
    this.dialects = parent?.dialects;
    this.checkRecursiveFunctions();
  }
  /**
   * Reject an alias that reaches itself through a function type (`type-recursive-function`): self-application
   * would make recursion expressible without a recursive definition. Recursive data types stay allowed.
   */
  private checkRecursiveFunctions(): void {
    const aliases = Object.keys(this.names);
    if (!aliases.length) return;
    type Reference = { name: string; throughFunction: boolean };
    // Extract each definition's direct edges once in this lexical scope. Checking
    // multiple roots reuses these edges instead of walking their shared syntax.
    const references = new Map<string, Reference[]>();
    const collect = (type: Type, throughFunction: boolean, edges: Reference[]): void => {
      if (type.kind === 'name') {
        edges.push({ name: type.name, throughFunction });
        return;
      }
      for (const child of children(type)) collect(child, throughFunction || type.kind === 'lambda', edges);
    };
    const edgesOf = (name: string): readonly Reference[] => {
      let edges = references.get(name);
      if (edges) return edges;
      edges = [];
      const type = this.lookup(name);
      if (type) collect(type, false, edges);
      references.set(name, edges);
      return edges;
    };
    const reachable = new Set(aliases);
    let hasFunctionEdge = false;
    for (const name of reachable) {
      for (const edge of edgesOf(name)) {
        reachable.add(edge.name);
        if (edge.throughFunction) hasFunctionEdge = true;
      }
    }
    if (!hasFunctionEdge) return;
    for (const alias of aliases) {
      const plain = new Set([alias]);
      const throughLambda = new Set<string>();
      const pending = [...edgesOf(alias)];
      while (pending.length > 0) {
        const { name, throughFunction } = pending.pop()!;
        if (name === alias && throughFunction)
          throw new TypeSyntaxError(`type-recursive-function: type ${alias} refers to itself through a function type`);
        const seen = throughFunction ? throughLambda : plain;
        if (seen.has(name)) continue;
        seen.add(name);
        for (const edge of edgesOf(name)) {
          pending.push(throughFunction && !edge.throughFunction ? { name: edge.name, throughFunction } : edge);
        }
      }
    }
  }
  child(names: Record<string, Type>): TypeEnv { return Object.keys(names).length ? new TypeEnv({ ...names }, this) : this; }
  lookup(name: string): Type | undefined { return this.names[name] ?? this.parent?.lookup(name); }
  /** The dialect a type's dialect name means here: `DefaultDialect` is the bound reader dialect, when there is one. */
  dialect(name: string): string { return name === DEFAULT_DIALECT ? this.dialects?.dialect ?? name : name; }
  resolve(type: Type): Type {
    const seen = new Set<string>();
    for (;;) {
      if (type.kind === 'name') {
        if (seen.has(type.name)) throw new TypeSyntaxError(`type ${type.name} is defined only in terms of itself`);
        seen.add(type.name);
        const found = this.lookup(type.name);
        if (!found) throw new TypeSyntaxError(`unknown type name ${type.name}`);
        if (found.kind === 'name' && found.unparsed !== undefined) throw unreadableAlias(found);
        type = found;
      } else if (type.kind === 'host' && type.contract.kind === 'tag' && type.name === type.contract.tag &&
        this.lookup(type.name)) {
        // A program's own type shadows a platform global of the same name in its scope, as in TypeScript:
        // `type Request = { id: string }` makes `Request` that record, not fetch's Request.
        type = { kind: 'name', name: type.name };
      } else if (type.kind === 'dict' && (type.key || type.keyType))
        type = this.recordOfKeys(type.keyType ?? { kind: 'name', name: type.key! }, type.element);
      else if (type.kind === 'intersection') type = mergeIntersection(type.members, member => this.resolve(member));
      else if (type.kind === 'index') type = indexedType(type.base, type.key, member => this.resolve(member));
      else if (type.kind === 'utility') type = applyUtility(type, member => this.resolve(member));
      else return type;
    }
  }
  /** `Record<Kind, T>`: the record with a field for each string literal `Kind` names, or a dictionary for `string`. */
  private recordOfKeys(keyType: Type, element: Type): Type {
    const members = flattenUnion(keyType, member => this.resolve(member));
    if (members.some(member => member.kind === 'prim' && member.name === 'string') &&
        members.every(member => (member.kind === 'prim' && member.name === 'string') || (member.kind === 'lit' && typeof member.value === 'string')))
      return { kind: 'dict', element };
    if (!members.length || !members.every(member => member.kind === 'lit' && typeof member.value === 'string'))
      throw new TypeSyntaxError(`Record keys are strings: ${formatType(keyType)} is ${formatType(this.resolve(keyType))}; write Record<string, T> or a union of string literals for its keys`);
    return { kind: 'record', fields: [...new Set(members.map(member => String((member as { value: string }).value)))]
      .map(name => ({ name, type: element, optional: false })) };
  }
  checkNames(type: Type): void {
    if (type.kind === 'name') {
      const found = this.lookup(type.name);
      if (!found) throw new TypeSyntaxError(`unknown type name ${type.name}`);
      if (found.kind === 'name' && found.unparsed !== undefined) throw unreadableAlias(found);
      return;
    }
    if (type.kind === 'dict' && type.key) this.checkNames({ kind: 'name', name: type.key });
    if (type.kind === 'dict' && type.keyType) this.checkNames(type.keyType);
    if (type.kind === 'record') for (const field of type.fields) this.checkNames(field.type);
    if (type.kind === 'list' || type.kind === 'dict') this.checkNames(type.element);
    if (type.kind === 'tuple') for (const element of type.elements) this.checkNames(element);
    if (type.kind === 'refined' || type.kind === 'untrusted') this.checkNames(type.base);
    if (type.kind === 'union' || type.kind === 'intersection') for (const member of type.members) this.checkNames(member);
    if (type.kind === 'index') this.checkNames(type.base);
    if (type.kind === 'utility') {
      this.checkNames(type.target);
      if (type.op === 'Pick' || type.op === 'Omit') this.checkNames(type.keys);
    }
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
  // Untrusted<T> is a T for crisp code; a plain value fits an Untrusted slot (it only loses trust, never gains it).
  if (b.kind === 'untrusted') return fitsType(a.kind === 'untrusted' ? a.base : a, b.base, env, seen);
  if (a.kind === 'untrusted') return fitsType(a.base, b, env, seen);
  // A refinement is forgotten going out; going in, the value must fit the base (the predicate is an obligation,
  // see fitObligations) and an already-refined value must carry every predicate of the target.
  if (b.kind === 'refined') {
    const wanted = refinementChain(b, env), have = a.kind === 'refined' ? refinementChain(a, env) : { base: a, predicates: [] };
    return fitsType(have.base, wanted.base, env, seen) && (!have.predicates.length || wanted.predicates.every(p => have.predicates.includes(p)));
  }
  if (a.kind === 'refined') return fitsType(a.base, b, env, seen);
  if (b.kind === 'union') return b.members.some(member => fitsType(a, member, env, seen));
  if (a.kind === 'lit' && b.kind === 'prim') return b.name === (typeof a.value === 'string' ? 'string' : 'number');
  if ((a.kind === 'list' || a.kind === 'dict') && a.kind === b.kind) return fitsType(a.element, b.element, env, seen);
  if (a.kind === 'tuple' && b.kind === 'tuple') {
    const aMin = a.minLength ?? a.elements.length, bMin = b.minLength ?? b.elements.length;
    if (aMin < bMin || a.elements.length > b.elements.length) return false;
    return a.elements.every((element, index) => fitsType(element, b.elements[index]!, env, seen));
  }
  if (a.kind === 'tuple' && b.kind === 'list')
    return a.elements.every(element => fitsType(element, b.element, env, seen));
  if (a.kind === 'record' && b.kind === 'record') {
    for (const field of b.fields) {
      const got = a.fields.find(f => f.name === field.name);
      if (!got) { if (!field.optional) return false; continue; }
      if (got.optional && !field.optional) return false;
      if (!fitsType(got.type, field.type, env, seen)) return false;
    }
    return a.fields.every(field => b.fields.some(other => other.name === field.name));
  }
  if (a.kind === 'host' && b.kind === 'host')
    return b.contract.kind === 'any' || JSON.stringify(a.contract) === JSON.stringify(b.contract);
  if (b.kind === 'host' && b.contract.kind === 'any') return true;
  if (a.kind === 'neuralese' && b.kind === 'neuralese')
    return env.dialect(a.dialect) === env.dialect(b.dialect) && fitsType(a.element, b.element, env, seen);
  if (a.kind === 'lambda' && b.kind === 'lambda') {
    // Parameters match by position, as in TypeScript: names are documentation. Each parameter the function takes must
    // accept what the slot passes there (contravariant), it may take fewer than the slot passes, and the result must
    // fit the slot's (covariant).
    const have = a.params.fields, wanted = b.params.fields;
    if (have.some((field, index) => !field.optional && index >= wanted.length)) return false;
    return fitsType(a.returns, b.returns, env, seen) &&
      have.every((field, index) => index >= wanted.length || fitsType(wanted[index]!.type, field.type, env, seen));
  }
  // A typed function fits a slot that only asks for some function.
  if (a.kind === 'lambda' && b.kind === 'host' && b.contract.kind === 'function') return true;
  return false;
}

/**
 * The checked signature every natlang callable carries (non-enumerable, under this symbol): its function type and the
 * type aliases its parameter and result types name, in the scope that defined it.
 */
export const NATLANG_SIGNATURE: unique symbol = Symbol.for('natlang.signature') as never;
export type FunctionSignature = { readonly type: Extract<Type, { kind: 'lambda' }>; readonly types: Readonly<Record<string, Type>> };

/** The signature a function carries, if it is a natlang callable or a function a typed slot already checked. */
export function signatureOf(value: unknown): FunctionSignature | undefined {
  if (typeof value !== 'function') return;
  try { return (value as unknown as Record<symbol, FunctionSignature | undefined>)[NATLANG_SIGNATURE]; } catch { return; }
}

/** `type` with every alias name replaced by `rename(name)`. */
export function renameTypeNames(type: Type, rename: (name: string) => string): Type {
  if (type.kind === 'name') return { ...type, name: rename(type.name) };
  const mapped = mapChildTypes(type, inner => renameTypeNames(inner, rename));
  return mapped.kind === 'dict' && mapped.key ? { ...mapped, key: rename(mapped.key) } : mapped;
}

/**
 * TypeScript text for a natlang type: a function type's result is a promise, since callers await natlang functions
 * (`(t: Ticket) => boolean` is `(t: Ticket) => Promise<boolean>`); everything else is written as `formatType` writes it.
 */
export function formatTypeScript(type: Type): string {
  const promised = (inner: Type): Type => inner.kind === 'lambda' ?
    { ...inner, params: promised(inner.params) as typeof inner.params,
      returns: { kind: 'host', name: `Promise<${formatTypeScript(inner.returns)}>`, contract: { kind: 'any' } } } :
    mapChildTypes(inner, promised);
  return formatType(promised(type));
}

/** `type` with each direct child type replaced by `child(type)`. */
function mapChildTypes(type: Type, child: (inner: Type) => Type): Type {
  switch (type.kind) {
    case 'record': return { ...type, fields: type.fields.map(field => ({ ...field, type: child(field.type) })) };
    case 'list': case 'neuralese': return { ...type, element: child(type.element) };
    case 'tuple': return { ...type, elements: type.elements.map(child) };
    case 'dict': return { ...type, element: child(type.element), ...(type.keyType ? { keyType: child(type.keyType) } : {}) };
    case 'union': case 'intersection': return { ...type, members: type.members.map(child) };
    case 'index': case 'refined': case 'untrusted': return { ...type, base: child(type.base) };
    case 'utility': return type.op === 'Pick' || type.op === 'Omit' ?
      { ...type, target: child(type.target), keys: child(type.keys) } : { ...type, target: child(type.target) };
    case 'lambda': return { ...type, params: child(type.params) as typeof type.params, returns: child(type.returns) };
    default: return type;
  }
}

/**
 * Whether a function with `signature` fits a slot of function type `wanted` in `env`. The signature's aliases are its
 * definer's: one that `env` defines the same way is shared, and one it defines differently (or not at all) is compared
 * by its own definition.
 */
export function signatureFits(signature: FunctionSignature, wanted: Type, env: TypeEnv): boolean {
  const own = signature.types;
  // An alias is foreign when the slot's scope defines the same name differently; a name the slot's scope does not
  // define (an eval's own alias, seen from a binding checked without it) means what the definer's scope says.
  const foreign = new Set(Object.keys(own).filter(name => {
    const there = env.lookup(name);
    return there !== undefined && JSON.stringify(there) !== JSON.stringify(own[name]);
  }));
  if (!foreign.size) return fitsType(signature.type, wanted, env.child({ ...own }));
  const rename = (name: string) => foreign.has(name) ? `${name}\u0000signature` : name;
  const names = Object.fromEntries(Object.entries(own).map(([name, type]) => [rename(name), renameTypeNames(type, rename)]));
  return fitsType(renameTypeNames(signature.type, rename), wanted, env.child(names));
}

/** The predicates of a (possibly nested) refinement, outermost last, and the unrefined type beneath them. */
export function refinementChain(type: Type, env: TypeEnv): { base: Type; predicates: string[] } {
  const predicates: string[] = [];
  let current = env.resolve(type);
  while (current.kind === 'refined') {
    predicates.unshift(current.predicate);
    current = env.resolve(current.base);
  }
  return { base: current, predicates: [...new Set(predicates)] };
}

/** Whether `type` mentions a refinement anywhere (named types followed once each). */
export function containsRefinement(type: Type, env: TypeEnv, seen = new Set<string>()): boolean {
  if (type.kind === 'refined') return true;
  if (type.kind === 'name') {
    if (seen.has(type.name)) return false;
    seen.add(type.name);
    const found = env.lookup(type.name);
    return !!found && containsRefinement(found, env, seen);
  }
  if (type.kind === 'utility') {
    try { return containsRefinement(env.resolve(type), env, seen); } catch { /* fall through to children */ }
  }
  if (type.kind === 'neuralese' || type.kind === 'host') return false;
  return children(type).some(child => containsRefinement(child, env, seen));
}

/**
 * The check obligations of putting a value of `source` into `target`: the predicates of refined target positions that
 * the source does not already carry, by path. null when `source` does not fit `target` at all.
 */
export function fitObligations(source: Type, target: Type, env = new TypeEnv(), path = 'value'): { path: string; predicate: string }[] | null {
  if (!fitsType(source, target, env)) return null;
  const a = env.resolve(source), b = env.resolve(target);
  if (b.kind === 'untrusted' || a.kind === 'untrusted')
    return fitObligations(a.kind === 'untrusted' ? a.base : a, b.kind === 'untrusted' ? b.base : b, env, path);
  if (b.kind === 'refined') {
    const wanted = refinementChain(b, env), have = refinementChain(a, env);
    return wanted.predicates.filter(p => !have.predicates.includes(p)).map(predicate => ({ path, predicate }));
  }
  const out: { path: string; predicate: string }[] = [];
  const sub = (x: Type, y: Type, at: string) => { out.push(...fitObligations(x, y, env, at) ?? []); };
  if ((a.kind === 'list' && b.kind === 'list') || (a.kind === 'dict' && b.kind === 'dict')) sub(a.element, b.element, `${path}/*`);
  else if (a.kind === 'tuple' && b.kind === 'tuple')
    for (let i = 0; i < Math.min(a.elements.length, b.elements.length); i++) sub(a.elements[i]!, b.elements[i]!, `${path}/${i}`);
  else if (a.kind === 'tuple' && b.kind === 'list')
    for (let i = 0; i < a.elements.length; i++) sub(a.elements[i]!, b.element, `${path}/${i}`);
  else if (a.kind === 'record' && b.kind === 'record')
    for (const field of b.fields) { const got = a.fields.find(f => f.name === field.name); if (got) sub(got.type, field.type, `${path}/${field.name}`); }
  return out;
}

/**
 * Crisp checkers for refinement predicates, by normalized predicate text. A checker that returns a boolean decides;
 * `undefined` defers to the judge (plans/REFINEMENT_TYPES.md section 3). Register more with `refinements[predicate] = fn`;
 * a runtime's `refinements.crisp` setting is consulted before this table.
 */
export const refinements: Record<string, (value: unknown) => boolean | undefined> = Object.create(null);

/** Flatten nested unions while resolving aliases, guarding against recursive unions. */
function flattenUnion(type: Type, resolve: (type: Type) => Type, activeUnions = new Set<Type>()): Type[] {
  const resolved = resolve(type);
  if (resolved.kind !== 'union') return [resolved];
  if (activeUnions.has(resolved)) throw new TypeSyntaxError('cannot flatten a recursive union');
  activeUnions.add(resolved);
  try {
    return resolved.members.flatMap(member => flattenUnion(member, resolve, activeUnions));
  } finally {
    activeUnions.delete(resolved);
  }
}

/** Resolve `Partial<R>`, `Required<R>`, `Pick<R, K>`, `Omit<R, K>` over records and unions of records. */
function applyUtility(type: Extract<Type, { kind: 'utility' }>, resolve: (type: Type) => Type, activeUnions = new Set<Type>()): Type {
  const resolved = resolve(type.target);
  if (resolved.kind === 'union') {
    if (activeUnions.has(resolved))
      throw new TypeSyntaxError(`cannot apply ${type.op} through a recursive union`);
    activeUnions.add(resolved);
    try {
      const members = resolved.members.flatMap(member => {
        const applied = applyUtility({ ...type, target: member }, resolve, activeUnions);
        return applied.kind === 'union' ? applied.members : [applied];
      });
      return members.length === 1 ? members[0]! : { kind: 'union', members };
    } finally {
      activeUnions.delete(resolved);
    }
  }
  if (resolved.kind !== 'record')
    throw new TypeSyntaxError(`${formatType(type)} requires a record type or a union of record types, got ${formatType(resolved)}`);
  if (type.op === 'Partial')
    return { kind: 'record', fields: resolved.fields.map(field => ({ ...field, optional: true })) };
  if (type.op === 'Required')
    return { kind: 'record', fields: resolved.fields.map(field => ({ ...field, optional: false })) };
  if (!('keys' in type)) return resolved;
  const keyMembers = flattenUnion(type.keys, resolve);
  if (!keyMembers.length || !keyMembers.every(member => member.kind === 'lit' && typeof member.value === 'string'))
    throw new TypeSyntaxError(`${formatType(type)} requires string literal keys, got ${formatType(type.keys)}`);
  const keyNames = [...new Set(keyMembers.map(member => String((member as { value: string }).value)))];
  const keySet = new Set(keyNames);
  if (type.op === 'Pick') {
    for (const key of keyNames) {
      if (!resolved.fields.some(field => field.name === key))
        throw new TypeSyntaxError(`${formatType(type)}: ${formatType(resolved)} has no field ${JSON.stringify(key)}`);
    }
    return { kind: 'record', fields: resolved.fields.filter(field => keySet.has(field.name)) };
  }
  return { kind: 'record', fields: resolved.fields.filter(field => !keySet.has(field.name)) };
}

/** Select a field from every possible base. Optional fields also admit null. */
function indexedType(base: Type, key: string, resolve: (type: Type) => Type, activeUnions = new Set<Type>()): Type {
  const resolved = resolve(base);
  if (resolved.kind === 'union') {
    if (activeUnions.has(resolved))
      throw new TypeSyntaxError(`cannot index field ${JSON.stringify(key)} through a recursive union`);
    activeUnions.add(resolved);
    try {
      const members = resolved.members.flatMap(member => {
        const field = indexedType(member, key, resolve, activeUnions);
        return field.kind === 'union' ? field.members : [field];
      });
      return { kind: 'union', members };
    } finally {
      activeUnions.delete(resolved);
    }
  }
  if (resolved.kind === 'record') {
    const field = resolved.fields.find(item => item.name === key);
    if (!field) throw new TypeSyntaxError(`${formatType(resolved)} has no field ${JSON.stringify(key)}; index it with one of its field names`);
    return field.optional ? { kind: 'union', members: [field.type, { kind: 'prim', name: 'null' }] } : field.type;
  }
  if (resolved.kind === 'dict') return resolved.element;
  throw new TypeSyntaxError(`${formatType(resolved)}[${JSON.stringify(key)}] requires a record, dictionary, or a union of indexable types`);
}

/** The intersection of structural types. Record fields remain lazy so recursive aliases are not expanded eagerly. */
function mergeIntersection(members: Type[], resolve: (type: Type) => Type, activeUnions = new Set<Type>()): Type {
  const parts = members.map(resolve).filter(part => !(part.kind === 'prim' && part.name === 'unknown'));
  if (!parts.length) return { kind: 'prim', name: 'unknown' };
  const union = parts.find(part => part.kind === 'union');
  if (union?.kind === 'union') {
    if (activeUnions.has(union)) throw new TypeSyntaxError('cannot intersect through a recursive union');
    activeUnions.add(union);
    try {
      const rest = parts.filter(part => part !== union);
      const alternatives = union.members.flatMap(member => {
        const narrowed = mergeIntersection([member, ...rest], resolve, activeUnions);
        return narrowed.kind === 'union' ? narrowed.members : [narrowed];
      });
      const unique = [...new Map(alternatives.map(type => [JSON.stringify(type), type])).values()];
      return unique.length === 1 ? unique[0]! : { kind: 'union', members: unique };
    } finally { activeUnions.delete(union); }
  }
  if (parts.length === 1 || parts.every(part => JSON.stringify(part) === JSON.stringify(parts[0]))) return parts[0]!;
  // This path also runs while parsing inline records, before their aliases can be resolved.
  if (parts.some(part => part.kind === 'name' || part.kind === 'intersection' || part.kind === 'index' || part.kind === 'utility'))
    return { kind: 'intersection', members: parts };
  if (parts.some(part => part.kind === 'untrusted')) return { kind: 'untrusted',
    base: mergeIntersection(parts.map(part => part.kind === 'untrusted' ? part.base : part), resolve, activeUnions) };
  if (parts.some(part => part.kind === 'refined')) {
    const predicates: string[] = [], bases: Type[] = [];
    for (const part of parts) {
      let base = part;
      const seen = new Set<Type>();
      while (base.kind === 'refined') {
        if (seen.has(base)) throw new TypeSyntaxError('cannot intersect through a recursive refinement');
        seen.add(base); predicates.push(base.predicate); base = resolve(base.base);
      }
      bases.push(base);
    }
    const base = mergeIntersection(bases, resolve, activeUnions);
    if (base.kind === 'union' && !base.members.length) return base;
    return [...new Set(predicates)].reduce<Type>((type, predicate) => ({ kind: 'refined', base: type, predicate }), base);
  }
  const scalar = (part: Type) => part.kind === 'lit' || part.kind === 'prim' && ['string', 'number', 'boolean', 'null'].includes(part.name);
  if (parts.some(scalar)) {
    // Live/soft values and the legacy Blob codec can transform their input representation. Do not infer disjointness
    // from their type tag, or silently erase their contract when intersecting a scalar.
    if (parts.some(part => part.kind === 'host' || part.kind === 'neuralese' || part.kind === 'prim' && part.name === 'Blob'))
      throw new TypeSyntaxError(`${members.map(formatType).join(' & ')} has an unsupported structural intersection involving a live or encoded value`);
    const literal = parts.find(part => part.kind === 'lit');
    if (literal?.kind === 'lit' && parts.every(part => part.kind === 'lit' ? part.value === literal.value :
      part.kind === 'prim' && part.name === typeof literal.value)) return literal;
    return { kind: 'union', members: [] };
  }
  const atomic = (type: Type): boolean => type.kind === 'prim' || type.kind === 'lit' ||
    type.kind === 'union' && type.members.every(atomic);
  const combined = (types: Type[]): Type => types.every(type => JSON.stringify(type) === JSON.stringify(types[0]))
    ? types[0]! : types.every(atomic) ? mergeIntersection(types, type => type) : { kind: 'intersection', members: types };
  if (parts.every(part => part.kind === 'record' || part.kind === 'dict') && parts.some(part => part.kind === 'record')) {
    const fields: Extract<Type, { kind: 'record' }>['fields'] = [];
    const dictionaries = parts.filter(part => part.kind === 'dict');
    for (const part of parts) if (part.kind === 'record') for (const field of part.fields) {
      const at = fields.findIndex(item => item.name === field.name);
      if (at < 0) { fields.push({ ...field }); continue; }
      const had = fields[at]!;
      fields[at] = { name: field.name, optional: had.optional && field.optional, type: combined([had.type, field.type]) };
    }
    const narrowed = fields.map(field => ({ ...field,
      type: combined([field.type, ...dictionaries.map(part => part.element)]) }));
    if (narrowed.some(field => !field.optional && field.type.kind === 'union' && !field.type.members.length))
      return { kind: 'union', members: [] };
    return { kind: 'record', fields: narrowed };
  }
  if (parts.every(part => part.kind === 'dict')) return { kind: 'dict', element: combined(parts.map(part => part.element)) };
  if (parts.every(part => part.kind === 'list')) return { kind: 'list', element: combined(parts.map(part => part.element)) };
  if (parts.every(part => part.kind === 'tuple' || part.kind === 'list') && parts.some(part => part.kind === 'tuple')) {
    const tuples = parts.filter(part => part.kind === 'tuple');
    const length = Math.min(...tuples.map(part => part.elements.length));
    const required = Math.max(...tuples.map(part => part.minLength ?? part.elements.length));
    if (required > length) return { kind: 'union', members: [] };
    const elements = Array.from({ length }, (_, i) => combined(parts.map(part =>
      part.kind === 'tuple' ? part.elements[i]! : part.element)));
    if (elements.some((type, i) => i < required && type.kind === 'union' && !type.members.length))
      return { kind: 'union', members: [] };
    return { kind: 'tuple', elements, ...(required < length ? { minLength: required } : {}) };
  }
  // Exact records, arrays and dictionaries have distinct native value representations.
  if (parts.every(part => ['record', 'dict', 'list', 'tuple'].includes(part.kind))) return { kind: 'union', members: [] };
  throw new TypeSyntaxError(`${members.map(formatType).join(' & ')} has an unsupported structural intersection; preserve its constraints in an explicit supported type`);
}
