/** The dialect a `Neuralese<T>` without a second argument names; a program's configuration binds it. */
export const DEFAULT_DIALECT = 'DefaultDialect';
/** `Adapter`: a weight adapter value (a stored coefficient block whose dialect is its spec; model/tiny_adapters.py).
 * It is a soft value of a marker element, so it fits `Adapter` and `Neuralese<unknown>` slots and no other. */
export const ADAPTER_TYPE = { kind: 'neuralese', dialect: DEFAULT_DIALECT,
    element: { kind: 'record', fields: [{ name: '$adapter', type: { kind: 'lit', value: 'adapter' }, optional: false }] } };
export const isAdapterType = (type) => type.kind === 'neuralese' && type.dialect === DEFAULT_DIALECT &&
    type.element.kind === 'record' && type.element.fields.length === 1 && type.element.fields[0].name === '$adapter';
const BUILTIN_HOST_TAGS = new Set(['Date', 'Map', 'Set', 'WeakMap', 'WeakSet', 'RegExp', 'Error', 'URL',
    'URLSearchParams', 'Uint8Array', 'ArrayBuffer', 'Headers', 'Response', 'Request']);
/** Check a live value against a host contract. Built-ins use their internal tag, which works across realms. */
export function checkHost(value, contract, classes) {
    if (contract.kind === 'any')
        return value !== undefined;
    if (contract.kind === 'function')
        return typeof value === 'function';
    if (value === null || (typeof value !== 'object' && typeof value !== 'function'))
        return false;
    if (contract.kind === 'tag')
        return Object.prototype.toString.call(value) === `[object ${contract.tag}]`;
    if (contract.kind === 'class') {
        const constructor = classes?.get(contract.name);
        return constructor ? value instanceof constructor :
            value.constructor?.name === contract.name;
    }
    return contract.members.every(member => member in value);
}
export class TypeSyntaxError extends Error {
}
/** A predicate with its whitespace collapsed: two predicates are the same when their normalized texts are equal. */
export const normalizePredicate = (predicate) => predicate.replace(/\s+/g, ' ').trim();
const refinementInvalid = (got) => `refinement-predicate-invalid: Is<T, P> takes a nonempty string literal as P, ` +
    `for example Is<string, "one line of at most 60 characters">; it got ${got}.`;
// Whitespace and comments (field doc comments in types.ts aliases) separate tokens.
const TOKEN = /(?:\s|\/\*[^]*?\*\/|\/\/[^\n]*)*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|(-?\d+(?:\.\d+)?)|([A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*)|(\[\]|=>)|([{}<>|&,;:?()\[\]]))/y;
const PRIMS = new Set(['string', 'number', 'boolean', 'null', 'Blob', 'Folder', 'FileHandle', 'unknown']);
function tokenize(source) {
    const text = source.trim(), result = [];
    let position = 0;
    while (position < text.length) {
        TOKEN.lastIndex = position;
        const match = TOKEN.exec(text);
        if (!match)
            throw new TypeSyntaxError(`bad character at ${position}: ${JSON.stringify(text.slice(position, position + 12))}`);
        // TypeScript spells string literals with either quote; a single-quoted one is the same literal.
        const single = match[2]?.replace(/\\'/g, "'");
        const kind = match[1] !== undefined || single !== undefined ? 'str' : match[3] !== undefined ? 'num' :
            match[4] !== undefined ? 'id' : 'p';
        // Double-quoted names/literals use JSON escaping, matching generated type
        // signatures and compiler targets. Decode before duplicate-field checks.
        const quoted = match[1] !== undefined ? JSON.parse(`"${match[1]}"`) : undefined;
        result.push({ kind, value: quoted ?? single ?? match[3] ?? match[4] ?? match[5] ?? match[6] });
        position = TOKEN.lastIndex;
    }
    return result;
}
class Parser {
    tokens;
    index = 0;
    constructor(tokens) {
        this.tokens = tokens;
    }
    peek() { return this.tokens[this.index]; }
    eat(value) {
        const token = this.tokens[this.index];
        if (!token || (value !== undefined && token.value !== value))
            throw new TypeSyntaxError(`expected ${JSON.stringify(value ?? 'token')}, got ${JSON.stringify(token?.value)}`);
        this.index++;
        return token;
    }
    parse() {
        const result = this.union();
        if (this.peek())
            throw new TypeSyntaxError(`trailing input at ${JSON.stringify(this.peek().value)}`);
        return result;
    }
    union() {
        // TypeScript allows a leading `|`, as multi-line unions are often written.
        if (this.peek()?.value === '|')
            this.eat('|');
        const members = [this.intersection()];
        while (this.peek()?.value === '|') {
            this.eat('|');
            members.push(this.intersection());
        }
        if (members.length === 1)
            return members[0];
        return { kind: 'union', members: members.flatMap(m => m.kind === 'union' ? m.members : [m]) };
    }
    intersection() {
        if (this.peek()?.value === '&')
            this.eat('&');
        const members = [this.postfix()];
        while (this.peek()?.value === '&') {
            this.eat('&');
            members.push(this.postfix());
        }
        if (members.length === 1)
            return members[0];
        const flat = members.flatMap(m => m.kind === 'intersection' ? m.members : [m]);
        // Object types written inline merge now; a member that is an alias merges when the alias is resolved.
        return flat.every(m => m.kind === 'record') ? mergeIntersection(flat, type => type) : { kind: 'intersection', members: flat };
    }
    postfix() {
        let type = this.atom();
        for (;;) {
            if (this.peek()?.value === '[]') {
                this.eat('[]');
                type = { kind: 'list', element: type };
            }
            else if (this.peek()?.value === '[' && this.tokens[this.index + 1]?.kind === 'str') {
                this.eat('[');
                const key = this.eat().value;
                this.eat(']');
                type = type.kind === 'record' ? indexedType(type, key, item => item) : { kind: 'index', base: type, key };
            }
            else
                break;
        }
        return type;
    }
    args(count) {
        this.eat('<');
        const result = [this.union()];
        while (this.peek()?.value === ',') {
            this.eat(',');
            result.push(this.union());
        }
        this.eat('>');
        if (result.length !== count)
            throw new TypeSyntaxError(`expected ${count} type arguments, got ${result.length}`);
        return result;
    }
    record() {
        this.eat('{');
        const fields = [];
        if (this.peek()?.value === '[')
            return this.indexSignature();
        while (this.peek()?.value !== '}') {
            const name = this.eat();
            if (!['id', 'str'].includes(name.kind))
                throw new TypeSyntaxError(`bad field name ${JSON.stringify(name.value)}`);
            const optional = this.peek()?.value === '?';
            if (optional)
                this.eat('?');
            this.eat(':');
            fields.push({ name: name.value, type: this.union(), optional });
            if (this.peek()?.value === ',' || this.peek()?.value === ';')
                this.eat();
        }
        this.eat('}');
        if (new Set(fields.map(f => f.name)).size !== fields.length)
            throw new TypeSyntaxError('duplicate field name');
        return { kind: 'record', fields };
    }
    /** `{ [key: string]: T }` is a dictionary; the opening brace is already read. */
    indexSignature() {
        this.eat('[');
        const name = this.eat();
        if (name.kind !== 'id')
            throw new TypeSyntaxError(`bad index signature key ${JSON.stringify(name.value)}`);
        this.eat(':');
        const keyType = this.eat();
        if (keyType.value !== 'string')
            throw new TypeSyntaxError('an index signature takes string keys: write { [key: string]: T }');
        this.eat(']');
        this.eat(':');
        const element = this.union();
        if (this.peek()?.value === ',' || this.peek()?.value === ';')
            this.eat();
        if (this.peek()?.value !== '}')
            throw new TypeSyntaxError('an index signature stands alone in its braces: write { [key: string]: T }, or combine fields with `&`');
        this.eat('}');
        return { kind: 'dict', element };
    }
    atom() {
        const token = this.peek();
        if (!token)
            throw new TypeSyntaxError('unexpected end of type');
        if (token.value === '(') {
            const start = this.index;
            this.eat('(');
            const fields = [];
            let callable = this.peek()?.value === ')' ||
                (this.peek()?.kind === 'id' && [':', '?'].includes(this.tokens[this.index + 1]?.value ?? ''));
            if (callable) {
                while (this.peek()?.value !== ')') {
                    const name = this.eat();
                    if (name.kind !== 'id') {
                        callable = false;
                        break;
                    }
                    const optional = this.peek()?.value === '?';
                    if (optional)
                        this.eat('?');
                    this.eat(':');
                    fields.push({ name: name.value, optional, type: this.union() });
                    if (this.peek()?.value === ',')
                        this.eat(',');
                    else
                        break;
                }
                if (callable && this.peek()?.value === ')') {
                    this.eat(')');
                    if (this.peek()?.value === '=>') {
                        this.eat('=>');
                        return { kind: 'lambda', params: { kind: 'record', fields }, returns: this.union() };
                    }
                }
            }
            this.index = start;
            this.eat('(');
            const inner = this.union();
            this.eat(')');
            return inner;
        }
        if (token.value === '{')
            return this.record();
        this.eat();
        if (token.kind === 'str')
            return { kind: 'lit', value: token.value };
        if (token.kind === 'num')
            return { kind: 'lit', value: Number(token.value) };
        if (token.kind !== 'id')
            throw new TypeSyntaxError(`unexpected ${JSON.stringify(token.value)}`);
        if (PRIMS.has(token.value))
            return { kind: 'prim', name: token.value };
        if (token.value === 'Record') {
            const [key, value] = this.args(2);
            if (key?.kind !== 'prim' || key.name !== 'string')
                throw new TypeSyntaxError('Record keys must be string');
            return { kind: 'dict', element: value };
        }
        if (token.value === 'Adapter')
            return ADAPTER_TYPE;
        if (token.value === 'Neuralese' && this.peek()?.value === '<') {
            // Neuralese<T> or Neuralese<T, D>; D is a dialect name or string literal.
            this.eat('<');
            const element = this.union();
            let dialect = DEFAULT_DIALECT;
            if (this.peek()?.value === ',') {
                this.eat(',');
                const item = this.eat();
                if (item.kind !== 'str' && item.kind !== 'id')
                    throw new TypeSyntaxError('Neuralese<T, D> takes a dialect name or string');
                dialect = item.value;
            }
            this.eat('>');
            if (element.kind === 'neuralese')
                throw new TypeSyntaxError('neuralese-nested: Neuralese<Neuralese<T>> is not a type');
            return { kind: 'neuralese', element, dialect };
        }
        if (token.value === 'Is' && this.peek()?.value === '<') {
            // Is<T, "predicate">: one nonempty string literal; nesting means both predicates.
            this.eat('<');
            const base = this.union();
            if (this.peek()?.value !== ',')
                throw new TypeSyntaxError(refinementInvalid('a second type argument'));
            this.eat(',');
            const item = this.peek();
            if (item?.kind !== 'str')
                throw new TypeSyntaxError(refinementInvalid(item ? `${JSON.stringify(item.value)}, which is not a string literal` : 'nothing'));
            this.eat();
            const predicate = normalizePredicate(item.value);
            if (!predicate)
                throw new TypeSyntaxError(refinementInvalid('an empty string'));
            this.eat('>');
            return { kind: 'refined', base, predicate };
        }
        if (token.value === 'Untrusted' && this.peek()?.value === '<')
            return { kind: 'untrusted', base: this.args(1)[0] };
        if (token.value === 'Live' && this.peek()?.value === '<') {
            // Live<"TypeScript text", "tag" | "class" | "shape" | "function" | "any", "detail">
            this.eat('<');
            const parts = [];
            while (true) {
                const item = this.eat();
                if (item.kind !== 'str')
                    throw new TypeSyntaxError('Live<...> takes string arguments');
                parts.push(item.value);
                if (this.peek()?.value === ',') {
                    this.eat(',');
                    continue;
                }
                break;
            }
            this.eat('>');
            const [name = 'object', kind = 'any', detail = ''] = parts;
            const contract = kind === 'tag' ? { kind, tag: detail } : kind === 'class' ? { kind, name: detail } :
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
                    if (item.value === '<')
                        depth++;
                    else if (item.value === '>')
                        depth--;
                } while (depth > 0);
                text += this.tokens.slice(start, this.index).map(item => item.kind === 'str' ? JSON.stringify(item.value) : item.value)
                    .join('').replace(/,/g, ', ');
            }
            return { kind: 'host', name: text, contract: { kind: 'tag', tag: token.value } };
        }
        return { kind: 'name', name: token.value };
    }
}
export function parseType(text) {
    if (typeof text !== 'string')
        throw new TypeSyntaxError('type must be a string');
    return new Parser(tokenize(text)).parse();
}
export function formatType(type) {
    switch (type.kind) {
        case 'prim':
        case 'name': return type.name;
        case 'lit': return typeof type.value === 'string' ? JSON.stringify(type.value) : String(type.value);
        case 'record': return type.fields.length ? `{ ${type.fields.map(f => `${/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(f.name) ? f.name : JSON.stringify(f.name)}${f.optional ? '?' : ''}: ${formatType(f.type)}`).join(', ')} }` : '{}';
        case 'list': return (type.element.kind === 'union' ? `(${formatType(type.element)})` : formatType(type.element)) + '[]';
        case 'dict': return `Record<string, ${formatType(type.element)}>`;
        case 'union': return type.members.map(formatType).join(' | ');
        case 'lambda': return `(${type.params.fields.map(field => `${field.name}${field.optional ? '?' : ''}: ` +
            formatType(field.type)).join(', ')}) => ${formatType(type.returns)}`;
        case 'host': return type.name;
        case 'intersection': return type.members.map(m => m.kind === 'union' ? `(${formatType(m)})` : formatType(m)).join(' & ');
        case 'index': return `${type.base.kind === 'union' || type.base.kind === 'intersection' ? `(${formatType(type.base)})` : formatType(type.base)}[${JSON.stringify(type.key)}]`;
        case 'refined': return `Is<${formatType(type.base)}, ${JSON.stringify(type.predicate)}>`;
        case 'untrusted': return `Untrusted<${formatType(type.base)}>`;
        case 'neuralese': return isAdapterType(type) ? 'Adapter' : `Neuralese<${formatType(type.element)}${type.dialect === DEFAULT_DIALECT ? '' :
            `, ${JSON.stringify(type.dialect)}`}>`;
    }
}
/** Child types of a type, for structural walks. */
function children(type) {
    switch (type.kind) {
        case 'record': return type.fields.map(field => field.type);
        case 'list':
        case 'dict':
        case 'neuralese': return [type.element];
        case 'refined':
        case 'untrusted': return [type.base];
        case 'union':
        case 'intersection': return type.members;
        case 'index': return [type.base];
        case 'lambda': return [type.params, type.returns];
        default: return [];
    }
}
export class TypeEnv {
    names;
    parent;
    /** Constructors used to check `class` host contracts; inherited by child environments. */
    classes;
    /** The runtime's dialect binding; inherited by child environments. Without one, `DefaultDialect` is unbound. */
    dialects;
    constructor(names = {}, parent) {
        this.names = names;
        this.parent = parent;
        this.classes = parent?.classes;
        this.dialects = parent?.dialects;
        for (const name of Object.keys(names))
            this.checkRecursiveFunction(name);
    }
    /**
     * Reject an alias that reaches itself through a function type (`type-recursive-function`): self-application
     * would make recursion expressible without a recursive definition. Recursive data types stay allowed.
     */
    checkRecursiveFunction(alias) {
        const visiting = new Set();
        const walk = (type, throughFunction) => {
            if (type.kind === 'name') {
                if (type.name === alias && throughFunction)
                    throw new TypeSyntaxError(`type-recursive-function: type ${alias} refers to itself through a function type`);
                const key = `${type.name}\0${throughFunction}`;
                if (visiting.has(key))
                    return;
                visiting.add(key);
                const found = this.lookup(type.name);
                if (found)
                    walk(found, throughFunction);
                return;
            }
            for (const child of children(type))
                walk(child, throughFunction || type.kind === 'lambda');
        };
        const own = this.names[alias];
        if (own)
            walk(own, false);
    }
    child(names) { return Object.keys(names).length ? new TypeEnv({ ...names }, this) : this; }
    lookup(name) { return this.names[name] ?? this.parent?.lookup(name); }
    /** The dialect a type's dialect name means here: `DefaultDialect` is the bound reader dialect, when there is one. */
    dialect(name) { return name === DEFAULT_DIALECT ? this.dialects?.dialect ?? name : name; }
    resolve(type) {
        const seen = new Set();
        for (;;) {
            if (type.kind === 'name') {
                if (seen.has(type.name))
                    throw new TypeSyntaxError(`type ${type.name} is defined only in terms of itself`);
                seen.add(type.name);
                const found = this.lookup(type.name);
                if (!found)
                    throw new TypeSyntaxError(`unknown type name ${type.name}`);
                type = found;
            }
            else if (type.kind === 'intersection')
                type = mergeIntersection(type.members, member => this.resolve(member));
            else if (type.kind === 'index')
                type = indexedType(this.resolve(type.base), type.key, member => this.resolve(member));
            else
                return type;
        }
    }
    checkNames(type) {
        if (type.kind === 'name') {
            if (!this.lookup(type.name))
                throw new TypeSyntaxError(`unknown type name ${type.name}`);
            return;
        }
        if (type.kind === 'record')
            for (const field of type.fields)
                this.checkNames(field.type);
        if (type.kind === 'list' || type.kind === 'dict')
            this.checkNames(type.element);
        if (type.kind === 'refined' || type.kind === 'untrusted')
            this.checkNames(type.base);
        if (type.kind === 'union' || type.kind === 'intersection')
            for (const member of type.members)
                this.checkNames(member);
        if (type.kind === 'index')
            this.checkNames(type.base);
        if (type.kind === 'lambda') {
            this.checkNames(type.params);
            this.checkNames(type.returns);
        }
        if (type.kind === 'neuralese') {
            this.checkNames(type.element);
            // A view of a view has no meaning distinct from a view.
            if (this.resolveQuiet(type.element)?.kind === 'neuralese')
                throw new TypeSyntaxError(`neuralese-nested: ${formatType(type)} nests a Neuralese type`);
        }
    }
    resolveQuiet(type) { try {
        return this.resolve(type);
    }
    catch {
        return undefined;
    } }
}
export function fitsType(source, target, env = new TypeEnv(), seen = new Set()) {
    if (target.kind === 'prim' && target.name === 'unknown')
        return true;
    if (source.kind === 'name' && target.kind === 'name') {
        const key = `${source.name}\0${target.name}`;
        if (source.name === target.name || seen.has(key))
            return true;
        seen.add(key);
    }
    const a = env.resolve(source), b = env.resolve(target);
    if (JSON.stringify(a) === JSON.stringify(b))
        return true;
    if (a.kind === 'union')
        return a.members.every(member => fitsType(member, b, env, seen));
    // Untrusted<T> is a T for crisp code; a plain value fits an Untrusted slot (it only loses trust, never gains it).
    if (b.kind === 'untrusted')
        return fitsType(a.kind === 'untrusted' ? a.base : a, b.base, env, seen);
    if (a.kind === 'untrusted')
        return fitsType(a.base, b, env, seen);
    // A refinement is forgotten going out; going in, the value must fit the base (the predicate is an obligation,
    // see fitObligations) and an already-refined value must carry every predicate of the target.
    if (b.kind === 'refined') {
        const wanted = refinementChain(b, env), have = a.kind === 'refined' ? refinementChain(a, env) : { base: a, predicates: [] };
        return fitsType(have.base, wanted.base, env, seen) && (!have.predicates.length || wanted.predicates.every(p => have.predicates.includes(p)));
    }
    if (a.kind === 'refined')
        return fitsType(a.base, b, env, seen);
    if (b.kind === 'union')
        return b.members.some(member => fitsType(a, member, env, seen));
    if (a.kind === 'lit' && b.kind === 'prim')
        return b.name === (typeof a.value === 'string' ? 'string' : 'number');
    if ((a.kind === 'list' || a.kind === 'dict') && a.kind === b.kind)
        return fitsType(a.element, b.element, env, seen);
    if (a.kind === 'record' && b.kind === 'record') {
        for (const field of b.fields) {
            const got = a.fields.find(f => f.name === field.name);
            if (!got) {
                if (!field.optional)
                    return false;
                continue;
            }
            if (!fitsType(got.type, field.type, env, seen))
                return false;
        }
        return a.fields.every(field => b.fields.some(other => other.name === field.name));
    }
    if (a.kind === 'host' && b.kind === 'host')
        return b.contract.kind === 'any' || JSON.stringify(a.contract) === JSON.stringify(b.contract);
    if (b.kind === 'host' && b.contract.kind === 'any')
        return true;
    if (a.kind === 'neuralese' && b.kind === 'neuralese')
        return env.dialect(a.dialect) === env.dialect(b.dialect) && fitsType(a.element, b.element, env, seen);
    if (a.kind === 'lambda' && b.kind === 'lambda')
        return fitsType(a.returns, b.returns, env, seen) && fitsType(b.params, a.params, env, seen);
    return false;
}
/** The predicates of a (possibly nested) refinement, outermost last, and the unrefined type beneath them. */
export function refinementChain(type, env) {
    const predicates = [];
    let current = env.resolve(type);
    while (current.kind === 'refined') {
        predicates.unshift(current.predicate);
        current = env.resolve(current.base);
    }
    return { base: current, predicates: [...new Set(predicates)] };
}
/** Whether `type` mentions a refinement anywhere (named types followed once each). */
export function containsRefinement(type, env, seen = new Set()) {
    if (type.kind === 'refined')
        return true;
    if (type.kind === 'name') {
        if (seen.has(type.name))
            return false;
        seen.add(type.name);
        const found = env.lookup(type.name);
        return !!found && containsRefinement(found, env, seen);
    }
    if (type.kind === 'neuralese' || type.kind === 'host')
        return false;
    return children(type).some(child => containsRefinement(child, env, seen));
}
/**
 * The check obligations of putting a value of `source` into `target`: the predicates of refined target positions that
 * the source does not already carry, by path. null when `source` does not fit `target` at all.
 */
export function fitObligations(source, target, env = new TypeEnv(), path = 'value') {
    if (!fitsType(source, target, env))
        return null;
    const a = env.resolve(source), b = env.resolve(target);
    if (b.kind === 'untrusted' || a.kind === 'untrusted')
        return fitObligations(a.kind === 'untrusted' ? a.base : a, b.kind === 'untrusted' ? b.base : b, env, path);
    if (b.kind === 'refined') {
        const wanted = refinementChain(b, env), have = refinementChain(a, env);
        return wanted.predicates.filter(p => !have.predicates.includes(p)).map(predicate => ({ path, predicate }));
    }
    const out = [];
    const sub = (x, y, at) => { out.push(...fitObligations(x, y, env, at) ?? []); };
    if ((a.kind === 'list' && b.kind === 'list') || (a.kind === 'dict' && b.kind === 'dict'))
        sub(a.element, b.element, `${path}/*`);
    else if (a.kind === 'record' && b.kind === 'record')
        for (const field of b.fields) {
            const got = a.fields.find(f => f.name === field.name);
            if (got)
                sub(got.type, field.type, `${path}/${field.name}`);
        }
    return out;
}
/**
 * Crisp checkers for refinement predicates, by normalized predicate text. A checker that returns a boolean decides;
 * `undefined` defers to the judge (plans/REFINEMENT_TYPES.md section 3). Register more with `refinements[predicate] = fn`;
 * a runtime's `refinements.crisp` setting is consulted before this table.
 */
export const refinements = Object.create(null);
/** The field `key` of a record (an optional field also admits null), or a dictionary's element. */
function indexedType(base, key, resolve) {
    const resolved = resolve(base);
    if (resolved.kind === 'record') {
        const field = resolved.fields.find(item => item.name === key);
        if (!field)
            throw new TypeSyntaxError(`${formatType(resolved)} has no field ${JSON.stringify(key)}; index it with one of its field names`);
        return field.optional ? { kind: 'union', members: [field.type, { kind: 'prim', name: 'null' }] } : field.type;
    }
    if (resolved.kind === 'dict')
        return resolved.element;
    throw new TypeSyntaxError(`${formatType(resolved)}[${JSON.stringify(key)}] indexes a record or dictionary; index a record type, or name the field's type directly`);
}
/** One type from `A & B`: records merge field by field, dictionaries merge their elements, equal types are that type. */
function mergeIntersection(members, resolve) {
    const parts = members.map(resolve);
    if (parts.every(part => part.kind === 'record')) {
        const fields = [];
        for (const part of parts)
            for (const field of part.fields) {
                const at = fields.findIndex(item => item.name === field.name);
                if (at < 0) {
                    fields.push({ ...field });
                    continue;
                }
                const had = fields[at];
                fields[at] = { name: field.name, optional: had.optional && field.optional,
                    type: JSON.stringify(had.type) === JSON.stringify(field.type) ? had.type : { kind: 'intersection', members: [had.type, field.type] } };
            }
        return { kind: 'record', fields };
    }
    if (parts.every(part => part.kind === 'dict'))
        return { kind: 'dict', element: parts.length === 1 ? parts[0].element :
                { kind: 'intersection', members: parts.map(part => part.element) } };
    if (parts.every(part => JSON.stringify(part) === JSON.stringify(parts[0])))
        return parts[0];
    throw new TypeSyntaxError(`${members.map(formatType).join(' & ')} intersects types that are not all objects; intersect record types, or write one record with the fields you need`);
}
