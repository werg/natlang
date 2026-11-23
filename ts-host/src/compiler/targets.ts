import ts from 'typescript';
import { DEFAULT_DIALECT, neuraleseParts } from './neuralese.js';

/** Runtime contract for a value that is checked as a live host object rather than portable data. */
export type HostContract =
  | { kind: 'tag'; tag: string }                 // built-ins, checked with Object.prototype.toString across realms
  | { kind: 'class'; name: string }              // instanceof against a constructor reachable at the call site
  | { kind: 'shape'; members: string[] }         // structural: required members exist
  | { kind: 'function' }
  | { kind: 'folder' } | { kind: 'file' };

/**
 * A resolved TypeScript target. `natlang` is the portable type text understood by the
 * interpreter's checker; when absent, `host` gives the live-object contract.
 */
export type TargetDescriptor = {
  text: string;
  natlang?: string;
  aliases: Record<string, string>;
  host?: HostContract;
};

/** Parse one exact TypeScript primitive literal descriptor; reject unions and non-primitive literals. */
export function portablePrimitiveLiteral(descriptor: string):
  { type: 'string' | 'number' | 'boolean'; value: string | number | boolean } | undefined {
  if (descriptor === 'true' || descriptor === 'false') return { type: 'boolean', value: descriptor === 'true' };
  if (/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(descriptor)) {
    const value = Number(descriptor);
    return Number.isFinite(value) && !Object.is(value, -0) && String(value) === descriptor ? { type: 'number', value } : undefined;
  }
  try {
    const value: unknown = JSON.parse(descriptor);
    return typeof value === 'string' && JSON.stringify(value) === descriptor ? { type: 'string', value } : undefined;
  } catch { return undefined; }
}

export class TargetError extends Error {}

const BUILTIN_TAGS: Record<string, string> = {
  Date: 'Date', Map: 'Map', Set: 'Set', WeakMap: 'WeakMap', WeakSet: 'WeakSet', RegExp: 'RegExp', Error: 'Error',
  URL: 'URL', URLSearchParams: 'URLSearchParams', Uint8Array: 'Uint8Array', ArrayBuffer: 'ArrayBuffer',
  Headers: 'Headers', Response: 'Response', Request: 'Request',
};

const isLibFile = (file: ts.SourceFile, program: ts.Program) =>
  program.isSourceFileDefaultLibrary(file) || program.isSourceFileFromExternalLibrary(file) ||
  /[\\/]typescript[\\/]lib[\\/]/.test(file.fileName);
const isIntrinsicFile = (file: ts.SourceFile) => file.text.includes('__natlangUnspecified');
const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;
const RESERVED_NAMES = new Set(['string', 'number', 'boolean', 'null', 'Blob', 'Folder', 'FileHandle', 'Record']);

/** Unwrap `Promise<T>` (and unions containing promises) to the awaited value type. */
export function awaitedType(checker: ts.TypeChecker, type: ts.Type): { type: ts.Type; promised: boolean } {
  const promised = (type.isUnion() ? type.types : [type]).some(member => isPromiseLike(checker, member));
  const awaited = (checker as ts.TypeChecker & { getAwaitedType?(type: ts.Type): ts.Type | undefined })
    .getAwaitedType?.(type) ?? type;
  return { type: awaited, promised };
}

export function isPromiseLike(checker: ts.TypeChecker, type: ts.Type): boolean {
  const then = type.getProperty('then');
  if (!then) return false;
  const declaration = then.valueDeclaration ?? then.declarations?.[0];
  const thenType = declaration ? checker.getTypeOfSymbolAtLocation(then, declaration) : undefined;
  return !!thenType?.getCallSignatures().length;
}

/** Convert a checked TypeScript type into a target descriptor, or throw a TargetError explaining why not. */
export function describeTarget(program: ts.Program, checker: ts.TypeChecker, type: ts.Type,
  options: { allowHost?: boolean; location?: ts.Node } = {}): TargetDescriptor {
  const aliases: Record<string, string> = {};
  const pending = new Set<string>();
  const text = checker.typeToString(type, options.location, ts.TypeFormatFlags.NoTruncation |
    ts.TypeFormatFlags.UseAliasDefinedOutsideCurrentScope);

  const namedUserSymbol = (candidate: ts.Type): ts.Symbol | undefined => {
    const symbol = candidate.aliasSymbol && !candidate.aliasTypeArguments?.length ? candidate.aliasSymbol :
      candidate.getSymbol() && (candidate.getSymbol()!.flags & ts.SymbolFlags.Interface) &&
        !(candidate as ts.TypeReference).typeArguments?.length ? candidate.getSymbol() : undefined;
    if (!symbol) return;
    const declaration = symbol.declarations?.[0];
    if (!declaration) return;
    const file = declaration.getSourceFile();
    if (isLibFile(file, program) || isIntrinsicFile(file)) return;
    if (!IDENTIFIER.test(symbol.name) || RESERVED_NAMES.has(symbol.name)) return;
    return symbol;
  };

  const hostOf = (candidate: ts.Type): HostContract | undefined => {
    const symbol = candidate.getSymbol() ?? candidate.aliasSymbol;
    const declaration = symbol?.declarations?.[0];
    if (symbol && declaration && isIntrinsicFile(declaration.getSourceFile())) {
      if (symbol.name === 'Folder') return { kind: 'folder' };
      if (symbol.name === 'FileHandle') return { kind: 'file' };
    }
    if (candidate.getCallSignatures().length) return { kind: 'function' };
    if (symbol && declaration && isLibFile(declaration.getSourceFile(), program) && BUILTIN_TAGS[symbol.name])
      return { kind: 'tag', tag: BUILTIN_TAGS[symbol.name]! };
    if (symbol && (symbol.flags & ts.SymbolFlags.Class)) return { kind: 'class', name: symbol.name };
    const methods = candidate.getProperties().filter(property => {
      const declared = property.valueDeclaration ?? property.declarations?.[0];
      return !!declared && checker.getTypeOfSymbolAtLocation(property, declared).getCallSignatures().length > 0;
    });
    if (methods.length) return { kind: 'shape', members: candidate.getProperties().map(property => property.name) };
    return;
  };

  const INTERSECTION_RULE = 'an intersection is portable only as object types joined with & (Base & { extra: string }), ' +
    'or inside Is<T, P> or Untrusted<T>';
  /** The parts of an intersection that are not natlang brands, and the brands: \`Is<T, P>\` predicates and \`Untrusted<T>\`. */
  const splitBrands = (candidate: ts.IntersectionType): { rest: ts.Type[]; predicates: string[]; untrusted: boolean } => {
    let untrusted = false;
    const predicates: string[] = [], rest: ts.Type[] = [];
    for (const part of candidate.types) {
      const properties = part.getProperties();
      const only = properties.length === 1 ? properties[0]! : undefined;
      if (only?.name === '__natlangUntrusted') { untrusted = true; continue; }
      if (only?.name !== '__natlangRefinement') { rest.push(part); continue; }
      const declaration = only.valueDeclaration ?? only.declarations?.[0];
      const map = declaration ? checker.getTypeOfSymbolAtLocation(only, declaration) : checker.getTypeOfSymbol(only);
      const names = map.getProperties().map(item => item.name);
      // Is<T, string> (or any P that is not string literals) names no predicate: the frontmatter parser's error.
      if (!names.length || names.some(name => !name.trim())) throw new TargetError('refinement-predicate-invalid: Is<T, P> takes a ' +
        'nonempty string literal as P, for example Is<string, "one line of at most 60 characters">');
      for (const predicate of names) if (!predicates.includes(predicate)) predicates.push(predicate);
    }
    return { rest, predicates, untrusted };
  };
  const wrapBrands = (inner: string, predicates: readonly string[], untrusted: boolean): string => {
    for (const predicate of predicates) inner = `Is<${inner}, ${JSON.stringify(predicate)}>`;
    return untrusted ? `Untrusted<${inner}>` : inner;
  };
  const unionText = (members: readonly ts.Type[], depth: number): string => {
    const hasTrue = members.some(t => t.flags & ts.TypeFlags.BooleanLiteral && checker.typeToString(t) === 'true');
    const hasFalse = members.some(t => t.flags & ts.TypeFlags.BooleanLiteral && checker.typeToString(t) === 'false');
    const parts: string[] = [];
    if (hasTrue && hasFalse) parts.push('boolean');
    for (const member of members) {
      if (member.flags & ts.TypeFlags.BooleanLiteral && hasTrue && hasFalse) continue;
      const converted = convert(member, depth + 1);
      if (!parts.includes(converted)) parts.push(converted);
    }
    return parts.map(part => part.includes('=>') ? `(${part})` : part).join(' | ');
  };

  const convert = (candidate: ts.Type, depth: number): string => {
    if (depth > 24) throw new TargetError('the type is nested too deeply to check');
    const flags = candidate.flags;
    // Nothing says what an any or unknown part is, so it is open: its value is kept as it is, as a whole target
    // typed any is (compiler/inline.ts).
    if (flags & (ts.TypeFlags.Any | ts.TypeFlags.Unknown)) return 'unknown';
    // A soft value is portable as a reference; its element may be a function type (an applicable soft function).
    const soft = neuraleseParts(checker, candidate);
    if (soft) {
      const signature = soft.element.getCallSignatures()[0];
      const element = signature ? `(${signature.getParameters().map(parameter => {
        const declaration = parameter.valueDeclaration ?? parameter.declarations?.[0];
        const parameterType = declaration ? checker.getTypeOfSymbolAtLocation(parameter, declaration) : checker.getTypeOfSymbol(parameter);
        return `${parameter.name}: ${convert(parameterType, depth + 1)}`;
      }).join(', ')}) => ${convert(awaitedType(checker, checker.getReturnTypeOfSignature(signature)).type, depth + 1)}` :
        convert(soft.element, depth + 1);
      return `Neuralese<${element}${soft.dialect === DEFAULT_DIALECT ? '' : `, ${JSON.stringify(soft.dialect)}`}>`;
    }
    if (flags & ts.TypeFlags.Never) throw new TargetError('the type is `never`');
    if (flags & ts.TypeFlags.TypeParameter) throw new TargetError('the type is an unresolved generic parameter');
    if (flags & (ts.TypeFlags.BigInt | ts.TypeFlags.BigIntLiteral | ts.TypeFlags.ESSymbolLike))
      throw new TargetError('bigint and symbol values are not supported targets');
    const named = namedUserSymbol(candidate);
    if (named && !(flags & ts.TypeFlags.Union && !candidate.aliasSymbol) && !hostOf(candidate)) {
      const name = named.name;
      if (!Object.hasOwn(aliases, name) && !pending.has(name)) {
        pending.add(name);
        const expanded = named.flags & ts.SymbolFlags.TypeAlias ? checker.getDeclaredTypeOfSymbol(named) : candidate;
        aliases[name] = convertStructure(expanded, depth + 1);
        pending.delete(name);
      }
      return name;
    }
    return convertStructure(candidate, depth);
  };

  /**
   * An asynchronous function type with one plain signature and portable parameter and result types, such as
   * `(t: Ticket) => Promise<boolean>` or a natlang function `NatlangFunction<[Ticket], boolean>`: the natlang function
   * type `(t: Ticket) => boolean`, whose values the runtime checks by signature. Any other function stays a live host
   * function (`Live<..., "function">`), passed by identity.
   */
  const asyncFunctionText = (candidate: ts.Type, depth: number): string | undefined => {
    if (candidate.isUnion() || !(candidate.flags & ts.TypeFlags.Object || candidate.isIntersection())) return;
    const signatures = candidate.getCallSignatures();
    const signature = signatures.length === 1 ? signatures[0]! : undefined;
    if (!signature || signature.typeParameters?.length || candidate.getConstructSignatures().length) return;
    const returned = checker.getReturnTypeOfSignature(signature);
    if (!(returned.isUnion() ? returned.types : [returned]).some(member => isPromiseLike(checker, member))) return;
    const saved = { ...aliases };
    try {
      const parameters = signature.getParameters().map(parameter => {
        const declaration = parameter.valueDeclaration ?? parameter.declarations?.[0];
        if (declaration && ts.isParameter(declaration) && declaration.dotDotDotToken) throw new TargetError('a rest parameter');
        const optional = !!declaration && ts.isParameter(declaration) && (!!declaration.questionToken || !!declaration.initializer);
        let type = declaration ? checker.getTypeOfSymbolAtLocation(parameter, declaration) : checker.getTypeOfSymbol(parameter);
        if (optional) type = checker.getNonNullableType(type);
        const name = IDENTIFIER.test(parameter.name) ? parameter.name : `input${signature.getParameters().indexOf(parameter) + 1}`;
        return `${name}${optional ? '?' : ''}: ${convert(type, depth + 1)}`;
      });
      return `(${parameters.join(', ')}) => ${convert(awaitedType(checker, returned).type, depth + 1)}`;
    } catch (error) {
      if (!(error instanceof TargetError) && !(error instanceof HostTarget)) throw error;
      for (const name of Object.keys(aliases)) if (!Object.hasOwn(saved, name)) delete aliases[name];
      return;
    }
  };

  const convertStructure = (candidate: ts.Type, depth: number): string => {
    const flags = candidate.flags;
    if (flags & ts.TypeFlags.StringLiteral) return JSON.stringify((candidate as ts.StringLiteralType).value);
    if (flags & ts.TypeFlags.NumberLiteral) return String((candidate as ts.NumberLiteralType).value);
    if (flags & (ts.TypeFlags.String | ts.TypeFlags.TemplateLiteral | ts.TypeFlags.StringMapping)) return 'string';
    if (flags & ts.TypeFlags.Number) return 'number';
    if (flags & (ts.TypeFlags.Boolean | ts.TypeFlags.BooleanLiteral)) return 'boolean';
    if (flags & (ts.TypeFlags.Null | ts.TypeFlags.Undefined | ts.TypeFlags.Void)) return 'null';
    if (flags & ts.TypeFlags.EnumLiteral && candidate.isUnion()) return candidate.types.map(t => convert(t, depth + 1)).join(' | ');
    const lambda = asyncFunctionText(candidate, depth);
    if (lambda) return lambda;
    if (candidate.isUnion()) {
      // TypeScript distributes a brand over a union (\`Is<"a" | "b", P>\` is \`("a" & brand) | ("b" & brand)\`, and
      // \`Is<boolean, P>\` brands \`true\` and \`false\` apart): one brand on every member is the brand of the union.
      const split = candidate.types.map(member => member.isIntersection() ? splitBrands(member) : undefined);
      const first = split[0];
      const key = (item: typeof first) => item && JSON.stringify([[...item.predicates].sort(), item.untrusted]);
      if (first && (first.predicates.length || first.untrusted) && split.every(item => item && item.rest.length === 1 && key(item) === key(first)))
        return wrapBrands(unionText(split.map(item => item!.rest[0]!), depth), first.predicates, first.untrusted);
      return unionText(candidate.types, depth);
    }
    if (candidate.isIntersection()) {
      // Unwrap \`Untrusted<T>\` (\`__natlangUntrusted\`) and \`Is<T, P>\` (\`__natlangRefinement\`) brand intersections,
      // and support record/object intersections (\`A & { extra: string }\`) matching \`native/types.ts\`.
      const { rest, predicates, untrusted } = splitBrands(candidate);
      if (!rest.length) throw new TargetError(INTERSECTION_RULE);
      let inner: string;
      if (rest.length === 1) {
        inner = convert(rest[0]!, depth + 1);
      } else {
        const isRecordMember = (part: ts.Type): boolean =>
          !!(part.flags & ts.TypeFlags.Object) && !checker.isArrayType(part) && !checker.isTupleType(part) &&
          !part.isUnion() && !hostOf(part) && !neuraleseParts(checker, part);
        if (!rest.every(isRecordMember)) throw new TargetError(INTERSECTION_RULE);
        inner = rest.map(part => convert(part, depth + 1)).join(' & ');
      }
      return wrapBrands(inner, predicates, untrusted);
    }
    if (checker.isArrayType(candidate) || checker.isTupleType(candidate)) {
      const elements = checker.isArrayType(candidate) ? [checker.getTypeArguments(candidate as ts.TypeReference)[0]!] :
        checker.getTypeArguments(candidate as ts.TypeReference);
      const converted = [...new Set(elements.map(element => convert(element, depth + 1)))];
      if (!converted.length) throw new TargetError('an empty tuple is not a useful target');
      return `(${converted.join(' | ')})[]`;
    }
    if (flags & ts.TypeFlags.Object || flags & ts.TypeFlags.NonPrimitive) {
      const host = hostOf(candidate);
      if (host) {
        if (host.kind === 'folder') return 'Folder';
        if (host.kind === 'file') return 'FileHandle';
        throw new HostTarget(host, depth);
      }
      const properties = checker.getPropertiesOfType(candidate);
      const stringIndex = checker.getIndexInfoOfType(candidate, ts.IndexKind.String);
      if (stringIndex && !properties.length) return `Record<string, ${convert(stringIndex.type, depth + 1)}>`;
      if (stringIndex) throw new TargetError('records that mix fixed fields and an index signature are not supported');
      // TypeScript's {} (also produced by unknown!) accepts primitives. It is
      // not an empty portable record. The portable surface has no non-nullish
      // top type, so preserve the open value rather than require an object.
      if (!properties.length && checker.isTypeAssignableTo(checker.getStringType(), candidate)) return 'unknown';
      const fields = properties.map(property => {
        const declaration = property.valueDeclaration ?? property.declarations?.[0];
        let fieldType = declaration ? checker.getTypeOfSymbolAtLocation(property, declaration) : checker.getTypeOfSymbol(property);
        const optional = !!(property.flags & ts.SymbolFlags.Optional);
        if (optional) fieldType = checker.getNonNullableType(fieldType);
        return `${IDENTIFIER.test(property.name) ? property.name : JSON.stringify(property.name)}${optional ? '?' : ''}: ${convert(fieldType, depth + 1)}`;
      });
      return fields.length ? `{ ${fields.join(', ')} }` : '{}';
    }
    throw new TargetError(`unsupported type ${checker.typeToString(candidate)}`);
  };

  try {
    return { text, natlang: convert(type, 0), aliases };
  } catch (error) {
    if (error instanceof HostTarget) {
      if (!options.allowHost) throw new TargetError(`${text} contains a host object, which is not portable data`);
      if (error.depth === 0) return { text, aliases: {}, host: error.contract };
      if (type.getCallSignatures().length) return { text, aliases: {}, host: { kind: 'function' } };
      return { text, aliases: {}, host: { kind: 'shape', members: type.getProperties().map(property => property.name) } };
    }
    throw error;
  }
}

class HostTarget extends Error {
  constructor(readonly contract: HostContract, readonly depth: number) { super('host target'); }
}
