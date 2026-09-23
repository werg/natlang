import ts from 'typescript';

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

  const convert = (candidate: ts.Type, depth: number): string => {
    if (depth > 24) throw new TargetError('the type is nested too deeply to check');
    const flags = candidate.flags;
    if (flags & ts.TypeFlags.Any) throw new TargetError('the type resolves to `any`');
    if (flags & ts.TypeFlags.Unknown) throw new TargetError('the type is unconstrained `unknown`');
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

  const convertStructure = (candidate: ts.Type, depth: number): string => {
    const flags = candidate.flags;
    if (flags & ts.TypeFlags.StringLiteral) return JSON.stringify((candidate as ts.StringLiteralType).value);
    if (flags & ts.TypeFlags.NumberLiteral) return String((candidate as ts.NumberLiteralType).value);
    if (flags & (ts.TypeFlags.String | ts.TypeFlags.TemplateLiteral | ts.TypeFlags.StringMapping)) return 'string';
    if (flags & ts.TypeFlags.Number) return 'number';
    if (flags & (ts.TypeFlags.Boolean | ts.TypeFlags.BooleanLiteral)) return 'boolean';
    if (flags & (ts.TypeFlags.Null | ts.TypeFlags.Undefined | ts.TypeFlags.Void)) return 'null';
    if (flags & ts.TypeFlags.EnumLiteral && candidate.isUnion()) return candidate.types.map(t => convert(t, depth + 1)).join(' | ');
    if (candidate.isUnion()) {
      const members = candidate.types;
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
    }
    if (candidate.isIntersection()) throw new TargetError('intersection types are not supported as portable targets');
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
      const fields = properties.map(property => {
        if (!IDENTIFIER.test(property.name)) throw new TargetError(`field name ${JSON.stringify(property.name)} is not an identifier`);
        const declaration = property.valueDeclaration ?? property.declarations?.[0];
        let fieldType = declaration ? checker.getTypeOfSymbolAtLocation(property, declaration) : checker.getTypeOfSymbol(property);
        const optional = !!(property.flags & ts.SymbolFlags.Optional);
        if (optional) fieldType = checker.getNonNullableType(fieldType);
        return `${property.name}${optional ? '?' : ''}: ${convert(fieldType, depth + 1)}`;
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
