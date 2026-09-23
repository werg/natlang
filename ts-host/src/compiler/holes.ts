/**
 * Local result-type inference for `nl` expressions that have no annotation and no contextual type.
 *
 * Each such expression gets a unique placeholder type (a typed hole) in a second checked program. The hole
 * flows through ordinary TypeScript (locals, arrays, `Promise.all`, element access, `for...of`), and every
 * use of a hole-typed value adds a requirement: the type TypeScript expects there, or what the syntax
 * implies where TypeScript expects nothing (a condition needs a boolean, arithmetic needs a number). A hole
 * whose requirements agree gets that type. This is inference within one file, as in Rust or ML: the
 * requirement comes from a use in the same snippet, so a disagreement can name both uses.
 */
import ts from 'typescript';
import { awaitedType, isPromiseLike } from './targets.js';

const HOLE = '__NlHole';
/** The function an eval snippet is checked inside (compiler/eval-check.ts). */
const EVAL_WRAPPER = '__natlang_scope';

export type HoleSite = { id: number; node: ts.TaggedTemplateExpression };

/** One requirement on a hole, with the use that imposed it (a position in the original file). */
export type HoleUse = { type: ts.Type; typeText: string; strong: boolean; position: number; text: string; why: string };

export type HoleSolution =
  | { kind: 'solved'; type: ts.Type; program: ts.Program; location: ts.Node }
  | { kind: 'ambiguous'; uses: HoleUse[] }
  | { kind: 'unknown'; fields: string[]; indexed: boolean };

/** Methods that only strings (or only numbers) have: calling one on a result says what the result is. */
const STRING_METHODS = new Set(['toLowerCase', 'toUpperCase', 'trim', 'trimStart', 'trimEnd', 'split', 'startsWith',
  'endsWith', 'replace', 'replaceAll', 'match', 'matchAll', 'padStart', 'padEnd', 'charAt', 'charCodeAt', 'localeCompare',
  'normalize', 'toLocaleLowerCase', 'toLocaleUpperCase']);
const NUMBER_METHODS = new Set(['toFixed', 'toPrecision', 'toExponential']);
/** Array methods whose callback result is tested for truthiness. */
const PREDICATE_METHODS = new Set(['filter', 'some', 'every', 'find', 'findIndex', 'findLast', 'findLastIndex']);
const ARITHMETIC = new Set([ts.SyntaxKind.MinusToken, ts.SyntaxKind.AsteriskToken, ts.SyntaxKind.SlashToken,
  ts.SyntaxKind.PercentToken, ts.SyntaxKind.AsteriskAsteriskToken, ts.SyntaxKind.MinusEqualsToken,
  ts.SyntaxKind.AsteriskEqualsToken, ts.SyntaxKind.SlashEqualsToken, ts.SyntaxKind.PercentEqualsToken]);
const RELATIONAL = new Set([ts.SyntaxKind.LessThanToken, ts.SyntaxKind.GreaterThanToken,
  ts.SyntaxKind.LessThanEqualsToken, ts.SyntaxKind.GreaterThanEqualsToken]);
const EQUALITY = new Set([ts.SyntaxKind.EqualsEqualsEqualsToken, ts.SyntaxKind.ExclamationEqualsEqualsToken,
  ts.SyntaxKind.EqualsEqualsToken, ts.SyntaxKind.ExclamationEqualsToken]);

/** Solve the result type of each site. Returns undefined when the hole program cannot be built. */
export function solveHoles(program: ts.Program, sites: readonly HoleSite[]): Map<number, HoleSolution> | undefined {
  const byFile = new Map<ts.SourceFile, HoleSite[]>();
  for (const site of sites) {
    const file = site.node.getSourceFile();
    byFile.set(file, [...(byFile.get(file) ?? []), site]);
  }
  // Insert `<__NlHoleK>` after each tag and declare the hole types at the end of the file.
  const texts = new Map<string, string>();
  const inserts = new Map<string, { at: number; length: number }[]>();
  for (const [file, fileSites] of byFile) {
    const ordered = [...fileSites].sort((a, b) => a.node.tag.getEnd() - b.node.tag.getEnd());
    let text = '', last = 0;
    const shifts: { at: number; length: number }[] = [];
    for (const site of ordered) {
      const at = site.node.tag.getEnd(), piece = `<${HOLE}${site.id}>`;
      text += file.text.slice(last, at) + piece;
      shifts.push({ at, length: piece.length });
      last = at;
    }
    text += file.text.slice(last) + '\n' + ordered.map(site => `interface ${HOLE}${site.id} { readonly ${HOLE}${site.id}: true }`).join('\n') + '\n';
    texts.set(file.fileName, text);
    inserts.set(file.fileName, shifts);
  }
  const holed = rebuild(program, texts);
  if (!holed) return;
  const checker = holed.getTypeChecker();
  const uses = new Map<number, HoleUse[]>();
  const fields = new Map<number, Set<string>>();
  const indexed = new Set<number>();

  for (const [fileName, text] of texts) {
    const file = holed.getSourceFile(fileName);
    if (!file || file.text !== text) return;
    const shifts = inserts.get(fileName)!;
    const original = (position: number) => {
      let shifted = 0;
      for (const shift of shifts) {
        const start = shift.at + shifted;
        if (position < start) break;
        if (position < start + shift.length) return shift.at;
        shifted += shift.length;
      }
      return position - shifted;
    };
    const require = (node: ts.Node, type: ts.Type, strong: boolean, why: string, id: number) => {
      // Quote the statement that makes the use: line numbers differ between an eval and its checked wrapper.
      const statement = ts.findAncestor(node, candidate => ts.isStatement(candidate) && !ts.isBlock(candidate)) ?? node;
      const quoted = statement.getText().split('\n')[0]!.replace(/\s+/g, ' ').trim();
      uses.set(id, [...(uses.get(id) ?? []), { type, typeText: checker.typeToString(type), strong, position: original(node.getStart()), why,
        text: quoted.length > 60 ? `${quoted.slice(0, 57)}...` : quoted }]);
    };
    const visit = (node: ts.Node): void => {
      if (ts.isExpression(node) && !isBindingName(node) && !isPropertyName(node) && !ts.isTaggedTemplateExpression(node.parent)) {
        const type = checker.getTypeAtLocation(node);
        if (holesIn(checker, type).size) collect(node, type);
      }
      ts.forEachChild(node, visit);
    };
    const collect = (node: ts.Expression, type: ts.Type): void => {
      // What TypeScript expects at this position, matched against the value's structure.
      const expected = checker.getContextualType(node);
      if (expected) unify(checker, type, expected, (id, target) => require(node, target, true, 'expected here', id));
      const hole = holeId(type);
      if (hole === undefined) return;
      const outer = skipParentheses(node);
      const parent = outer.parent;
      if (inTruthinessPosition(outer)) require(node, checker.getBooleanType(), true, 'tested as a condition', hole);
      if (ts.isPrefixUnaryExpression(parent) && [ts.SyntaxKind.MinusToken, ts.SyntaxKind.PlusToken, ts.SyntaxKind.TildeToken,
        ts.SyntaxKind.PlusPlusToken, ts.SyntaxKind.MinusMinusToken].includes(parent.operator))
        require(node, checker.getNumberType(), true, 'used as a number', hole);
      if (ts.isBinaryExpression(parent)) {
        const other = parent.left === outer ? parent.right : parent.left;
        const otherType = checker.getTypeAtLocation(other);
        const kind = parent.operatorToken.kind;
        if (ARITHMETIC.has(kind)) require(node, checker.getNumberType(), true, 'used as a number', hole);
        else if (kind === ts.SyntaxKind.PlusToken || kind === ts.SyntaxKind.PlusEqualsToken || RELATIONAL.has(kind)) {
          if (otherType.flags & ts.TypeFlags.NumberLike) require(node, checker.getNumberType(), true, 'used as a number', hole);
          else if (otherType.flags & ts.TypeFlags.StringLike) require(node, checker.getStringType(), false, 'used as text', hole);
        } else if (EQUALITY.has(kind) && !holesIn(checker, otherType).size) {
          const base = baseType(checker, otherType);
          if (base) require(node, base, false, 'compared with a value', hole);
        }
      }
      if ((ts.isTemplateSpan(parent) && parent.expression === outer) || isDisplayed(outer))
        require(node, checker.getStringType(), false, 'used as text', hole);
      if (ts.isSwitchStatement(parent) && parent.expression === outer) {
        for (const clause of parent.caseBlock.clauses) if (ts.isCaseClause(clause)) {
          const base = baseType(checker, checker.getTypeAtLocation(clause.expression));
          if (base) require(clause.expression, base, false, 'compared with a value', hole);
        }
      }
      if (ts.isPropertyAccessExpression(parent) && parent.expression === outer) {
        const name = parent.name.text;
        const called = ts.isCallExpression(parent.parent) && parent.parent.expression === parent;
        if (called && STRING_METHODS.has(name)) require(node, checker.getStringType(), true, `given .${name}()`, hole);
        else if (called && NUMBER_METHODS.has(name)) require(node, checker.getNumberType(), true, `given .${name}()`, hole);
        else fields.set(hole, new Set([...(fields.get(hole) ?? []), name]));
      }
      if (ts.isVariableDeclaration(parent) && parent.initializer === outer && ts.isObjectBindingPattern(parent.name))
        for (const element of parent.name.elements) {
          const name = element.propertyName ?? element.name;
          if (ts.isIdentifier(name)) fields.set(hole, new Set([...(fields.get(hole) ?? []), name.text]));
        }
      if ((ts.isElementAccessExpression(parent) && parent.expression === outer) ||
        (ts.isForOfStatement(parent) && parent.expression === outer)) indexed.add(hole);
    };
    visit(file);
  }

  const solutions = new Map<number, HoleSolution>();
  for (const site of sites) {
    const found = uses.get(site.id) ?? [];
    const strong = found.filter(use => use.strong);
    const candidates = strong.length ? strong : found;
    if (!candidates.length) {
      solutions.set(site.id, { kind: 'unknown', fields: [...(fields.get(site.id) ?? [])], indexed: indexed.has(site.id) });
      continue;
    }
    // The most specific requirement that satisfies all of them: `"low" | "high"` over `string`.
    const best = candidates.find(use => candidates.every(other => checker.isTypeAssignableTo(use.type, other.type)));
    if (!best) {
      const distinct: HoleUse[] = [];
      for (const use of candidates) if (!distinct.some(seen => checker.isTypeAssignableTo(seen.type, use.type) &&
        checker.isTypeAssignableTo(use.type, seen.type))) distinct.push(use);
      solutions.set(site.id, { kind: 'ambiguous', uses: distinct });
      continue;
    }
    solutions.set(site.id, { kind: 'solved', type: best.type, program: holed, location: holed.getSourceFile(site.node.getSourceFile().fileName)! });
  }
  return solutions;
}

/** The same program with some files' text replaced; everything else, including module resolution, is reused. */
function rebuild(program: ts.Program, texts: Map<string, string>): ts.Program | undefined {
  const options = program.getCompilerOptions();
  const defaultLib = program.getSourceFiles().find(file => program.isSourceFileDefaultLibrary(file))?.fileName;
  const libDir = defaultLib ? defaultLib.slice(0, defaultLib.lastIndexOf('/')) : '/';
  const known = (fileName: string) => texts.get(fileName) ?? program.getSourceFile(fileName)?.text;
  const host: ts.CompilerHost = {
    getSourceFile: (fileName, version) => texts.has(fileName) ?
      ts.createSourceFile(fileName, texts.get(fileName)!, version, true) : program.getSourceFile(fileName),
    getDefaultLibFileName: compilerOptions => `${libDir}/${ts.getDefaultLibFileName(compilerOptions)}`,
    getDefaultLibLocation: () => libDir,
    writeFile: () => undefined,
    getCurrentDirectory: () => program.getCurrentDirectory(),
    getDirectories: () => [],
    getCanonicalFileName: name => name,
    useCaseSensitiveFileNames: () => true,
    getNewLine: () => '\n',
    fileExists: fileName => known(fileName) !== undefined,
    readFile: known,
  };
  try {
    return ts.createProgram({ rootNames: program.getRootFileNames(), options, host, oldProgram: program });
  } catch {
    return;
  }
}

function holeId(type: ts.Type): number | undefined {
  const name = type.getSymbol()?.name;
  return name?.startsWith(HOLE) ? Number(name.slice(HOLE.length)) : undefined;
}

/** The holes a type mentions: itself, union members, type arguments, fields, and call results. */
function holesIn(checker: ts.TypeChecker, type: ts.Type, depth = 0, found = new Set<number>()): Set<number> {
  if (depth > 4) return found;
  const id = holeId(type);
  if (id !== undefined) { found.add(id); return found; }
  if (type.isUnionOrIntersection()) for (const member of type.types) holesIn(checker, member, depth + 1, found);
  else if (type.flags & ts.TypeFlags.Object) {
    const object = type as ts.ObjectType;
    if (object.objectFlags & ts.ObjectFlags.Reference)
      for (const argument of checker.getTypeArguments(type as ts.TypeReference)) holesIn(checker, argument, depth + 1, found);
    const index = checker.getIndexInfoOfType(type, ts.IndexKind.String);
    if (index) holesIn(checker, index.type, depth + 1, found);
    if (object.objectFlags & ts.ObjectFlags.Anonymous) {
      for (const property of type.getProperties().slice(0, 32)) holesIn(checker, checker.getTypeOfSymbol(property), depth + 1, found);
      for (const signature of type.getCallSignatures()) holesIn(checker, signature.getReturnType(), depth + 1, found);
    }
  }
  return found;
}

/** Match a value type containing holes against the type expected for it; report what each hole must be. */
function unify(checker: ts.TypeChecker, source: ts.Type, target: ts.Type, require: (id: number, type: ts.Type) => void, depth = 0): void {
  if (depth > 6 || !usable(target) || holesIn(checker, target).size) return;
  const id = holeId(source);
  if (id !== undefined) {
    const awaited = awaitedType(checker, target).type;
    if (usable(awaited)) require(id, withoutPromises(checker, awaited));
    return;
  }
  if (isPromiseLike(checker, source)) {
    unify(checker, awaitedType(checker, source).type, awaitedType(checker, target).type, require, depth + 1);
    return;
  }
  const shape = (candidate: ts.Type) => checker.isArrayType(candidate) || checker.isTupleType(candidate) ? 'array' :
    candidate.getCallSignatures().length ? 'function' : candidate.flags & ts.TypeFlags.Object ? 'object' : 'other';
  const kind = shape(source);
  if (target.isUnion()) {
    // A plain value against `T | PromiseLike<T>` (an await or async return) matches T.
    const matching = target.types.filter(member => shape(member) === kind && !isPromiseLike(checker, member));
    if (matching.length === 1) unify(checker, source, matching[0]!, require, depth + 1);
    return;
  }
  if (kind !== shape(target)) return;
  if (kind === 'array') {
    const sources = checker.getTypeArguments(source as ts.TypeReference);
    const targets = checker.getTypeArguments(target as ts.TypeReference);
    const element = (index: number) => checker.isTupleType(target) ? targets[index] : targets[0];
    sources.forEach((item, index) => { const goal = element(index); if (goal) unify(checker, item, goal, require, depth + 1); });
  } else if (kind === 'function') {
    const [from] = source.getCallSignatures(), [to] = target.getCallSignatures();
    if (from && to) unify(checker, from.getReturnType(), to.getReturnType(), require, depth + 1);
  } else if (kind === 'object') {
    const sourceIndex = checker.getIndexInfoOfType(source, ts.IndexKind.String);
    const targetIndex = checker.getIndexInfoOfType(target, ts.IndexKind.String);
    if (sourceIndex && targetIndex) unify(checker, sourceIndex.type, targetIndex.type, require, depth + 1);
    for (const property of source.getProperties()) {
      const goal = target.getProperty(property.name);
      if (goal) unify(checker, checker.getTypeOfSymbol(property), checker.getTypeOfSymbol(goal), require, depth + 1);
    }
  }
}

const usable = (type: ts.Type) => !(type.flags & (ts.TypeFlags.Any | ts.TypeFlags.Unknown | ts.TypeFlags.Never |
  ts.TypeFlags.TypeParameter | ts.TypeFlags.Void));

/** `T | PromiseLike<T>` (what an await or async return expects) as `T`. */
function withoutPromises(checker: ts.TypeChecker, type: ts.Type): ts.Type {
  if (!type.isUnion()) return type;
  const members = type.types.filter(member => !isPromiseLike(checker, member));
  return members.length === type.types.length || members.length !== 1 ? type : members[0]!;
}

/** `"high"` → string, `3` → number; a declared union such as `Tone` stays as it is. */
function baseType(checker: ts.TypeChecker, type: ts.Type): ts.Type | undefined {
  if (!usable(type) || type.flags & (ts.TypeFlags.Null | ts.TypeFlags.Undefined)) return;
  if (type.flags & ts.TypeFlags.Literal && !type.aliasSymbol) return checker.getBaseTypeOfLiteralType(type);
  return type;
}

function skipParentheses(node: ts.Node): ts.Node {
  while (node.parent && ts.isParenthesizedExpression(node.parent)) node = node.parent;
  return node;
}

/** A position whose value is only tested for truthiness. */
function inTruthinessPosition(node: ts.Node): boolean {
  const parent = node.parent;
  if (!parent) return false;
  if (ts.isParenthesizedExpression(parent)) return inTruthinessPosition(parent);
  if ((ts.isIfStatement(parent) || ts.isWhileStatement(parent) || ts.isDoStatement(parent)) && parent.expression === node) return true;
  if (ts.isForStatement(parent) && parent.condition === node) return true;
  if (ts.isConditionalExpression(parent) && parent.condition === node) return true;
  if (ts.isPrefixUnaryExpression(parent) && parent.operator === ts.SyntaxKind.ExclamationToken) return true;
  if (ts.isBinaryExpression(parent) && [ts.SyntaxKind.AmpersandAmpersandToken, ts.SyntaxKind.BarBarToken].includes(parent.operatorToken.kind))
    return inTruthinessPosition(parent);
  // The result of a filter/some/every/find callback.
  const callback = ts.isArrowFunction(parent) && parent.body === node ? parent :
    ts.isReturnStatement(parent) ? ts.findAncestor(parent, ts.isFunctionLike) : undefined;
  if (callback && callback.parent && ts.isCallExpression(callback.parent) && callback.parent.arguments.includes(callback as ts.Expression)) {
    const callee = callback.parent.expression;
    return ts.isPropertyAccessExpression(callee) && PREDICATE_METHODS.has(callee.name.text);
  }
  return false;
}

/**
 * A value that is only shown: logged, or the final expression of an eval (whose value is displayed). Showing
 * a value is a weak requirement: it makes the result text unless another use says otherwise.
 */
function isDisplayed(node: ts.Node): boolean {
  const parent = node.parent;
  if (parent && ts.isCallExpression(parent) && parent.arguments.includes(node as ts.Expression)) {
    const callee = parent.expression;
    return ts.isPropertyAccessExpression(callee) && ts.isIdentifier(callee.expression) && callee.expression.text === 'console';
  }
  if (!parent || !ts.isExpressionStatement(parent)) return false;
  const body = parent.parent;
  return ts.isBlock(body) && body.statements.at(-1) === parent && ts.isFunctionDeclaration(body.parent) &&
    body.parent.name?.text === EVAL_WRAPPER;
}

function isBindingName(node: ts.Node): boolean {
  const parent = node.parent;
  return !!parent && ((ts.isVariableDeclaration(parent) || ts.isParameter(parent) || ts.isBindingElement(parent) ||
    ts.isFunctionDeclaration(parent) || ts.isPropertyAssignment(parent)) && (parent as { name?: ts.Node }).name === node);
}

function isPropertyName(node: ts.Node): boolean {
  const parent = node.parent;
  return !!parent && ts.isPropertyAccessExpression(parent) && parent.name === node;
}
