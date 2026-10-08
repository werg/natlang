/**
 * Neuralese checks over checked TypeScript (S0 §2, §3, §8).
 *
 * `Neuralese<T, D>` is declared in the intrinsics as a branded type (`NeuraleseValue`). The TypeScript checker rejects
 * most misuse by itself; these checks add the natlang rules TypeScript allows: soft values in conditions, template
 * interpolation, arithmetic and comparison, serialisation, spreading and iteration. They also type the literals the
 * runtime substitutes for model-written blocks (`__neuralese("nz1_…")`) from their contextual type, and reject type
 * aliases that reach themselves through a function type.
 */
import ts from 'typescript';
import type { NatlangDiagnostic, SourceSpan } from './inline.js';

/** Brand property of `NeuraleseValue`; no natlang code reads it. */
export const NEURALESE_BRAND = '__natlangNeuralese';
/** Intrinsic the runtime writes in place of a model-written literal before compiling eval code. */
export const NEURALESE_LITERAL_INTRINSIC = '__neuralese';
export const DEFAULT_DIALECT = 'DefaultDialect';

/** A model-written literal and the type its context gives it. */
export type NeuraleseLiteral = SourceSpan & { id: string; type: string };
/** A soft expression that JavaScript would otherwise coerce to text. */
export type NeuraleseReadout = SourceSpan & { kind?: 'join' | 'concat' };

type Report = (node: ts.Node, code: NatlangDiagnostic['code'], message: string) => void;

/** The element type and dialect of a soft value type, or undefined when the type is not one. */
export function neuraleseParts(checker: ts.TypeChecker, type: ts.Type): { element: ts.Type; dialect: string } | undefined {
  if (type.isUnion()) return;
  const brand = checker.getPropertyOfType(type, NEURALESE_BRAND);
  if (!brand) return;
  const declaration = brand.valueDeclaration ?? brand.declarations?.[0];
  const brandType = declaration ? checker.getTypeOfSymbolAtLocation(brand, declaration) : checker.getTypeOfSymbol(brand);
  const element = brandType.getProperty('type'), dialect = brandType.getProperty('dialect');
  if (!element || !dialect) return;
  const dialectType = checker.getTypeOfSymbol(dialect);
  return { element: checker.getTypeOfSymbol(element),
    dialect: dialectType.isStringLiteral() ? dialectType.value : DEFAULT_DIALECT };
}

const isNeuralese = (checker: ts.TypeChecker, type: ts.Type | undefined): boolean =>
  !!type && !!neuraleseParts(checker, checker.getNonNullableType(type));

const ARITHMETIC = new Set([ts.SyntaxKind.PlusToken, ts.SyntaxKind.MinusToken, ts.SyntaxKind.AsteriskToken,
  ts.SyntaxKind.SlashToken, ts.SyntaxKind.PercentToken, ts.SyntaxKind.AsteriskAsteriskToken,
  ts.SyntaxKind.LessThanToken, ts.SyntaxKind.GreaterThanToken, ts.SyntaxKind.LessThanEqualsToken,
  ts.SyntaxKind.GreaterThanEqualsToken, ts.SyntaxKind.EqualsEqualsToken, ts.SyntaxKind.EqualsEqualsEqualsToken,
  ts.SyntaxKind.ExclamationEqualsToken, ts.SyntaxKind.ExclamationEqualsEqualsToken, ts.SyntaxKind.AmpersandToken,
  ts.SyntaxKind.BarToken, ts.SyntaxKind.CaretToken, ts.SyntaxKind.LessThanLessThanToken,
  ts.SyntaxKind.GreaterThanGreaterThanToken, ts.SyntaxKind.GreaterThanGreaterThanGreaterThanToken,
  ts.SyntaxKind.PlusEqualsToken, ts.SyntaxKind.MinusEqualsToken, ts.SyntaxKind.InKeyword, ts.SyntaxKind.InstanceOfKeyword]);
const CONDITIONAL = new Set([ts.SyntaxKind.AmpersandAmpersandToken, ts.SyntaxKind.BarBarToken]);

/** Natlang type text for a soft value type: `Neuralese<T>` or `Neuralese<T, "dialect">`. */
export function neuraleseTypeText(checker: ts.TypeChecker, type: ts.Type, location?: ts.Node): string | undefined {
  const parts = neuraleseParts(checker, type);
  if (!parts) return;
  const element = checker.typeToString(parts.element, location, ts.TypeFormatFlags.NoTruncation |
    ts.TypeFormatFlags.UseAliasDefinedOutsideCurrentScope);
  return `Neuralese<${element}${parts.dialect === DEFAULT_DIALECT ? '' : `, ${JSON.stringify(parts.dialect)}`}>`;
}

/**
 * Report misuse of soft values in `file` and return the model-written literals it contains, typed from context.
 */
export function checkNeuralese(checker: ts.TypeChecker, file: ts.SourceFile, report: Report,
  options: { recursiveTypes?: boolean; readouts?: NeuraleseReadout[] } = {}): NeuraleseLiteral[] {
  const literals: NeuraleseLiteral[] = [];
  const soft = (node: ts.Node | undefined): boolean => !!node && ts.isExpression(node) && isNeuralese(checker, checker.getTypeAtLocation(node));
  const opaque = (node: ts.Node, what: string) => report(node, 'neuralese-opaque-access',
    `A Neuralese value is opaque: ${what}. Read it with read(value) to get an ordinary value, or pass it to a function that takes it.`);
  const condition = (node: ts.Node) => report(node, 'neuralese-condition',
    'A Neuralese value cannot decide a branch. Read it with read(value), or ask a natural-language function about it.');
  const readout = (node: ts.Expression, kind?: NeuraleseReadout['kind']) => {
    options.readouts?.push({ ...span(node), ...(kind ? { kind } : {}) });
    let parent: ts.Node | undefined = node.parent;
    while (parent && !ts.isFunctionLike(parent)) parent = parent.parent;
    const async = parent && ts.canHaveModifiers(parent) && ts.getModifiers(parent)?.some(modifier => modifier.kind === ts.SyntaxKind.AsyncKeyword);
    if (!async)
      report(node, 'neuralese-readout-sync', 'Reading a Neuralese value needs async work; make this function async or move the text conversion into async code.');
  };
  const standardMethod = (property: ts.PropertyAccessExpression, names: readonly string[]): boolean => {
    const symbol = checker.getSymbolAtLocation(property.name);
    return !!symbol?.declarations?.some(declaration => {
      const source = declaration.getSourceFile();
      const owner = declaration.parent;
      return source.isDeclarationFile && source.fileName.replace(/\\/g, '/').includes('/typescript/lib/lib.') &&
        ts.isInterfaceDeclaration(owner) && names.includes(owner.name.text);
    });
  };
  const stringToString = (property: ts.PropertyAccessExpression): boolean => {
    const parent = property.parent;
    return ts.isCallExpression(parent) && parent.expression === property && parent.arguments.length === 0 &&
      property.name.text === 'toString' && standardMethod(property, ['Object']) && soft(property.expression);
  };
  const arrayJoinKind = (expression: ts.Expression): 'supported' | undefined => {
    const array = checker.getTypeAtLocation(expression);
    if (!checker.isArrayType(array) && !checker.isTupleType(array)) return;
    const element = checker.getIndexTypeOfType(array, ts.IndexKind.Number);
    if (!element) return;
    const members = element.isUnion() ? element.types : [element];
    let containsSoft = false;
    for (const member of members) {
      if (member.flags & (ts.TypeFlags.Null | ts.TypeFlags.Undefined)) continue;
      const value = checker.getNonNullableType(member);
      if (neuraleseParts(checker, value)) { containsSoft = true; continue; }
    }
    return containsSoft ? 'supported' : undefined;
  };
  const softArrayJoinKind = (call: ts.CallExpression): 'supported' | undefined => {
    if (!ts.isPropertyAccessExpression(call.expression) || call.expression.name.text !== 'join' ||
        !standardMethod(call.expression, ['Array', 'ReadonlyArray']) || call.arguments.length > 1) return;
    return arrayJoinKind(call.expression.expression);
  };
  const visit = (node: ts.Node): void => {
    if (ts.isTypeReferenceNode(node) && ts.isIdentifier(node.typeName) && node.typeName.text === 'Neuralese' && node.typeArguments?.[0] &&
        isNeuralese(checker, checker.getTypeFromTypeNode(node.typeArguments[0])))
      report(node, 'neuralese-nested', 'Neuralese<Neuralese<T>> is not a type: a view of a view means nothing a view does not.');
    else if ((ts.isPropertyAccessExpression(node) || ts.isElementAccessExpression(node)) && soft(node.expression)) {
      if (ts.isPropertyAccessExpression(node) && stringToString(node)) readout(node.expression);
      else opaque(node, 'it has no fields or elements to read');
    }
    else if (ts.isBinaryExpression(node) && (soft(node.left) || soft(node.right))) {
      const kind = node.operatorToken.kind;
      if (CONDITIONAL.has(kind) && soft(node.left)) condition(node.left);
      else if ((kind === ts.SyntaxKind.PlusToken || kind === ts.SyntaxKind.PlusEqualsToken) &&
          [node.left, node.right].some(side => checker.getTypeAtLocation(side).flags & ts.TypeFlags.StringLike)) {
        // `text += soft` coerces the right-hand value. A soft left side is not
        // a writable string accumulator and remains an ordinary type error.
        if (kind === ts.SyntaxKind.PlusToken && soft(node.left)) readout(node.left);
        if (soft(node.right)) readout(node.right);
      }
      else if (ARITHMETIC.has(kind)) opaque(node, 'it cannot be computed with or compared');
    } else if (ts.isPrefixUnaryExpression(node) && soft(node.operand)) {
      if (node.operator === ts.SyntaxKind.ExclamationToken) condition(node.operand); else opaque(node, 'it cannot be computed with');
    } else if ((ts.isIfStatement(node) || ts.isWhileStatement(node) || ts.isDoStatement(node)) && soft(node.expression))
      condition(node.expression);
    else if (ts.isForStatement(node) && soft(node.condition)) condition(node.condition!);
    else if (ts.isConditionalExpression(node) && soft(node.condition)) condition(node.condition);
    else if (ts.isTemplateSpan(node) && !ts.isTaggedTemplateExpression(node.parent.parent) && soft(node.expression)) readout(node.expression);
    else if ((ts.isSpreadElement(node) || ts.isSpreadAssignment(node)) && soft(node.expression)) opaque(node, 'it cannot be spread');
    else if ((ts.isForOfStatement(node) || ts.isForInStatement(node)) && soft(node.expression))
      opaque(node.expression, 'it cannot be iterated');
    else if (ts.isCallExpression(node)) {
      const callee = node.expression;
      if (ts.isIdentifier(callee) && callee.text === 'String' && isDefaultString(checker, callee)) {
        const first = node.arguments[0];
        if (first && soft(first)) readout(first);
      }
      if (ts.isPropertyAccessExpression(callee) && ts.isIdentifier(callee.expression) && callee.expression.text === 'JSON' &&
          callee.name.text === 'stringify' && isDefaultGlobal(checker, callee.expression)) {
        const [value, ...options] = node.arguments;
        if (value && soft(value)) readout(value);
        if (options.some(argument => soft(argument))) opaque(node, 'its payload cannot be serialised');
      }
      if (ts.isPropertyAccessExpression(callee) && callee.name.text === 'concat' && standardMethod(callee, ['String']) &&
          !!(checker.getTypeAtLocation(callee.expression).flags & ts.TypeFlags.StringLike)) {
        let softArgument = false;
        for (const argument of node.arguments) {
          // The call's existing concat readout lowering materializes argument spreads
          // before any awaited reads, then coerces each expanded argument in order.
          if (ts.isSpreadElement(argument) && arrayJoinKind(argument.expression)) softArgument = true;
          else if (soft(argument)) softArgument = true;
        }
        if (softArgument) readout(node, 'concat');
      }
      const joinKind = softArrayJoinKind(node);
      if (joinKind === 'supported') readout(node, 'join');
      if (ts.isIdentifier(callee) && callee.text === NEURALESE_LITERAL_INTRINSIC) literal(node);
    }
    ts.forEachChild(node, visit);
  };
  const literal = (node: ts.CallExpression): void => {
    const argument = node.arguments[0];
    if (node.arguments.length !== 1 || !argument || !ts.isStringLiteral(argument)) return;
    // An async return position gives `T | PromiseLike<T>`: the one soft member is the literal's type.
    const contextual = checker.getContextualType(node);
    const members = !contextual ? [] : contextual.isUnion() ? contextual.types : [contextual];
    const texts = [...new Set(members.map(member => neuraleseTypeText(checker, member, node)).filter((text): text is string => !!text))];
    const text = texts.length === 1 ? texts[0] : undefined;
    if (!text) {
      report(node, 'neuralese-untyped-literal', 'Nothing says what this Neuralese literal is: write its type first, as in ' +
        '`const plan: Neuralese<Plan> = …`, or pass it where a Neuralese parameter is expected.');
      return;
    }
    const start = node.getStart(), position = file.getLineAndCharacterOfPosition(start);
    literals.push({ file: file.fileName, start, end: node.getEnd(), line: position.line + 1, column: position.character + 1,
      id: argument.text, type: text });
  };
  visit(file);
  if (options.recursiveTypes) for (const statement of file.statements) visitAliases(statement);
  return literals;

  function span(node: ts.Node): NeuraleseReadout {
    const start = node.getStart(file), position = file.getLineAndCharacterOfPosition(start);
    return { file: file.fileName, start, end: node.getEnd(), line: position.line + 1, column: position.character + 1 };
  }

  function visitAliases(node: ts.Node): void {
    if (ts.isTypeAliasDeclaration(node)) checkRecursiveFunction(checker, node, report);
    ts.forEachChild(node, visitAliases);
  }
}

function isDefaultString(checker: ts.TypeChecker, identifier: ts.Identifier): boolean {
  const symbol = checker.getSymbolAtLocation(identifier);
  return symbol?.name === 'String' && !!symbol.declarations?.some(declaration => declaration.getSourceFile().isDeclarationFile &&
    declaration.getSourceFile().fileName.replace(/\\/g, '/').includes('/typescript/lib/lib.'));
}

function isDefaultGlobal(checker: ts.TypeChecker, identifier: ts.Identifier): boolean {
  const symbol = checker.getSymbolAtLocation(identifier);
  return !!symbol?.declarations?.some(declaration => declaration.getSourceFile().isDeclarationFile &&
    declaration.getSourceFile().fileName.replace(/\\/g, '/').includes('/typescript/lib/lib.'));
}

/**
 * Reject a type alias that reaches itself through a function type (directly or through other aliases):
 * self-application would make recursion expressible. Recursive data aliases stay allowed. Applied to model-written
 * eval code; natlang signature and frontmatter types are checked by the native type environment (native/types.ts).
 */
export function checkRecursiveFunction(checker: ts.TypeChecker, alias: ts.TypeAliasDeclaration, report: Report): void {
  const target = checker.getSymbolAtLocation(alias.name);
  if (!target) return;
  const seen = new Set<string>();
  let found = false;
  const walk = (node: ts.Node, throughFunction: boolean): void => {
    if (found) return;
    // Function types are what natlang can call; method-bearing shapes are host contracts (Live), checked by identity.
    const entering = throughFunction || ts.isFunctionTypeNode(node) || ts.isConstructorTypeNode(node);
    if (ts.isTypeReferenceNode(node)) {
      let symbol = checker.getSymbolAtLocation(node.typeName);
      if (symbol && symbol.flags & ts.SymbolFlags.Alias) symbol = checker.getAliasedSymbol(symbol);
      if (symbol === target && throughFunction) { found = true; return; }
      const declaration = symbol?.declarations?.find(ts.isTypeAliasDeclaration);
      if (symbol && declaration && !declaration.getSourceFile().isDeclarationFile) {
        const key = `${symbol.name}@${declaration.pos}:${throughFunction}`;
        if (!seen.has(key)) { seen.add(key); walk(declaration.type, throughFunction); }
      }
    }
    ts.forEachChild(node, child => walk(child, entering));
  };
  walk(alias.type, false);
  if (found) report(alias.name, 'type-recursive-function', `Type ${alias.name.text} refers to itself through a function type. ` +
    'Natlang types may be recursive data, but not recursive functions: give the function a non-recursive type.');
}
