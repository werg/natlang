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
import type { Representation } from '../native/representation.js';

/** Brand property of `NeuraleseValue`; no natlang code reads it. */
export const NEURALESE_BRAND = '__natlangNeuralese';
/** Intrinsic the runtime writes in place of a model-written literal before compiling eval code. */
export const NEURALESE_LITERAL_INTRINSIC = '__neuralese';
export const DEFAULT_DIALECT = 'DefaultDialect';

/** A model-written literal and the type its context gives it. */
export type NeuraleseLiteral = SourceSpan & { id: string; type: string };
/** A soft expression that JavaScript would otherwise coerce to text. */
export type NeuraleseReadout = SourceSpan & { kind?: 'join' | 'array-string' | 'concat' | 'json' | 'error' | 'string-argument' | 'scalar-conversion' | 'string-replace' | 'array-map';
  argument?: number; conversion?: 'Number' | 'Boolean'; conditional?: true; callback?: { start: number; end: number } };

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

/** Whether any union arm is a typed Neuralese value; intentionally local to coercion lowering. */
const hasSoftAlternative = (checker: ts.TypeChecker, type: ts.Type | undefined): boolean => {
  if (!type) return false;
  const alternatives = type.isUnion() ? type.types : [type];
  return alternatives.some(alternative => isNeuralese(checker, alternative));
};

/** Recover a soft type erased only by an explicit, local TypeScript assertion in a conversion operand. */
const assertedSoftSourceType = (checker: ts.TypeChecker, expression: ts.Expression): ts.Type | undefined => {
  let current = expression;
  let sawAssertion = false;
  while (true) {
    if (ts.isParenthesizedExpression(current) || ts.isNonNullExpression(current)) {
      current = current.expression;
      continue;
    }
    if (ts.isAsExpression(current) || ts.isTypeAssertionExpression(current) || ts.isSatisfiesExpression(current)) {
      sawAssertion = true;
      current = current.expression;
      continue;
    }
    break;
  }
  if (!sawAssertion) return;
  const sourceType = checker.getTypeAtLocation(current);
  return hasSoftAlternative(checker, sourceType) ? sourceType : undefined;
};

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

/** Phantom property of `NatlangGenericFunction`: the callee's result is representation-generic. */
export const GENERIC_RESULT_BRAND = '__natlangGenericResult';
/** A call of a function with a representation-generic result and the instance its expected type picks. */
export type GenericInstantiation = { start: number; end: number; line: number; representation: Representation };

/**
 * Representation-generic results at their call sites (DECISIONS.md 2026-10-09). For each call of a natlang function
 * whose result is generic, the instance the checker inferred from the expected type: a `Neuralese` instantiation runs
 * the Neuralese instance, anything else (no expected type, or one that admits the crisp type) the crisp one. A call of
 * a natlang function with a crisp result where a `Neuralese` value is expected is reported with how to make it generic.
 */
export function checkGenericResults(checker: ts.TypeChecker, file: ts.SourceFile, report: Report, instantiations: GenericInstantiation[]): void {
  const awaited = (type: ts.Type): ts.Type =>
    (checker as ts.TypeChecker & { getAwaitedType?(type: ts.Type): ts.Type | undefined }).getAwaitedType?.(type) ?? type;
  const open = (type: ts.Type | undefined) => !type || !!(type.flags & (ts.TypeFlags.Any | ts.TypeFlags.Unknown | ts.TypeFlags.TypeParameter));
  /** The value type the call's context expects: through `await`, or the promised type of a promise slot. */
  const expected = (call: ts.CallExpression): ts.Type | undefined => {
    let outer: ts.Expression = call;
    while (ts.isParenthesizedExpression(outer.parent)) outer = outer.parent;
    const type = ts.isAwaitExpression(outer.parent) ? checker.getContextualType(outer.parent) :
      (() => { const slot = checker.getContextualType(outer); return slot && awaited(slot); })();
    return open(type) ? undefined : checker.getNonNullableType(type!);
  };
  const visit = (node: ts.Node): void => {
    if (ts.isCallExpression(node)) {
      const callee = checker.getTypeAtLocation(node.expression);
      const generic = !!checker.getPropertyOfType(callee, GENERIC_RESULT_BRAND);
      // Named natlang functions carry `iterateOn`; inline `nl` values (with `.with`) take their result from context.
      const named = generic || (!!checker.getPropertyOfType(callee, 'iterateOn') && !checker.getPropertyOfType(callee, 'with') &&
        callee.getCallSignatures().length > 0);
      const signature = named ? checker.getResolvedSignature(node) : undefined;
      const result = signature && awaited(checker.getReturnTypeOfSignature(signature));
      if (generic && result) {
        const soft = neuraleseParts(checker, result);
        instantiations.push({ start: node.getStart(file), end: node.getEnd(),
          line: file.getLineAndCharacterOfPosition(node.getStart(file)).line + 1,
          representation: soft ? { kind: 'neuralese', ...(soft.dialect === DEFAULT_DIALECT ? {} : { dialect: soft.dialect }) } : { kind: 'crisp' } });
      } else if (named && result && !open(result) && !hasSoftAlternative(checker, result)) {
        const wanted = expected(node);
        if (wanted && neuraleseParts(checker, wanted) && !checker.isTypeAssignableTo(result, wanted)) {
          const name = node.expression.getText(file), crisp = checker.typeToString(result, node, ts.TypeFormatFlags.NoTruncation);
          report(node, 'neuralese-crisp-result', `${name} returns ${crisp}, and this call's result is used as ` +
            `${neuraleseTypeText(checker, wanted, node)}. Declare ${name}'s result representation-generic ` +
            `(\`generic: { R: ${crisp} | Neuralese<${crisp}> }\` with \`returns: R\`) so that this call writes its Neuralese form, or call ` +
            'a function that returns the Neuralese value.');
        }
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
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
  const readout = (node: ts.Expression, kind?: NeuraleseReadout['kind'], conditional = false, argument?: number,
    conversion?: NeuraleseReadout['conversion']) => {
    options.readouts?.push({ ...span(node), ...(kind ? { kind } : {}), ...(argument === undefined ? {} : { argument }),
      ...(conversion ? { conversion } : {}),
      ...(conditional ? { conditional: true } : {}) });
    let parent: ts.Node | undefined = node.parent;
    while (parent && !ts.isFunctionLike(parent)) parent = parent.parent;
    const async = parent && ts.canHaveModifiers(parent) && ts.getModifiers(parent)?.some(modifier => modifier.kind === ts.SyntaxKind.AsyncKeyword);
    if (!async && parent && (ts.isArrowFunction(parent) || ts.isFunctionExpression(parent))) {
      const map = asyncArrayMapCallback(parent);
      const enclosing = map && enclosingAsyncFunction(map.call);
      if (map && enclosing) {
        const key = `${map.call.getStart(file)}:${map.call.getEnd()}`;
        if (!asyncMapCalls.has(key)) {
          asyncMapCalls.add(key);
          options.readouts?.push({ ...span(map.call), kind: 'array-map',
            callback: { start: map.callback.getStart(file), end: map.callback.getEnd() } });
        }
        return;
      }
    }
    if (!async)
      report(node, 'neuralese-readout-sync', 'Reading a Neuralese value needs async work; make this function async or move the text conversion into async code.');
  };
  const asyncMapCalls = new Set<string>();
  const enclosingAsyncFunction = (call: ts.CallExpression): boolean => {
    for (let current: ts.Node | undefined = call.parent; current; current = current.parent) {
      if (ts.isFunctionLike(current)) return !!(ts.canHaveModifiers(current) &&
        ts.getModifiers(current)?.some(modifier => modifier.kind === ts.SyntaxKind.AsyncKeyword));
    }
    return false;
  };
  const asyncArrayMapCallback = (callback: ts.FunctionLikeDeclaration): { call: ts.CallExpression; callback: ts.FunctionLikeDeclaration } | undefined => {
    if (callback.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.AsyncKeyword)) return;
    let expression: ts.Node = callback;
    while (ts.isParenthesizedExpression(expression.parent)) expression = expression.parent;
    const argument = expression.parent;
    if (!ts.isCallExpression(argument) || argument.arguments[0] !== expression || !ts.isPropertyAccessExpression(argument.expression) ||
        argument.expression.name.text !== 'map' || !standardMethod(argument.expression, ['Array', 'ReadonlyArray'])) return;
    const receiver = checker.getTypeAtLocation(argument.expression.expression);
    const alternatives = receiver.isUnion() ? receiver.types : [receiver];
    if (!alternatives.length || alternatives.some(type => !checker.isArrayType(type) && !checker.isTupleType(type))) return;
    const signature = checker.getSignatureFromDeclaration(callback);
    const returns = signature && checker.getReturnTypeOfSignature(signature);
    if (!returns || returns.flags & (ts.TypeFlags.Any | ts.TypeFlags.Unknown)) return;
    const resultTypes = returns.isUnion() ? returns.types : [returns];
    if (resultTypes.some(type => !!checker.getPropertyOfType(type, 'then'))) return;
    return { call: argument, callback };
  };
  const standardDeclaration = (symbol: ts.Symbol | undefined, names: readonly string[]): boolean => {
    return !!symbol?.declarations?.some(declaration => {
      const source = declaration.getSourceFile();
      const owner = declaration.parent;
      return source.isDeclarationFile && source.fileName.replace(/\\/g, '/').includes('/typescript/lib/lib.') &&
        ts.isInterfaceDeclaration(owner) && names.includes(owner.name.text);
    });
  };
  const standardMethod = (property: ts.PropertyAccessExpression, names: readonly string[]): boolean =>
    standardDeclaration(checker.getSymbolAtLocation(property.name), names);
  const stringToString = (property: ts.PropertyAccessExpression): boolean => {
    const parent = property.parent;
    return ts.isCallExpression(parent) && parent.expression === property && parent.arguments.length === 0 &&
      property.name.text === 'toString' && standardMethod(property, ['Object']) &&
      (soft(property.expression) || hasSoftAlternative(checker, checker.getTypeAtLocation(property.expression)));
  };
  const typedStringReceiverMethods = new Set(['trim', 'trimStart', 'trimEnd', 'toLowerCase', 'toUpperCase',
    'includes', 'startsWith', 'endsWith', 'indexOf', 'lastIndexOf', 'slice', 'substring']);
  const softStringReceiverCall = (property: ts.PropertyAccessExpression): boolean => {
    const call = property.parent;
    if (!ts.isCallExpression(call) || call.expression !== property || !typedStringReceiverMethods.has(property.name.text) ||
        call.arguments.some(argument => ts.isSpreadElement(argument) || hasSoftAlternative(checker, checker.getTypeAtLocation(argument)))) return false;
    const receiver = checker.getTypeAtLocation(property.expression);
    const alternatives = receiver.isUnion() ? receiver.types : [receiver];
    let foundSoft = false;
    for (const alternative of alternatives) {
      const parts = neuraleseParts(checker, alternative);
      if (parts) {
        foundSoft = true;
        const payloads = parts.element.isUnion() ? parts.element.types : [parts.element];
        if (!payloads.length || payloads.some(payload => !(payload.flags & ts.TypeFlags.StringLike) ||
            !standardDeclaration(checker.getPropertyOfType(payload, property.name.text), ['String']))) return false;
      } else if (!(alternative.flags & ts.TypeFlags.StringLike) ||
          !standardDeclaration(checker.getPropertyOfType(alternative, property.name.text), ['String'])) return false;
    }
    return foundSoft;
  };
  const arrayJoinKind = (expression: ts.Expression): 'supported' | undefined => {
    const array = checker.getTypeAtLocation(expression);
    const alternatives = array.isUnion() ? array.types : [array];
    if (!alternatives.length || alternatives.some(alternative =>
      !checker.isArrayType(alternative) && !checker.isTupleType(alternative))) return;
    let containsSoft = false;
    for (const alternative of alternatives) {
      const element = checker.getIndexTypeOfType(alternative, ts.IndexKind.Number);
      if (!element) continue;
      const members = element.isUnion() ? element.types : [element];
      for (const member of members) {
        if (member.flags & (ts.TypeFlags.Null | ts.TypeFlags.Undefined)) continue;
        const value = checker.getNonNullableType(member);
        if (hasSoftAlternative(checker, value)) { containsSoft = true; continue; }
      }
    }
    return containsSoft ? 'supported' : undefined;
  };
  const softArrayJoinKind = (call: ts.CallExpression): 'supported' | undefined => {
    if (!ts.isPropertyAccessExpression(call.expression) || call.expression.name.text !== 'join' ||
        !standardMethod(call.expression, ['Array', 'ReadonlyArray']) || call.arguments.length > 1) return;
    return arrayJoinKind(call.expression.expression);
  };
  const softArrayToStringKind = (call: ts.CallExpression): 'supported' | undefined => {
    if (!ts.isPropertyAccessExpression(call.expression) || call.expression.name.text !== 'toString' ||
        !standardMethod(call.expression, ['Object', 'Array', 'ReadonlyArray'])) return;
    return arrayJoinKind(call.expression.expression);
  };
  const softBooleanAlternatives = (expression: ts.Expression): boolean => {
    const type = checker.getTypeAtLocation(expression);
    const alternatives = type.isUnion() ? type.types : [type];
    let found = false;
    for (const alternative of alternatives) {
      const parts = neuraleseParts(checker, alternative);
      if (!parts) continue;
      found = true;
      const members = parts.element.isUnion() ? parts.element.types : [parts.element];
      if (!members.length || members.some(member => !(member.flags & ts.TypeFlags.BooleanLike))) return false;
    }
    return found;
  };
  const guardExpressions = new WeakSet<ts.Expression>();
  const plannedGuardReadouts = new Set<string>();
  const validGuardDiscarded = (expression: ts.Expression): boolean => {
    if (ts.isParenthesizedExpression(expression)) return validGuardDiscarded(expression.expression);
    if (ts.isBinaryExpression(expression) && expression.operatorToken.kind === ts.SyntaxKind.CommaToken)
      return validGuardDiscarded(expression.left) && validGuardDiscarded(expression.right);
    if (ts.isPrefixUnaryExpression(expression) && expression.operator === ts.SyntaxKind.ExclamationToken)
      return validGuardTruthiness(expression.operand);
    if (ts.isBinaryExpression(expression) && CONDITIONAL.has(expression.operatorToken.kind))
      return validGuardTruthiness(expression.left) && validGuardDiscarded(expression.right);
    if (ts.isConditionalExpression(expression))
      return validGuardTruthiness(expression.condition) && validGuardDiscarded(expression.whenTrue) &&
        validGuardDiscarded(expression.whenFalse);
    return true;
  };
  const validGuardTruthiness = (expression: ts.Expression): boolean => {
    if (ts.isParenthesizedExpression(expression)) return validGuardTruthiness(expression.expression);
    if (ts.isPrefixUnaryExpression(expression) && expression.operator === ts.SyntaxKind.ExclamationToken)
      return validGuardTruthiness(expression.operand);
    if (ts.isBinaryExpression(expression) && CONDITIONAL.has(expression.operatorToken.kind))
      return validGuardTruthiness(expression.left) && validGuardTruthiness(expression.right);
    if (ts.isBinaryExpression(expression) && expression.operatorToken.kind === ts.SyntaxKind.CommaToken)
      return validGuardDiscarded(expression.left) && validGuardTruthiness(expression.right);
    if (ts.isConditionalExpression(expression))
      return validGuardTruthiness(expression.condition) && validGuardTruthiness(expression.whenTrue) &&
        validGuardTruthiness(expression.whenFalse);
    return !hasSoftAlternative(checker, checker.getTypeAtLocation(expression)) || softBooleanAlternatives(expression);
  };
  const planGuardTruthiness = (expression: ts.Expression): void => {
    guardExpressions.add(expression);
    if (ts.isParenthesizedExpression(expression)) return planGuardTruthiness(expression.expression);
    if (ts.isPrefixUnaryExpression(expression) && expression.operator === ts.SyntaxKind.ExclamationToken)
      return planGuardTruthiness(expression.operand);
    if (ts.isBinaryExpression(expression) && CONDITIONAL.has(expression.operatorToken.kind)) {
      planGuardTruthiness(expression.left);
      planGuardTruthiness(expression.right);
      return;
    }
    if (ts.isBinaryExpression(expression) && expression.operatorToken.kind === ts.SyntaxKind.CommaToken) {
      planGuardDiscarded(expression.left);
      planGuardTruthiness(expression.right);
      return;
    }
    if (ts.isConditionalExpression(expression)) {
      planGuardTruthiness(expression.condition);
      planGuardTruthiness(expression.whenTrue);
      planGuardTruthiness(expression.whenFalse);
      return;
    }
    if (!hasSoftAlternative(checker, checker.getTypeAtLocation(expression))) return;
    const key = `${expression.getStart(file)}:${expression.getEnd()}`;
    if (plannedGuardReadouts.has(key)) return;
    plannedGuardReadouts.add(key);
    readout(expression, undefined, !soft(expression));
  };
  const planGuardDiscarded = (expression: ts.Expression): void => {
    guardExpressions.add(expression);
    if (ts.isParenthesizedExpression(expression)) return planGuardDiscarded(expression.expression);
    if (ts.isBinaryExpression(expression) && expression.operatorToken.kind === ts.SyntaxKind.CommaToken) {
      planGuardDiscarded(expression.left);
      planGuardDiscarded(expression.right);
      return;
    }
    if (ts.isPrefixUnaryExpression(expression) && expression.operator === ts.SyntaxKind.ExclamationToken) {
      planGuardTruthiness(expression.operand);
      return;
    }
    if (ts.isBinaryExpression(expression) && CONDITIONAL.has(expression.operatorToken.kind)) {
      planGuardTruthiness(expression.left);
      planGuardDiscarded(expression.right);
      return;
    }
    if (ts.isConditionalExpression(expression)) {
      planGuardTruthiness(expression.condition);
      planGuardDiscarded(expression.whenTrue);
      planGuardDiscarded(expression.whenFalse);
    }
  };
  const planControlGuards = (node: ts.Node): void => {
    const guards: ts.Expression[] = [];
    if (ts.isIfStatement(node) || ts.isWhileStatement(node) || ts.isDoStatement(node)) guards.push(node.expression);
    else if (ts.isForStatement(node) && node.condition) guards.push(node.condition);
    else if (ts.isConditionalExpression(node)) guards.push(node.condition);
    for (const guard of guards) if (validGuardTruthiness(guard)) planGuardTruthiness(guard);
    ts.forEachChild(node, planControlGuards);
  };
  const scalarSoftAlternatives = (type: ts.Type | undefined): boolean => {
    if (!type) return false;
    const alternatives = type.isUnion() ? type.types : [type];
    let found = false;
    for (const alternative of alternatives) {
      const parts = neuraleseParts(checker, alternative);
      if (!parts) continue;
      found = true;
      const members = parts.element.isUnion() ? parts.element.types : [parts.element];
      if (members.some(member => !(member.flags & (ts.TypeFlags.StringLike | ts.TypeFlags.NumberLike | ts.TypeFlags.BooleanLike))))
        return false;
    }
    return found;
  };
  const stringTextArgument = (call: ts.CallExpression): number | undefined => {
    if (!ts.isPropertyAccessExpression(call.expression) || !standardMethod(call.expression, ['String']) ||
        !(checker.getTypeAtLocation(call.expression.expression).flags & ts.TypeFlags.StringLike) ||
        call.arguments.some(ts.isSpreadElement)) return;
    const index = new Map([['includes', 0], ['startsWith', 0], ['endsWith', 0], ['indexOf', 0], ['lastIndexOf', 0],
      ['localeCompare', 0], ['padStart', 1], ['padEnd', 1]]).get(call.expression.name.text);
    return index !== undefined && call.arguments[index] &&
      hasSoftAlternative(checker, checker.getTypeAtLocation(call.arguments[index]!)) ? index : undefined;
  };
  const stringOperand = (type: ts.Type, allowSoft: boolean): boolean => {
    const alternatives = type.isUnion() ? type.types : [type];
    return alternatives.length > 0 && alternatives.every(alternative => {
      const parts = neuraleseParts(checker, alternative);
      if (!parts) return !!(alternative.flags & ts.TypeFlags.StringLike);
      if (!allowSoft) return false;
      const elements = parts.element.isUnion() ? parts.element.types : [parts.element];
      return elements.length > 0 && elements.every(element => !!(element.flags & ts.TypeFlags.StringLike));
    });
  };
  const stringReplaceKind = (property: ts.PropertyAccessExpression): { conditional: boolean } | undefined => {
    if (property.name.text !== 'replace' || !ts.isCallExpression(property.parent) || property.parent.expression !== property ||
        property.parent.arguments.length < 1 || property.parent.arguments.length > 2 ||
        property.parent.arguments.some(ts.isSpreadElement)) return;
    const receiverType = checker.getTypeAtLocation(property.expression);
    const receivers = receiverType.isUnion() ? receiverType.types : [receiverType];
    let containsSoftReceiver = false;
    for (const receiver of receivers) {
      const parts = neuraleseParts(checker, receiver);
      if (parts) {
        containsSoftReceiver = true;
        const elements = parts.element.isUnion() ? parts.element.types : [parts.element];
        if (!elements.length || elements.some(element => !(element.flags & ts.TypeFlags.StringLike)) ||
            elements.some(element => !standardDeclaration(checker.getPropertyOfType(element, 'replace'), ['String']))) return;
      } else if (!(receiver.flags & ts.TypeFlags.StringLike) ||
          !standardDeclaration(checker.getPropertyOfType(receiver, 'replace'), ['String'])) return;
    }
    if (!containsSoftReceiver || !property.parent.arguments.every(argument =>
      stringOperand(checker.getTypeAtLocation(argument), false))) return;
    return { conditional: !soft(property.expression) };
  };
  const visit = (node: ts.Node): void => {
    if (ts.isTypeReferenceNode(node) && ts.isIdentifier(node.typeName) && node.typeName.text === 'Neuralese' && node.typeArguments?.[0] &&
        isNeuralese(checker, checker.getTypeFromTypeNode(node.typeArguments[0])))
      report(node, 'neuralese-nested', 'Neuralese<Neuralese<T>> is not a type: a view of a view means nothing a view does not.');
    else if (ts.isComputedPropertyName(node) &&
        hasSoftAlternative(checker, checker.getTypeAtLocation(node.expression)))
      readout(node.expression, undefined, !soft(node.expression));
    else if (ts.isPropertyAccessExpression(node) || ts.isElementAccessExpression(node)) {
      if (hasSoftAlternative(checker, checker.getTypeAtLocation(node.expression))) {
        if (ts.isPropertyAccessExpression(node) && (stringToString(node) || softStringReceiverCall(node)))
          readout(node.expression, undefined, !soft(node.expression));
        else if (ts.isPropertyAccessExpression(node) && stringReplaceKind(node)) { /* Lower the standard typed string call. */ }
        else opaque(node, 'it has no fields or elements to read');
      } else if (ts.isElementAccessExpression(node) && node.argumentExpression &&
          hasSoftAlternative(checker, checker.getTypeAtLocation(node.argumentExpression)))
        readout(node.argumentExpression, undefined, !soft(node.argumentExpression));
    }
    else if (ts.isBinaryExpression(node) &&
        (hasSoftAlternative(checker, checker.getTypeAtLocation(node.left)) ||
          hasSoftAlternative(checker, checker.getTypeAtLocation(node.right)))) {
      const kind = node.operatorToken.kind;
      if (CONDITIONAL.has(kind) && hasSoftAlternative(checker, checker.getTypeAtLocation(node.left)) &&
          !guardExpressions.has(node)) condition(node.left);
      else if ((kind === ts.SyntaxKind.PlusToken || kind === ts.SyntaxKind.PlusEqualsToken) &&
          [node.left, node.right].some(side => checker.getTypeAtLocation(side).flags & ts.TypeFlags.StringLike)) {
        // `text += soft` coerces the right-hand value. A soft left side is not
        // a writable string accumulator and remains an ordinary type error.
        if (kind === ts.SyntaxKind.PlusToken && hasSoftAlternative(checker, checker.getTypeAtLocation(node.left)))
          readout(node.left, undefined, !soft(node.left));
        if (hasSoftAlternative(checker, checker.getTypeAtLocation(node.right)))
          readout(node.right, undefined, !soft(node.right));
      }
      else if (ARITHMETIC.has(kind)) opaque(node, 'it cannot be computed with or compared');
    } else if (ts.isPrefixUnaryExpression(node) && hasSoftAlternative(checker, checker.getTypeAtLocation(node.operand))) {
      if (node.operator === ts.SyntaxKind.ExclamationToken) {
        if (!guardExpressions.has(node)) condition(node.operand);
      } else opaque(node, 'it cannot be computed with');
    } else if ((ts.isIfStatement(node) || ts.isWhileStatement(node) || ts.isDoStatement(node)) &&
        hasSoftAlternative(checker, checker.getTypeAtLocation(node.expression)))
      { if (!guardExpressions.has(node.expression)) condition(node.expression); }
    else if (ts.isForStatement(node) && node.condition && hasSoftAlternative(checker, checker.getTypeAtLocation(node.condition)))
      { if (!guardExpressions.has(node.condition)) condition(node.condition); }
    else if (ts.isConditionalExpression(node) && hasSoftAlternative(checker, checker.getTypeAtLocation(node.condition)))
      { if (!guardExpressions.has(node.condition)) condition(node.condition); }
    else if (ts.isTemplateSpan(node) && !ts.isTaggedTemplateExpression(node.parent.parent) &&
        hasSoftAlternative(checker, checker.getTypeAtLocation(node.expression)))
      readout(node.expression, undefined, !soft(node.expression));
    else if ((ts.isSpreadElement(node) || ts.isSpreadAssignment(node)) && soft(node.expression)) opaque(node, 'it cannot be spread');
    else if ((ts.isForOfStatement(node) || ts.isForInStatement(node)) && soft(node.expression))
      opaque(node.expression, 'it cannot be iterated');
    else if (ts.isNewExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'Error' &&
        isDefaultGlobal(checker, node.expression) && node.arguments?.length &&
        hasSoftAlternative(checker, checker.getTypeAtLocation(node.arguments[0]!)))
      readout(node, 'error', !soft(node.arguments[0]!));
    else if (ts.isCallExpression(node)) {
      const callee = node.expression;
      if (ts.isIdentifier(callee) && callee.text === 'Error' && isDefaultGlobal(checker, callee) && node.arguments.length &&
          hasSoftAlternative(checker, checker.getTypeAtLocation(node.arguments[0]!)))
        readout(node, 'error', !soft(node.arguments[0]!));
      if (ts.isIdentifier(callee) && callee.text === 'String' && isDefaultString(checker, callee)) {
        const first = node.arguments[0];
        if (first && (hasSoftAlternative(checker, checker.getTypeAtLocation(first)) || assertedSoftSourceType(checker, first)))
          readout(first, undefined, !soft(first));
      }
      if (ts.isIdentifier(callee) && (callee.text === 'Number' || callee.text === 'Boolean') &&
          isDefaultGlobal(checker, callee) && node.arguments.length > 0) {
        const first = node.arguments[0]!;
        const type = checker.getTypeAtLocation(first);
        if (hasSoftAlternative(checker, type)) {
          if (scalarSoftAlternatives(type))
            readout(node, 'scalar-conversion', !soft(first), 0, callee.text as 'Number' | 'Boolean');
          else opaque(node, `${callee.text}() cannot convert an opaque Neuralese value; read a scalar value explicitly`);
        }
      }
      if (ts.isPropertyAccessExpression(callee) && ts.isIdentifier(callee.expression) && callee.expression.text === 'JSON' &&
          callee.name.text === 'stringify' && isDefaultGlobal(checker, callee.expression)) {
        const [value, ...options] = node.arguments;
        if (value && hasSoftAlternative(checker, checker.getTypeAtLocation(value)))
          readout(node, 'json', !soft(value));
        if (options.some(argument => soft(argument))) opaque(node, 'its payload cannot be serialised');
      }
      if (ts.isPropertyAccessExpression(callee) && callee.name.text === 'concat' && standardMethod(callee, ['String']) &&
          !!(checker.getTypeAtLocation(callee.expression).flags & ts.TypeFlags.StringLike)) {
        let softArgument = false;
        for (const argument of node.arguments) {
          // The call's existing concat readout lowering materializes argument spreads
          // before any awaited reads, then coerces each expanded argument in order.
          if (ts.isSpreadElement(argument) && arrayJoinKind(argument.expression)) softArgument = true;
          else if (hasSoftAlternative(checker, checker.getTypeAtLocation(argument))) softArgument = true;
        }
        if (softArgument) readout(node, 'concat');
      }
      const joinKind = softArrayJoinKind(node);
      if (joinKind === 'supported') readout(node, 'join');
      const arrayStringKind = softArrayToStringKind(node);
      if (arrayStringKind === 'supported') readout(node, 'array-string');
      const stringArgument = stringTextArgument(node);
      if (stringArgument !== undefined) {
        const argument = node.arguments[stringArgument]!;
        readout(node, 'string-argument', !soft(argument), stringArgument);
      }
      if (ts.isPropertyAccessExpression(callee)) {
        const replace = stringReplaceKind(callee);
        if (replace) readout(node, 'string-replace', replace.conditional);
      }
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
  planControlGuards(file);
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
