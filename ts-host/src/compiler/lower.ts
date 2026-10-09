/**
 * The natlang lowering transformer, shared by project builds and runtime module loading.
 *
 * - `nl` tagged templates become `__natlang.inline(plan, values, accessors, context)`.
 * - `x.iterateOn(...)` and `iterateOn(...)` get a stable call-site identity.
 * - In constrained code (callable folders, `natlang.d/`), authored functions get recursion entry
 *   guards and `for ... of` sources are wrapped in the finite-iteration guard.
 * - For browser targets, every `await` restores the natlang task context on resumption.
 */
import ts from 'typescript';
import type { InlineLambdaPlan } from './inline.js';
import { resolveIntrinsic, suffixWithCallOf } from './inline.js';
import { authoredCallables, finiteCounterComparison, guardArguments, loopLabel, makesCalls } from './policy.js';

/** The predicates of an `Is<T, P>` type (nested refinements give several), read from its brand property. */
export function refinementPredicates(checker: ts.TypeChecker, node: ts.TypeNode): string[] {
  const type = checker.getTypeFromTypeNode(node);
  const brand = checker.getPropertiesOfType(type).find(property => property.name === '__natlangRefinement');
  return brand ? checker.getPropertiesOfType(checker.getTypeOfSymbol(brand)).map(property => property.name) : [];
}

export type LowerOptions = {
  /** Plans for this file, keyed by `start:end` of the tagged template in the original source. */
  plans: ReadonlyMap<string, InlineLambdaPlan>;
  checker?: ts.TypeChecker;
  /** Source spans of typed Neuralese values in JavaScript text-coercion positions. */
  readouts?: ReadonlySet<string>;
  /** Readout spans whose declared union also permits crisp values. */
  conditionalReadouts?: ReadonlySet<string>;
  /** Call spans of array joins whose element type is Neuralese<string>. */
  joins?: ReadonlySet<string>;
  /** Call spans of Array#toString whose array element type contains Neuralese values. */
  arrayStrings?: ReadonlySet<string>;
  /** Calls whose inline synchronous Array#map callback contains a typed async readout. */
  arrayMaps?: ReadonlySet<string>;
  /** Callback spans for those calls; their recursion guard must retain the generated async boundary. */
  arrayMapCallbacks?: ReadonlySet<string>;
  /** Call spans of string concatenations whose arguments include typed Neuralese values. */
  concats?: ReadonlySet<string>;
  /** Call spans of JSON.stringify whose first value argument needs typed Neuralese readout. */
  jsons?: ReadonlySet<string>;
  /** Call/new spans of the default Error constructor whose message needs typed Neuralese readout. */
  errors?: ReadonlySet<string>;
  /** Standard String method calls whose listed text argument needs typed Neuralese readout. */
  stringArguments?: ReadonlyMap<string, number>;
  /** Explicit scalar constructors that need a conditional typed Neuralese readout of argument zero. */
  scalarConversions?: ReadonlyMap<string, { argument: number; conversion: 'Number' | 'Boolean'; conditional?: true }>;
  /** Standard String#replace calls whose Neuralese string receiver is materialized before lookup. */
  stringReplaces?: ReadonlyMap<string, { conditional?: true }>;
  /** Calls of named functions that are part of a possible hand-off, by `start:end` in the source, with their site IDs. */
  fusionSites?: ReadonlyMap<string, string>;
  /** Identifier through which lowered code reaches the runtime support functions. */
  runtime: string;
  /** Expression text giving the callable context for inline lambdas, if any. */
  context?: string;
  constrained: boolean;
  guardPrefix: string;
  /** Project-relative module path used in call-site IDs. */
  modulePath: string;
  browser?: boolean;
  /** Project builds: `.nl` imports and runtime specifier rewriting. */
  module?: {
    /** `.nl` import specifiers of this file (each loads a generated `<specifier>.js` module). */
    natlangImports: ReadonlySet<string>;
    /** Module specifiers rewritten in emitted imports (for example `@natlang/node` to the running runtime). */
    rewrite?: ReadonlyMap<string, string>;
  };
};

const literal = (value: unknown): ts.Expression => ts.factory.createIdentifier(JSON.stringify(value));

function enclosingSymbolName(node: ts.Node): string {
  for (let current: ts.Node | undefined = node.parent; current; current = current.parent) {
    if ((ts.isFunctionDeclaration(current) || ts.isClassDeclaration(current) || ts.isMethodDeclaration(current)) && current.name)
      return current.name.getText();
    if (ts.isVariableDeclaration(current) && ts.isIdentifier(current.name)) return current.name.text;
  }
  return 'module';
}

export function natlangTransformer(options: LowerOptions): ts.TransformerFactory<ts.SourceFile> {
  return context => file => {
    const f = context.factory;
    const runtime = (name: string) => f.createPropertyAccessExpression(f.createIdentifier(options.runtime), name);
    const authored = options.constrained ? authoredCallables(file, options.guardPrefix) : [];
    const guardIds = new Map(authored.map(callable => [callable.node, callable.id]));
    const siteCounts = new Map<string, number>();
    const siteId = (node: ts.Node) => {
      const base = `${options.modulePath}#${enclosingSymbolName(node)}`;
      const count = (siteCounts.get(base) ?? 0) + 1;
      siteCounts.set(base, count);
      return count === 1 ? base : `${base}@${count}`;
    };
    const original = (node: ts.Node) => ts.getOriginalNode(node);
    const readNeuraleseValue = (value: ts.Expression, conditional: boolean | undefined): ts.Expression => {
      const result = f.createCallExpression(runtime(conditional ? 'readNeuraleseIfReference' : 'readNeuralese'), undefined, [value]);
      if (options.browser) {
        const bound = f.createCallExpression(f.createPropertyAccessExpression(f.createIdentifier('globalThis'), '__natlang_bindAwait'),
          undefined, [result]);
        return f.createCallExpression(f.createParenthesizedExpression(f.createAwaitExpression(bound)), undefined, []);
      }
      return f.createAwaitExpression(result);
    };
    const readNeuralese = (expression: ts.Expression): ts.Expression => {
      const lowered = ts.visitEachChild(expression, visit, context) as ts.Expression;
      const conditional = options.conditionalReadouts?.has(`${original(expression).getStart(file)}:${original(expression).getEnd()}`);
      return readNeuraleseValue(lowered, conditional);
    };

    const visit = (node: ts.Node): ts.Node => {
      const source = original(node);
      if (ts.isCallExpression(node) && ts.isCallExpression(source) && ts.isPropertyAccessExpression(node.expression) &&
          options.arrayMaps?.has(`${source.getStart(file)}:${source.getEnd()}`) && node.arguments[0]) {
        const receiver = f.createUniqueName('__natlang_map_receiver');
        const method = f.createUniqueName('__natlang_map_method');
        const args = f.createUniqueName('__natlang_map_args');
        const callback = ts.visitNode(node.arguments[0], visit) as ts.Expression;
        const makeAsync = (expression: ts.Expression): ts.Expression => {
          if (ts.isParenthesizedExpression(expression)) return f.updateParenthesizedExpression(expression, makeAsync(expression.expression));
          if (ts.isArrowFunction(expression)) {
            const modifiers = [...(expression.modifiers ?? [])];
            if (!modifiers.some(item => item.kind === ts.SyntaxKind.AsyncKeyword)) modifiers.unshift(f.createModifier(ts.SyntaxKind.AsyncKeyword));
            return f.updateArrowFunction(expression, modifiers, expression.typeParameters, expression.parameters,
              expression.type, expression.equalsGreaterThanToken, expression.body);
          }
          if (ts.isFunctionExpression(expression)) {
            const modifiers = [...(expression.modifiers ?? [])];
            if (!modifiers.some(item => item.kind === ts.SyntaxKind.AsyncKeyword)) modifiers.unshift(f.createModifier(ts.SyntaxKind.AsyncKeyword));
            return f.updateFunctionExpression(expression, modifiers, expression.asteriskToken, expression.name,
              expression.typeParameters, expression.parameters, expression.type, expression.body);
          }
          return expression;
        };
        const asyncCallback = makeAsync(callback);
        const body = f.createBlock([
          f.createVariableStatement(undefined, f.createVariableDeclarationList([
            f.createVariableDeclaration(method, undefined, undefined, f.createPropertyAccessExpression(receiver, 'map'))], ts.NodeFlags.Const)),
          f.createVariableStatement(undefined, f.createVariableDeclarationList([
            f.createVariableDeclaration(args, undefined, undefined, f.createArrayLiteralExpression([
              asyncCallback, ...node.arguments.slice(1).map(argument => ts.visitNode(argument, visit) as ts.Expression)]))], ts.NodeFlags.Const)),
          f.createReturnStatement(f.createCallExpression(runtime('mapNeuraleseReadout'), undefined, [receiver, method, args])),
        ], true);
        const lower = f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction(undefined, undefined,
          [f.createParameterDeclaration(undefined, undefined, receiver)], undefined, undefined, body)), undefined,
          [ts.visitNode(node.expression.expression, visit) as ts.Expression]);
        if (options.browser) {
          const bound = f.createCallExpression(f.createPropertyAccessExpression(f.createIdentifier('globalThis'), '__natlang_bindAwait'),
            undefined, [lower]);
          return f.createCallExpression(f.createParenthesizedExpression(f.createAwaitExpression(bound)), undefined, []);
        }
        return f.createAwaitExpression(lower);
      }
      if ((ts.isCallExpression(node) || ts.isNewExpression(node)) &&
          (ts.isCallExpression(source) || ts.isNewExpression(source)) &&
          options.errors?.has(`${source.getStart(file)}:${source.getEnd()}`) && node.arguments?.length) {
        const callee = f.createUniqueName('__natlang_error_constructor');
        const args = f.createUniqueName('__natlang_error_args');
        const message = readNeuraleseValue(f.createElementAccessExpression(args, f.createNumericLiteral(0)),
          options.conditionalReadouts?.has(`${source.getStart(file)}:${source.getEnd()}`));
        const rest = f.createSpreadElement(f.createCallExpression(f.createPropertyAccessExpression(args, 'slice'), undefined,
          [f.createNumericLiteral(1)]));
        const invoke = ts.isNewExpression(node) ? f.createNewExpression(callee, undefined, [message, rest]) :
          f.createCallExpression(callee, undefined, [message, rest]);
        const inner = f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction([f.createModifier(ts.SyntaxKind.AsyncKeyword)],
          undefined, [f.createParameterDeclaration(undefined, undefined, args)], undefined, undefined, invoke)), undefined,
          [f.createArrayLiteralExpression(node.arguments.map(argument => ts.visitNode(argument, visit) as ts.Expression))]);
        const outer = f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction(undefined, undefined,
          [f.createParameterDeclaration(undefined, undefined, callee)], undefined, undefined, inner)), undefined,
          [ts.visitNode(node.expression, visit) as ts.Expression]);
        return f.createAwaitExpression(outer);
      }
      if (ts.isCallExpression(node) && ts.isCallExpression(source) &&
          options.scalarConversions?.has(`${source.getStart(file)}:${source.getEnd()}`) && node.arguments.length) {
        const site = options.scalarConversions.get(`${source.getStart(file)}:${source.getEnd()}`)!;
        const callee = f.createUniqueName('__natlang_scalar_constructor');
        const args = f.createUniqueName('__natlang_scalar_args');
        const reader = site.conditional ? 'readNeuraleseIfReference' : 'readNeuralese';
        const convert = f.createBinaryExpression(f.createElementAccessExpression(args, f.createNumericLiteral(site.argument)),
          f.createToken(ts.SyntaxKind.EqualsToken), f.createAwaitExpression(f.createCallExpression(runtime(reader), undefined,
            [f.createElementAccessExpression(args, f.createNumericLiteral(site.argument))])));
        const invoke = f.createCallExpression(callee, undefined, [f.createSpreadElement(args)]);
        const inner = f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction([f.createModifier(ts.SyntaxKind.AsyncKeyword)],
          undefined, [f.createParameterDeclaration(undefined, undefined, args)], undefined, undefined,
          f.createBlock([f.createExpressionStatement(convert), f.createReturnStatement(invoke)], true))), undefined,
          [f.createArrayLiteralExpression(node.arguments.map(argument => ts.visitNode(argument, visit) as ts.Expression))]);
        return f.createAwaitExpression(f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction(undefined, undefined,
          [f.createParameterDeclaration(undefined, undefined, callee)], undefined, undefined, inner)), undefined,
          [ts.visitNode(node.expression, visit) as ts.Expression]));
      }
      if (ts.isCallExpression(node) && ts.isCallExpression(source) && ts.isPropertyAccessExpression(node.expression) &&
          options.stringReplaces?.has(`${source.getStart(file)}:${source.getEnd()}`)) {
        const site = options.stringReplaces.get(`${source.getStart(file)}:${source.getEnd()}`)!;
        const receiver = f.createUniqueName('__natlang_replace_receiver');
        const text = f.createUniqueName('__natlang_replace_text');
        const method = f.createUniqueName('__natlang_replace_method');
        const args = f.createUniqueName('__natlang_replace_args');
        const read = readNeuraleseValue(receiver, site.conditional);
        const result = f.createObjectLiteralExpression([f.createPropertyAssignment('value', f.createCallExpression(
          runtime('invokeWithReceiver'), undefined, [method, text, args]))]);
        const body = f.createBlock([
          f.createVariableStatement(undefined, f.createVariableDeclarationList([
            f.createVariableDeclaration(text, undefined, undefined, read)], ts.NodeFlags.Const)),
          f.createVariableStatement(undefined, f.createVariableDeclarationList([
            f.createVariableDeclaration(method, undefined, undefined, f.createPropertyAccessExpression(text, 'replace'))], ts.NodeFlags.Const)),
          f.createVariableStatement(undefined, f.createVariableDeclarationList([
            f.createVariableDeclaration(args, undefined, undefined, f.createArrayLiteralExpression(
              node.arguments.map(argument => ts.visitNode(argument, visit) as ts.Expression)))], ts.NodeFlags.Const)),
          f.createReturnStatement(result),
        ], true);
        const invocation = f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction(
          [f.createModifier(ts.SyntaxKind.AsyncKeyword)], undefined,
          [f.createParameterDeclaration(undefined, undefined, receiver)], undefined, undefined, body)), undefined,
          [ts.visitNode(node.expression.expression, visit) as ts.Expression]);
        return f.createPropertyAccessExpression(f.createAwaitExpression(invocation), 'value');
      }
      if (ts.isCallExpression(node) && ts.isCallExpression(source) && ts.isPropertyAccessExpression(node.expression) &&
          options.stringArguments?.has(`${source.getStart(file)}:${source.getEnd()}`)) {
        const receiver = f.createUniqueName('__natlang_text_receiver');
        const method = f.createUniqueName('__natlang_text_method');
        const args = f.createUniqueName('__natlang_text_args');
        const index = options.stringArguments.get(`${source.getStart(file)}:${source.getEnd()}`)!;
        const reader = readNeuraleseValue(f.createElementAccessExpression(args, f.createNumericLiteral(index)),
          options.conditionalReadouts?.has(`${source.getStart(file)}:${source.getEnd()}`));
        const invoke = f.createCallExpression(f.createPropertyAccessExpression(method, 'call'), undefined,
          [receiver, f.createSpreadElement(args)]);
        const body = f.createBlock([
          f.createExpressionStatement(f.createBinaryExpression(f.createElementAccessExpression(args, f.createNumericLiteral(index)),
            f.createToken(ts.SyntaxKind.EqualsToken), reader)),
          f.createReturnStatement(invoke),
        ], true);
        const withArgs = f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction([f.createModifier(ts.SyntaxKind.AsyncKeyword)],
          undefined, [f.createParameterDeclaration(undefined, undefined, method), f.createParameterDeclaration(undefined, undefined, args)],
          undefined, undefined, body)), undefined,
          [f.createPropertyAccessExpression(receiver, node.expression.name),
            f.createArrayLiteralExpression(node.arguments.map(argument => ts.visitNode(argument, visit) as ts.Expression))]);
        const applyReceiver = f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction(undefined, undefined,
          [f.createParameterDeclaration(undefined, undefined, receiver)], undefined, undefined, withArgs)), undefined,
          [ts.visitNode(node.expression.expression, visit) as ts.Expression]);
        return f.createAwaitExpression(applyReceiver);
      }
      if (ts.isCallExpression(node) && ts.isCallExpression(source) && options.jsons?.has(`${source.getStart(file)}:${source.getEnd()}`) &&
          ts.isPropertyAccessExpression(node.expression) && node.arguments.length > 0) {
        const readoutCall = `${source.getStart(file)}:${source.getEnd()}`;
        const conditional = options.conditionalReadouts?.has(readoutCall);
        const receiver = f.createUniqueName('__natlang_json_receiver');
        const call = f.createUniqueName('__natlang_json_call');
        const args = f.createUniqueName('__natlang_json_args');
        const softValue = f.createElementAccessExpression(args, f.createNumericLiteral(0));
        const read = readNeuraleseValue(softValue, conditional);
        const invoke = f.createCallExpression(f.createPropertyAccessExpression(f.createElementAccessExpression(call, 1), 'call'), undefined,
          [f.createElementAccessExpression(call, 0), read,
            ...node.arguments.slice(1).map((_argument, index) => f.createElementAccessExpression(args, index + 1))]);
        const inner = f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction([f.createModifier(ts.SyntaxKind.AsyncKeyword)],
          undefined, [f.createParameterDeclaration(undefined, undefined, call), f.createParameterDeclaration(undefined, undefined, args)],
          undefined, undefined, invoke)), undefined, [
          f.createArrayLiteralExpression([receiver, f.createPropertyAccessExpression(receiver, 'stringify')]),
          f.createArrayLiteralExpression(node.arguments.map(argument => ts.visitNode(argument, visit) as ts.Expression))]);
        const lower = f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction(undefined, undefined,
          [f.createParameterDeclaration(undefined, undefined, receiver)], undefined, undefined, inner)), undefined,
          [ts.visitNode(node.expression.expression, visit) as ts.Expression]);
        // The receiver, method, and argument values are captured in native order before awaiting the soft value.
        return f.createAwaitExpression(lower);
      }
      if (ts.isCallExpression(node) && ts.isCallExpression(source) && options.joins?.has(`${source.getStart(file)}:${source.getEnd()}`) &&
          ts.isPropertyAccessExpression(node.expression)) {
        const receiver = f.createUniqueName('__natlang_join_receiver');
        const method = f.createUniqueName('__natlang_join_method');
        const args = f.createUniqueName('__natlang_join_args');
        const value = f.createUniqueName('__natlang_join_value');
        const reader = f.createArrowFunction([f.createModifier(ts.SyntaxKind.AsyncKeyword)], undefined,
          [f.createParameterDeclaration(undefined, undefined, value)], undefined, undefined, readNeuralese(value));
        const helper = f.createCallExpression(runtime('joinNeuralese'), undefined, [receiver, method, args, reader]);
        const body = f.createPropertyAccessExpression(f.createAwaitExpression(helper), 'value');
        const invoke = f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction([f.createModifier(ts.SyntaxKind.AsyncKeyword)],
          undefined, [f.createParameterDeclaration(undefined, undefined, method), f.createParameterDeclaration(undefined, undefined, args)],
          undefined, undefined, body)), undefined, [f.createPropertyAccessExpression(receiver, 'join'),
          f.createArrayLiteralExpression(node.arguments.map(argument => ts.visitNode(argument, visit) as ts.Expression))]);
        const lower = f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction(undefined, undefined,
          [f.createParameterDeclaration(undefined, undefined, receiver)], undefined, undefined, invoke)), undefined,
          [ts.visitNode(node.expression.expression, visit) as ts.Expression]);
        return f.createAwaitExpression(lower);
      }
      if (ts.isCallExpression(node) && ts.isCallExpression(source) && options.arrayStrings?.has(`${source.getStart(file)}:${source.getEnd()}`) &&
          ts.isPropertyAccessExpression(node.expression)) {
        const receiver = f.createUniqueName('__natlang_array_string_receiver');
        const method = f.createUniqueName('__natlang_array_string_method');
        const args = f.createUniqueName('__natlang_array_string_args');
        const value = f.createUniqueName('__natlang_array_string_value');
        const reader = f.createArrowFunction([f.createModifier(ts.SyntaxKind.AsyncKeyword)], undefined,
          [f.createParameterDeclaration(undefined, undefined, value)], undefined, undefined, readNeuralese(value));
        const helper = f.createCallExpression(runtime('arrayToStringNeuralese'), undefined, [receiver, method, args, reader]);
        const body = f.createPropertyAccessExpression(f.createAwaitExpression(helper), 'value');
        const invoke = f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction([f.createModifier(ts.SyntaxKind.AsyncKeyword)],
          undefined, [f.createParameterDeclaration(undefined, undefined, method), f.createParameterDeclaration(undefined, undefined, args)],
          undefined, undefined, body)), undefined, [f.createPropertyAccessExpression(receiver, 'toString'),
          f.createArrayLiteralExpression(node.arguments.map(argument => ts.visitNode(argument, visit) as ts.Expression))]);
        const lower = f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction(undefined, undefined,
          [f.createParameterDeclaration(undefined, undefined, receiver)], undefined, undefined, invoke)), undefined,
          [ts.visitNode(node.expression.expression, visit) as ts.Expression]);
        return f.createAwaitExpression(lower);
      }
      if (ts.isCallExpression(node) && ts.isCallExpression(source) && options.concats?.has(`${source.getStart(file)}:${source.getEnd()}`) &&
          ts.isPropertyAccessExpression(node.expression)) {
        const receiver = f.createUniqueName('__natlang_concat_receiver');
        const method = f.createUniqueName('__natlang_concat_method');
        const values = f.createUniqueName('__natlang_concat_values');
        const value = f.createUniqueName('__natlang_concat_value');
        const reader = f.createArrowFunction([f.createModifier(ts.SyntaxKind.AsyncKeyword)], undefined,
          [f.createParameterDeclaration(undefined, undefined, value)], undefined, undefined, readNeuralese(value));
        const statements: ts.Statement[] = [
          f.createVariableStatement(undefined, f.createVariableDeclarationList([
            f.createVariableDeclaration(method, undefined, undefined, f.createPropertyAccessExpression(receiver, 'concat'))], ts.NodeFlags.Const)),
          f.createVariableStatement(undefined, f.createVariableDeclarationList([
            f.createVariableDeclaration(values, undefined, undefined, f.createArrayLiteralExpression(
              node.arguments.map(argument => ts.visitNode(argument, visit) as ts.Expression)))], ts.NodeFlags.Const)),
          f.createReturnStatement(f.createCallExpression(runtime('concatNeuralese'), undefined, [receiver, method, values, reader]))];
        const invoke = f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction([f.createModifier(ts.SyntaxKind.AsyncKeyword)], undefined,
          [f.createParameterDeclaration(undefined, undefined, receiver)], undefined, undefined, f.createBlock(statements))), undefined,
          [ts.visitNode(node.expression.expression, visit) as ts.Expression]);
        return f.createAwaitExpression(invoke);
      }
      if (options.readouts?.has(`${source.getStart(file)}:${source.getEnd()}`) && ts.isExpression(node))
        return readNeuralese(node);
      // A call that is part of a possible hand-off between named functions: the arguments are evaluated where they stand,
      // the call itself runs under its site ID (runtime/fusion.ts matches it against the planned edges).
      if (ts.isCallExpression(node) && ts.isCallExpression(source) && !source.questionDotToken && options.fusionSites?.has(`${source.getStart(file)}:${source.getEnd()}`)) {
        const visited = ts.visitEachChild(node, visit, context);
        const rest = f.createUniqueName('__natlang_fuse_args');
        const call = f.createArrowFunction(undefined, undefined, [f.createParameterDeclaration(undefined, f.createToken(ts.SyntaxKind.DotDotDotToken), rest)],
          undefined, undefined, f.createCallExpression(visited.expression, undefined, [f.createSpreadElement(rest)]));
        return f.createCallExpression(runtime('fuseSite'), undefined,
          [f.createStringLiteral(options.fusionSites.get(`${source.getStart(file)}:${source.getEnd()}`)!), call, ...visited.arguments]);
      }
      // `nl<Result>`instructions`.with({ ... })` is equivalent to the explicit-capture tag form.
      // Revisit it with the existing explicitInline lowering, which evaluates interpolation expressions before
      // reading snapshot captures and builds live capture accessors.
      if (ts.isCallExpression(node) && ts.isCallExpression(source) && ts.isPropertyAccessExpression(source.expression) &&
          source.expression.name.text === 'with' && ts.isTaggedTemplateExpression(source.expression.expression)) {
        const tagged = source.expression.expression;
        const span = `${tagged.getStart(file)}:${tagged.getEnd()}`;
        const plan = options.plans.get(span);
        if (plan?.explicitCaptures && suffixWithCallOf(tagged) === source) {
          const tag = f.createCallExpression(f.createPropertyAccessExpression(ts.visitNode(tagged.tag, visit) as ts.Expression,
            'with'), undefined, source.arguments.map(argument => ts.visitNode(argument, visit) as ts.Expression));
          return visit(f.updateTaggedTemplateExpression(tagged, tag, tagged.typeArguments, tagged.template));
        }
      }
      // Inline natlang lambdas.
      if (ts.isTaggedTemplateExpression(node) && ts.isTaggedTemplateExpression(source)) {
        const plan = options.plans.get(`${source.getStart(file)}:${source.getEnd()}`);
        if (plan?.explicitCaptures) {
          // `nl.with({ a, b: expr, n: live(n) })`...``: getters for snapshots (read once, when the function is created)
          // and getter/setter pairs for live lets; the receiver is the `nl` the call is made on.
          const withCall = ts.isCallExpression(node.tag) ? node.tag : undefined;
          const receiver = withCall && ts.isPropertyAccessExpression(withCall.expression) ?
            ts.visitNode(withCall.expression.expression, visit) as ts.Expression : f.createIdentifier('nl');
          const listing = withCall?.arguments[0];
          const properties = listing && ts.isObjectLiteralExpression(listing) ? listing.properties : f.createNodeArray<ts.ObjectLiteralElementLike>();
          const recordTemp = listing && !ts.isObjectLiteralExpression(listing) ? f.createUniqueName('__natlang_capture_record') : undefined;
          const accessors = plan.captures.map(capture => {
            const property = properties.find(item => (ts.isShorthandPropertyAssignment(item) || ts.isPropertyAssignment(item)) &&
              (ts.isIdentifier(item.name) || ts.isStringLiteral(item.name)) && item.name.text === capture.name);
            const valueOf = (item: ts.ObjectLiteralElementLike | undefined): ts.Expression => recordTemp ?
              f.createPropertyAccessExpression(recordTemp, capture.name) : !item ? f.createIdentifier('undefined') :
              ts.isShorthandPropertyAssignment(item) ? f.createIdentifier(item.name.text) :
              ts.visitNode((item as ts.PropertyAssignment).initializer, visit) as ts.Expression;
            const getter = f.createArrowFunction(undefined, undefined, [], undefined, undefined,
              capture.mode === 'live' ? f.createIdentifier(capture.name) : valueOf(property));
            const setter = capture.mode === 'live' ? [f.createArrowFunction(undefined, undefined,
              [f.createParameterDeclaration(undefined, undefined, '__natlang_value')], undefined, undefined,
              f.createBlock([f.createExpressionStatement(f.createAssignment(f.createIdentifier(capture.name),
                f.createIdentifier('__natlang_value')))]))] : [];
            return f.createPropertyAssignment(capture.name, f.createArrayLiteralExpression([getter, ...setter]));
          });
          const accessorObject = f.createObjectLiteralExpression(accessors);
          const accessorExpression = recordTemp && listing ? f.createCallExpression(f.createParenthesizedExpression(
            f.createArrowFunction(undefined, undefined, [f.createParameterDeclaration(undefined, undefined, recordTemp)], undefined, undefined,
              accessorObject)), undefined, [ts.visitNode(listing, visit) as ts.Expression]) : accessorObject;
          const values = !plan.softBody && ts.isTemplateExpression(node.template) ?
            node.template.templateSpans.map(span => ts.visitNode(span.expression, visit) as ts.Expression) : [];
          return f.createCallExpression(f.createPropertyAccessExpression(receiver, '__inline'), undefined, [literal(plan),
            f.createArrayLiteralExpression(values), accessorExpression,
            options.context ? f.createIdentifier(options.context) : f.createIdentifier('undefined')]);
        }
        if (plan) {
          const values = ts.isTemplateExpression(node.template) ?
            node.template.templateSpans.map(span => ts.visitNode(span.expression, visit) as ts.Expression) : [];
          const accessors = plan.captures.map(capture => {
            const getter = f.createArrowFunction(undefined, undefined, [], undefined, undefined, f.createIdentifier(capture.name));
            const setter = capture.mutable ? [f.createArrowFunction(undefined, undefined,
              [f.createParameterDeclaration(undefined, undefined, '__natlang_value')], undefined, undefined,
              f.createBlock([f.createExpressionStatement(f.createAssignment(f.createIdentifier(capture.name),
                f.createIdentifier('__natlang_value')))]))] : [];
            return f.createPropertyAssignment(capture.name, f.createArrayLiteralExpression([getter, ...setter]));
          });
          // Reach the runtime through the file's own `nl` reference, so module transforms rebind it correctly.
          const tag = ts.visitNode(node.tag, visit) as ts.Expression;
          return f.createCallExpression(f.createPropertyAccessExpression(tag, '__inline'), undefined, [literal(plan),
            f.createArrayLiteralExpression(values), f.createObjectLiteralExpression(accessors),
            options.context ? f.createIdentifier(options.context) : f.createIdentifier('undefined')]);
        }
      }
      // iterateOn call sites.
      if (ts.isCallExpression(node) && ts.isCallExpression(source)) {
        const callee = source.expression;
        const isTypedRead = options.checker && resolveIntrinsic(options.checker, callee) === 'read';
        if (isTypedRead) {
          const args = node.arguments.map(argument => ts.visitNode(argument, visit) as ts.Expression);
          return f.createCallExpression(runtime('readNeuralese'), undefined, args);
        }
        // refine<Is<T, "p">>(value) and assume<R>(value): the predicate of the type argument becomes the second argument.
        const refinement = options.checker && source.typeArguments?.length === 1 && source.arguments.length === 1 ?
          resolveIntrinsic(options.checker, callee) : undefined;
        if (options.checker && (refinement === 'refine' || refinement === 'assume')) {
          const predicates = refinementPredicates(options.checker, source.typeArguments![0]!);
          if (predicates.length) return f.createCallExpression(ts.visitNode(callee, visit) as ts.Expression, undefined,
            [ts.visitNode(node.arguments[0]!, visit) as ts.Expression, f.createStringLiteral(predicates.join('; and '))]);
        }
        const isMethod = ts.isPropertyAccessExpression(callee) && callee.name.text === 'iterateOn';
        const isFree = !isMethod && options.checker && resolveIntrinsic(options.checker, callee) === 'iterateOn';
        const isFreeUnchecked = !isMethod && !options.checker && ts.isIdentifier(callee) && callee.text === 'iterateOn';
        if (isMethod || isFree || isFreeUnchecked) {
          const lowered = ts.visitEachChild(node, visit, context);
          // Self-contained: attach the call site's identity when the result is a natlang Iteration.
          const value = f.createUniqueName('__natlang_iteration');
          return f.createCallExpression(f.createParenthesizedExpression(f.createArrowFunction(undefined, undefined,
            [f.createParameterDeclaration(undefined, undefined, value)], undefined, undefined,
            f.createConditionalExpression(f.createBinaryExpression(f.createTypeOfExpression(f.createPropertyAccessChain(value,
              f.createToken(ts.SyntaxKind.QuestionDotToken), 'withCompilerSite')), ts.SyntaxKind.EqualsEqualsEqualsToken,
              f.createStringLiteral('function')), undefined,
              f.createCallExpression(f.createPropertyAccessExpression(value, 'withCompilerSite'), undefined, [f.createStringLiteral(siteId(source))]),
              undefined, value))), undefined, [lowered]);
        }
      }
      // A numeric loop has a fixed finite bound and a strictly advancing counter.
      if (options.constrained && ts.isForStatement(node) && node.initializer &&
          ts.isVariableDeclarationList(node.initializer) && node.initializer.declarations.length === 1 &&
          node.condition) {
        const declaration = node.initializer.declarations[0]!;
        if (ts.isIdentifier(declaration.name)) {
          const counter = declaration.name;
          const boundedCondition = finiteCounterComparison(node.condition, counter.text);
          if (!boundedCondition) return ts.visitEachChild(node, visit, context);
          const comparison = boundedCondition.comparison;
          const left = ts.isIdentifier(comparison.left) && comparison.left.text === counter.text;
          const bound = f.createUniqueName('__natlang_bound');
          const progress = f.createUniqueName('__natlang_progress');
          const kind = comparison.operatorToken.kind;
          const upward = left ? kind === ts.SyntaxKind.LessThanToken || kind === ts.SyntaxKind.LessThanEqualsToken :
            kind === ts.SyntaxKind.GreaterThanToken || kind === ts.SyntaxKind.GreaterThanEqualsToken;
          const initializer = f.updateVariableDeclarationList(node.initializer, [
            ts.visitNode(declaration, visit) as ts.VariableDeclaration,
            f.createVariableDeclaration(bound, undefined, undefined, ts.visitNode(left ? comparison.right : comparison.left, visit) as ts.Expression),
            f.createVariableDeclaration(progress, undefined, undefined,
              f.createCallExpression(runtime('numericProgress'), undefined, [counter, bound, upward ? f.createTrue() : f.createFalse()]))]);
          const body = ts.visitNode(node.statement, visit) as ts.Statement;
          const condition = ts.visitNode(node.condition, function replaceBoundedComparison(child: ts.Node): ts.Node {
            if (child === comparison)
              return f.updateBinaryExpression(comparison, left ? counter : bound, comparison.operatorToken,
                left ? bound : counter);
            return ts.visitEachChild(child, replaceBoundedComparison, context);
          }) as ts.Expression;
          return f.updateForStatement(node, initializer,
            condition,
            ts.visitNode(node.incrementor, visit) as ts.Expression | undefined,
            f.createBlock([f.createExpressionStatement(f.createCallExpression(progress, undefined, [counter])),
              ...(ts.isBlock(body) ? body.statements : [body])], true));
        }
      }
      // Finite iteration in constrained code.
      if (options.constrained && ts.isForOfStatement(node)) {
        let iterated = node.expression;
        while (ts.isParenthesizedExpression(iterated) || ts.isAsExpression(iterated) || ts.isTypeAssertionExpression(iterated) ||
            ts.isNonNullExpression(iterated)) iterated = iterated.expression;
        // `for (x of a.entries()/keys()/values())` iterates a snapshot-bounded view of the collection.
        if (!node.awaitModifier && ts.isCallExpression(iterated) && iterated.arguments.length === 0 &&
            ts.isPropertyAccessExpression(iterated.expression) && ['entries', 'keys', 'values'].includes(iterated.expression.name.text)) {
          const receiver = ts.visitNode(iterated.expression.expression, visit) as ts.Expression;
          return f.updateForOfStatement(node, node.awaitModifier, ts.visitNode(node.initializer, visit) as ts.ForInitializer,
            f.createCallExpression(runtime('finiteArrayIterator'), undefined,
              [receiver, f.createStringLiteral(iterated.expression.name.text), f.createStringLiteral(loopLabel(node.expression))]),
            ts.visitNode(node.statement, visit) as ts.Statement);
        }
        const expression = ts.visitNode(node.expression, visit) as ts.Expression;
        return f.updateForOfStatement(node, node.awaitModifier, ts.visitNode(node.initializer, visit) as ts.ForInitializer,
          f.createCallExpression(runtime(node.awaitModifier ? 'finiteAsync' : 'finite'), undefined,
            [expression, f.createStringLiteral(loopLabel(node.expression))]),
          ts.visitNode(node.statement, visit) as ts.Statement);
      }
      // Recursion entry guards on authored functions.
      const guardId = guardIds.get(source);
      if (guardId && (ts.isFunctionDeclaration(node) || ts.isFunctionExpression(node) || ts.isArrowFunction(node) ||
          ts.isMethodDeclaration(node) || ts.isGetAccessorDeclaration(node) || ts.isSetAccessorDeclaration(node)) && node.body &&
          makesCalls(source as ts.SignatureDeclaration)) {
        const visited = ts.visitEachChild(node, visit, context) as typeof node;
        const isAsync = !!visited.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.AsyncKeyword) ||
          options.arrayMapCallbacks?.has(`${source.getStart(source.getSourceFile())}:${source.getEnd()}`) === true;
        const body = visited.body!;
        const inner = f.createArrowFunction(isAsync ? [f.createModifier(ts.SyntaxKind.AsyncKeyword)] : undefined, undefined, [],
          undefined, undefined, ts.isBlock(body) ? body : f.createParenthesizedExpression(body as ts.Expression));
        const args = f.createArrayLiteralExpression(guardArguments(source as ts.SignatureDeclaration)
          .map(name => name === 'this' ? f.createThis() : f.createIdentifier(name)));
        const guarded = f.createCallExpression(runtime('guard'), undefined, [f.createStringLiteral(guardId), inner, args]);
        const block = f.createBlock([f.createReturnStatement(guarded)], true);
        if (ts.isFunctionDeclaration(visited)) return f.updateFunctionDeclaration(visited, visited.modifiers, visited.asteriskToken,
          visited.name, visited.typeParameters, visited.parameters, visited.type, block);
        if (ts.isFunctionExpression(visited)) return f.updateFunctionExpression(visited, visited.modifiers, visited.asteriskToken,
          visited.name, visited.typeParameters, visited.parameters, visited.type, block);
        if (ts.isMethodDeclaration(visited)) return f.updateMethodDeclaration(visited, visited.modifiers, visited.asteriskToken,
          visited.name, visited.questionToken, visited.typeParameters, visited.parameters, visited.type, block);
        if (ts.isGetAccessorDeclaration(visited)) return f.updateGetAccessorDeclaration(visited, visited.modifiers, visited.name,
          visited.parameters, visited.type, block);
        if (ts.isSetAccessorDeclaration(visited)) return f.updateSetAccessorDeclaration(visited, visited.modifiers, visited.name,
          visited.parameters, block);
        return f.updateArrowFunction(visited as ts.ArrowFunction, visited.modifiers, visited.typeParameters, visited.parameters,
          visited.type, (visited as ts.ArrowFunction).equalsGreaterThanToken, guarded);
      }
      // Browser: restore the task context after each await.
      if (options.browser && ts.isAwaitExpression(node)) {
        const operand = ts.visitNode(node.expression, visit) as ts.Expression;
        return f.createCallExpression(f.createParenthesizedExpression(f.createAwaitExpression(
          f.createCallExpression(f.createPropertyAccessExpression(f.createIdentifier('globalThis'), '__natlang_bindAwait'),
            undefined, [operand]))), undefined, []);
      }
      return ts.visitEachChild(node, visit, context);
    };
    const lowered = ts.visitNode(file, visit) as ts.SourceFile;
    const module = options.module;
    if (!module) return lowered;
    // `.nl` imports load a generated module next to the emitted file; bound runtime specifiers are rewritten.
    let changed = false;
    const statements = lowered.statements.map(statement => {
      if (!ts.isImportDeclaration(statement) || !ts.isStringLiteral(statement.moduleSpecifier)) return statement;
      const specifier = statement.moduleSpecifier.text;
      const target = module.natlangImports.has(specifier) ? `${specifier}.js` : module.rewrite?.get(specifier);
      if (!target) return statement;
      changed = true;
      return f.updateImportDeclaration(statement, statement.modifiers, statement.importClause, f.createStringLiteral(target), statement.attributes);
    });
    return changed ? f.updateSourceFile(lowered, statements) : lowered;
  };
}
