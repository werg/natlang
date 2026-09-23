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
import { resolveIntrinsic } from './inline.js';
import { authoredCallables } from './policy.js';

export type LowerOptions = {
  /** Plans for this file, keyed by `start:end` of the tagged template in the original source. */
  plans: ReadonlyMap<string, InlineLambdaPlan>;
  checker?: ts.TypeChecker;
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

    const visit = (node: ts.Node): ts.Node => {
      const source = original(node);
      // Inline natlang lambdas.
      if (ts.isTaggedTemplateExpression(node) && ts.isTaggedTemplateExpression(source)) {
        const plan = options.plans.get(`${source.getStart(file)}:${source.getEnd()}`);
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
      // Finite iteration in constrained code.
      if (options.constrained && ts.isForOfStatement(node) && !node.awaitModifier) {
        const expression = ts.visitNode(node.expression, visit) as ts.Expression;
        return f.updateForOfStatement(node, undefined, ts.visitNode(node.initializer, visit) as ts.ForInitializer,
          f.createCallExpression(runtime('finite'), undefined, [expression]), ts.visitNode(node.statement, visit) as ts.Statement);
      }
      // Recursion entry guards on authored functions.
      const guardId = guardIds.get(source);
      if (guardId && (ts.isFunctionDeclaration(node) || ts.isFunctionExpression(node) || ts.isArrowFunction(node) ||
          ts.isMethodDeclaration(node)) && node.body) {
        const visited = ts.visitEachChild(node, visit, context) as typeof node;
        const isAsync = !!visited.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.AsyncKeyword);
        const body = visited.body!;
        const inner = f.createArrowFunction(isAsync ? [f.createModifier(ts.SyntaxKind.AsyncKeyword)] : undefined, undefined, [],
          undefined, undefined, ts.isBlock(body) ? body : f.createParenthesizedExpression(body as ts.Expression));
        const guarded = f.createCallExpression(runtime('guard'), undefined, [f.createStringLiteral(guardId), inner]);
        const block = f.createBlock([f.createReturnStatement(guarded)], true);
        if (ts.isFunctionDeclaration(visited)) return f.updateFunctionDeclaration(visited, visited.modifiers, visited.asteriskToken,
          visited.name, visited.typeParameters, visited.parameters, visited.type, block);
        if (ts.isFunctionExpression(visited)) return f.updateFunctionExpression(visited, visited.modifiers, visited.asteriskToken,
          visited.name, visited.typeParameters, visited.parameters, visited.type, block);
        if (ts.isMethodDeclaration(visited)) return f.updateMethodDeclaration(visited, visited.modifiers, visited.asteriskToken,
          visited.name, visited.questionToken, visited.typeParameters, visited.parameters, visited.type, block);
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
