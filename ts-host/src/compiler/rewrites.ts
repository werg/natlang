/**
 * Law-based combinator rewrites (spec/NEURALESE_REWRITES.md): a source-to-source pass over TypeScript that applies
 * the approximate laws of the Neuralese combinators to make programs cheaper.
 *
 * A rule applies only when it is enabled for the current model and dialect version: a whole-program comparison of
 * rewritten against unrewritten executions on held-out programs found that it does no harm (judged by a rubric, not
 * a fixed threshold). Until such a comparison is recorded, the rule is off. Measurements are supplied to the
 * `RewriteGate`; this module does not run them.
 *
 * A rule matches only syntactic occurrences within one function body whose intermediate values have no other use.
 * It never moves a call across an effect (a call, an `await`, an assignment) or a capture write-back. A site whose
 * statement is marked `// natlang-no-rewrite` is left alone, and a whole program can be excluded.
 *
 * The combinators are recognised by name: `map`, `read`, `combine`, `empty`, `split`, `zip` (as identifiers or as the
 * last member of a property access), and `compose` for composition.
 */
import ts from 'typescript';
import { graphNode, type GraphInput } from '../native/graph.js';
import type { NativeTraceRecorder } from '../native/trace.js';

export const REWRITE_RULES = ['map-fusion', 'read-map', 'map-identity', 'combine-reassociate', 'combine-identity', 'split-zip'] as const;
export type RewriteRule = typeof REWRITE_RULES[number];
export const NO_REWRITE_MARKER = 'natlang-no-rewrite';

/**
 * One whole-program comparison of a rule, for one model and dialect version. `verdict` is the rubric's judgment of
 * the rewritten executions against the unrewritten ones on held-out programs.
 */
export type RuleMeasurement = { rule: RewriteRule; model: string; dialect: string; comparison: string;
  verdict: 'no-harm' | 'harm' | 'inconclusive'; rubric?: string; measuredAt?: string;
  /** For `read-map`: which side the comparison found cheaper (default forward: `read(map(v, f))` → `f(read(v))`). */
  direction?: 'forward' | 'reverse' };

/** Which rules are enabled for the current versions, and by which comparison. */
export class RewriteGate {
  readonly #enabled = new Map<RewriteRule, RuleMeasurement>();
  constructor(readonly version: { model: string; dialect: string }, measurements: readonly RuleMeasurement[] = [],
    options: { disabled?: readonly RewriteRule[] } = {}) {
    const disabled = new Set(options.disabled ?? []);
    // The latest comparison for these versions decides; one from another version never enables a rule.
    const latest = new Map<RewriteRule, RuleMeasurement>();
    for (const measurement of measurements) {
      if (measurement.model !== version.model || measurement.dialect !== version.dialect) continue;
      const previous = latest.get(measurement.rule);
      if (!previous || (measurement.measuredAt ?? '') >= (previous.measuredAt ?? '')) latest.set(measurement.rule, measurement);
    }
    for (const [rule, measurement] of latest) if (measurement.verdict === 'no-harm' && !disabled.has(rule)) this.#enabled.set(rule, measurement);
  }
  /** Every rule off: no comparisons yet. */
  static off(version = { model: 'unknown', dialect: 'unknown' }): RewriteGate { return new RewriteGate(version); }
  enabled(rule: RewriteRule): RuleMeasurement | undefined { return this.#enabled.get(rule); }
  /** The enabled set, recorded in each trace's manifest. */
  enabledRules(): RewriteRule[] { return [...this.#enabled.keys()].sort(); }
}

export type RewriteRecord = { rule: RewriteRule; site: string; enabled_by: string; replaced: string[] };
export type RewriteSkip = { rule: RewriteRule; site: string; reason: 'not-enabled' | 'disabled-site' | 'disabled-program' | 'effect' | 'other-use' };
export type RewriteResult = { code: string; applied: RewriteRecord[]; skipped: RewriteSkip[] };

const COMBINATORS = new Set(['map', 'read', 'combine', 'empty', 'split', 'zip', 'compose']);

/** The combinator a call invokes, by name. */
function combinatorOf(node: ts.Node): string | undefined {
  if (!ts.isCallExpression(node)) return;
  const callee = node.expression;
  const name = ts.isIdentifier(callee) ? callee.text : ts.isPropertyAccessExpression(callee) ? callee.name.text : undefined;
  return name && COMBINATORS.has(name) ? name : undefined;
}

/** Strip parentheses and one `await`: the value an argument position consumes. */
function operand(node: ts.Expression): { call: ts.Expression; awaited: boolean } {
  let inner = node;
  while (ts.isParenthesizedExpression(inner)) inner = inner.expression;
  if (ts.isAwaitExpression(inner)) {
    let call: ts.Expression = inner.expression;
    while (ts.isParenthesizedExpression(call)) call = call.expression;
    return { call, awaited: true };
  }
  return { call: inner, awaited: false };
}

/** An expression whose evaluation has no effect and does not depend on when it runs. */
function pure(node: ts.Expression): boolean {
  if (ts.isParenthesizedExpression(node)) return pure(node.expression);
  if (ts.isIdentifier(node) || ts.isLiteralExpression(node) || ts.isNoSubstitutionTemplateLiteral(node) ||
    node.kind === ts.SyntaxKind.TrueKeyword || node.kind === ts.SyntaxKind.FalseKeyword || node.kind === ts.SyntaxKind.NullKeyword ||
    ts.isArrowFunction(node) || ts.isFunctionExpression(node)) return true;
  if (ts.isPropertyAccessExpression(node)) return pure(node.expression);
  if (ts.isArrayLiteralExpression(node)) return node.elements.every(element => ts.isExpression(element) && pure(element));
  // Composition makes a function value; it calls no model.
  if (ts.isCallExpression(node) && combinatorOf(node) === 'compose') return node.arguments.every(pure);
  return false;
}

/** `x => x`, `(x) => x`, `function (x) { return x; }`. */
function isIdentity(node: ts.Expression): boolean {
  while (ts.isParenthesizedExpression(node)) node = node.expression;
  const fn = ts.isArrowFunction(node) || ts.isFunctionExpression(node) ? node : undefined;
  if (!fn || fn.parameters.length !== 1 || !ts.isIdentifier(fn.parameters[0]!.name)) return false;
  const name = fn.parameters[0]!.name.text;
  const body = ts.isBlock(fn.body) ? fn.body.statements.length === 1 && ts.isReturnStatement(fn.body.statements[0]!) ?
    fn.body.statements[0]!.expression : undefined : fn.body;
  let value = body;
  while (value && ts.isParenthesizedExpression(value)) value = value.expression;
  return !!value && ts.isIdentifier(value) && value.text === name;
}

/** Whether evaluating `node` may have an effect: a call (other than composition), an await, an assignment, `new`. */
function hasEffect(node: ts.Node): boolean {
  let found = false;
  const visit = (item: ts.Node): void => {
    if (found || ts.isArrowFunction(item) || ts.isFunctionExpression(item) || ts.isFunctionDeclaration(item)) return;
    if ((ts.isCallExpression(item) && combinatorOf(item) !== 'compose') || ts.isAwaitExpression(item) || ts.isNewExpression(item) ||
      ts.isDeleteExpression(item) || (ts.isBinaryExpression(item) && item.operatorToken.kind >= ts.SyntaxKind.FirstAssignment &&
        item.operatorToken.kind <= ts.SyntaxKind.LastAssignment) ||
      ((ts.isPrefixUnaryExpression(item) || ts.isPostfixUnaryExpression(item)) &&
        (item.operator === ts.SyntaxKind.PlusPlusToken || item.operator === ts.SyntaxKind.MinusMinusToken))) { found = true; return; }
    ts.forEachChild(item, visit);
  };
  visit(node);
  return found;
}

/**
 * Apply the enabled rules to `source`. Intermediate values bound by `const a = map(v, g)` and used once, in
 * `map(a, f)` later in the same block with nothing effectful in between, are fused as if written nested.
 */
export function rewriteCombinators(source: string, options: { gate: RewriteGate; fileName?: string; disabled?: boolean }): RewriteResult {
  const fileName = options.fileName ?? 'program.ts';
  const file = ts.createSourceFile(fileName, source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  const applied: RewriteRecord[] = [];
  const skipped: RewriteSkip[] = [];
  const siteOf = (node: ts.Node): string => {
    const original = ts.getOriginalNode(node);
    const start = original.pos >= 0 && original.getSourceFile() ? original.getStart(file) : 0;
    const { line, character } = file.getLineAndCharacterOfPosition(start);
    return `${fileName}:${line + 1}:${character + 1}`;
  };
  const spanOf = (node: ts.Node): string => {
    const original = ts.getOriginalNode(node);
    return original.pos >= 0 ? `${fileName}#${original.getStart(file)}-${original.getEnd()}` : `${fileName}#synthesized`;
  };
  const marked = (node: ts.Node): boolean => {
    for (let current: ts.Node | undefined = ts.getOriginalNode(node); current && current.pos >= 0; current = current.parent) {
      const ranges = ts.getLeadingCommentRanges(source, current.getFullStart()) ?? [];
      if (ranges.some(range => source.slice(range.pos, range.end).includes(NO_REWRITE_MARKER))) return true;
      if (ts.isStatement(current) && !ts.isBlock(current)) break;
    }
    return false;
  };
  /** Whether `rule` may apply at `node`; records why not. */
  const allowed = (rule: RewriteRule, node: ts.Node): RuleMeasurement | undefined => {
    const site = siteOf(node);
    if (options.disabled) { skipped.push({ rule, site, reason: 'disabled-program' }); return; }
    if (marked(node)) { skipped.push({ rule, site, reason: 'disabled-site' }); return; }
    const measurement = options.gate.enabled(rule);
    if (!measurement) { skipped.push({ rule, site, reason: 'not-enabled' }); return; }
    return measurement;
  };
  const record = (rule: RewriteRule, node: ts.Node, measurement: RuleMeasurement, replaced: ts.Node[]) =>
    applied.push({ rule, site: siteOf(node), enabled_by: measurement.comparison, replaced: replaced.map(spanOf) });

  const transformer: ts.TransformerFactory<ts.SourceFile> = context => {
    const f = context.factory;
    const callNamed = (like: ts.CallExpression, name: string, args: ts.Expression[]) => {
      const callee = like.expression;
      const target = ts.isPropertyAccessExpression(callee) ? f.updatePropertyAccessExpression(callee, callee.expression, f.createIdentifier(name)) :
        f.createIdentifier(name);
      return f.createCallExpression(target, undefined, args);
    };

    /** Fold `const a = map(v, g)` into its single later use as `map(a, f)`, when nothing effectful runs between. */
    const foldIntermediates = (statements: ts.NodeArray<ts.Statement>): ts.Statement[] => {
      const list = [...statements];
      for (let index = 0; index < list.length; index++) {
        const statement = list[index]!;
        if (!ts.isVariableStatement(statement) || !(statement.declarationList.flags & ts.NodeFlags.Const)) continue;
        const declarations = statement.declarationList.declarations;
        if (declarations.length !== 1 || !ts.isIdentifier(declarations[0]!.name) || !declarations[0]!.initializer) continue;
        const name = declarations[0]!.name.text, initializer = declarations[0]!.initializer;
        const inner = operand(initializer);
        if (combinatorOf(inner.call) !== 'map') continue;
        const uses: { statement: number; node: ts.Identifier }[] = [];
        list.forEach((later, position) => {
          if (position <= index) return;
          const visit = (item: ts.Node): void => {
            if (ts.isIdentifier(item) && item.text === name && !(ts.isPropertyAccessExpression(item.parent) && item.parent.name === item))
              uses.push({ statement: position, node: item });
            ts.forEachChild(item, visit);
          };
          visit(later);
        });
        if (uses.length !== 1) continue;
        const use = uses[0]!;
        // The use must be the value argument of a map call.
        let argument: ts.Node = use.node;
        while (ts.isParenthesizedExpression(argument.parent) || ts.isAwaitExpression(argument.parent)) argument = argument.parent;
        const outer = argument.parent;
        if (!ts.isCallExpression(outer) || combinatorOf(outer) !== 'map' || outer.arguments[0] !== argument) continue;
        const call = inner.call as ts.CallExpression;
        const between = list.slice(index + 1, use.statement);
        const blocked = between.some(hasEffect) || !call.arguments.every(pure) || !outer.arguments.slice(1).every(pure);
        if (blocked) { skipped.push({ rule: 'map-fusion', site: siteOf(outer), reason: 'effect' }); continue; }
        if (!allowed('map-fusion', outer)) continue;
        // Inline the intermediate into its use; the expression pass then fuses the nested form.
        const target = use.node;
        const replaceUse: ts.TransformerFactory<ts.Node> = inlineContext => node => {
          const visit = (item: ts.Node): ts.Node => item === target ? f.createParenthesizedExpression(initializer) :
            ts.visitEachChild(item, visit, inlineContext);
          return visit(node);
        };
        list[use.statement] = ts.transform(list[use.statement]!, [replaceUse]).transformed[0] as ts.Statement;
        list.splice(index, 1);
        index--;
      }
      return list;
    };

    const visit = (node: ts.Node): ts.Node => {
      if (ts.isBlock(node)) node = f.updateBlock(node, foldIntermediates(node.statements));
      else if (ts.isSourceFile(node)) node = f.updateSourceFile(node, foldIntermediates(node.statements));
      node = ts.visitEachChild(node, visit, context);
      if (!ts.isCallExpression(node)) return node;
      const kind = combinatorOf(node);
      const args = node.arguments;
      if (kind === 'map' && args.length === 2) {
        // map-identity: map(v, x => x) → v
        if (isIdentity(args[1]!)) {
          const measurement = allowed('map-identity', node);
          if (measurement) { record('map-identity', node, measurement, [node]); return args[0]!; }
          return node;
        }
        // map-fusion: map(map(v, g), f) → map(v, compose(f, g))
        const inner = operand(args[0]!);
        if (combinatorOf(inner.call) === 'map' && (inner.call as ts.CallExpression).arguments.length === 2) {
          const [v, g] = (inner.call as ts.CallExpression).arguments as unknown as [ts.Expression, ts.Expression];
          if (!pure(args[1]!)) { skipped.push({ rule: 'map-fusion', site: siteOf(node), reason: 'effect' }); return node; }
          const measurement = allowed('map-fusion', node);
          if (!measurement) return node;
          record('map-fusion', node, measurement, [inner.call, node]);
          return callNamed(node, 'map', [v, callNamed(node, 'compose', [args[1]!, g])]);
        }
      }
      if (kind === 'read' && args.length === 1) {
        // read-map: read(map(v, f)) → f(read(v)) (forward), when the comparison found the exact call cheaper.
        const inner = operand(args[0]!);
        if (combinatorOf(inner.call) === 'map' && (inner.call as ts.CallExpression).arguments.length === 2) {
          const [v, fn] = (inner.call as ts.CallExpression).arguments as unknown as [ts.Expression, ts.Expression];
          if (!pure(fn) || !pure(v)) { skipped.push({ rule: 'read-map', site: siteOf(node), reason: 'effect' }); return node; }
          const measurement = allowed('read-map', node);
          if (!measurement || measurement.direction === 'reverse') return node;
          record('read-map', node, measurement, [inner.call, node]);
          return f.createCallExpression(f.createParenthesizedExpression(fn), undefined, [f.createAwaitExpression(callNamed(node, 'read', [v]))]);
        }
      }
      if (kind === 'combine' && args.length === 2) {
        // combine-identity: combine(v, empty()) → v
        const right = operand(args[1]!);
        if (combinatorOf(right.call) === 'empty') {
          const measurement = allowed('combine-identity', node);
          if (measurement) { record('combine-identity', node, measurement, [right.call, node]); return args[0]!; }
          return node;
        }
        // combine-reassociate: a left-leaning chain of three or more leaves → a balanced tree.
        const leaves: ts.Expression[] = [];
        let chain: ts.Expression = node;
        const replaced: ts.Node[] = [];
        while (true) {
          const current = operand(chain);
          if (combinatorOf(current.call) !== 'combine' || (current.call as ts.CallExpression).arguments.length !== 2) break;
          const call = current.call as ts.CallExpression;
          replaced.push(call);
          leaves.unshift(call.arguments[1]!);
          chain = call.arguments[0]!;
        }
        leaves.unshift(chain);
        if (leaves.length >= 4 && leaves.every(pure)) {
          const measurement = allowed('combine-reassociate', node);
          if (!measurement) return node;
          record('combine-reassociate', node, measurement, replaced);
          const balanced = (items: ts.Expression[]): ts.Expression => items.length === 1 ? items[0]! :
            f.createAwaitExpression(callNamed(node, 'combine', [balanced(items.slice(0, Math.ceil(items.length / 2))),
              balanced(items.slice(Math.ceil(items.length / 2)))]));
          const tree = balanced(leaves);
          return ts.isAwaitExpression(tree) ? tree.expression : tree;
        }
      }
      if (kind === 'split' && args.length === 1) {
        // split-zip: split(zip(a, b)) → { 0: a, 1: b }
        const inner = operand(args[0]!);
        if (combinatorOf(inner.call) === 'zip' && (inner.call as ts.CallExpression).arguments.length === 2) {
          const [a, b] = (inner.call as ts.CallExpression).arguments as unknown as [ts.Expression, ts.Expression];
          const measurement = allowed('split-zip', node);
          if (!measurement) return node;
          record('split-zip', node, measurement, [inner.call, node]);
          return f.createObjectLiteralExpression([f.createPropertyAssignment(f.createNumericLiteral(0), a),
            f.createPropertyAssignment(f.createNumericLiteral(1), b)]);
        }
      }
      return node;
    };
    return root => visit(root) as ts.SourceFile;
  };
  const result = ts.transform(file, [transformer]);
  const transformed = result.transformed[0]!;
  const code = applied.length ? ts.createPrinter({ newLine: ts.NewLineKind.LineFeed }).printFile(transformed) : source;
  result.dispose();
  return { code, applied, skipped };
}

/** Record applied rewrites as `rewrite` nodes of an execution graph. */
export function recordRewrites(trace: NativeTraceRecorder | undefined, records: readonly RewriteRecord[], inputs: readonly GraphInput[] = []): string[] {
  return records.flatMap(item => {
    const node = graphNode(trace, 'rewrite', { rule: item.rule, site: item.site, replaced: item.replaced, enabled_by: item.enabled_by }, inputs);
    return node ? [node] : [];
  });
}
