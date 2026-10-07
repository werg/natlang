import ts from 'typescript';
import { hexDigest } from '../native/hash.js';
import { solveHoles } from './holes.js';
import { awaitedType, describeTarget, isPromiseLike, TargetError, type TargetDescriptor } from './targets.js';
import { checkNeuralese, neuraleseParts, type NeuraleseLiteral } from './neuralese.js';

export type SourceSpan = { file: string; start: number; end: number; line: number; column: number };

export type NatlangDiagnostic = SourceSpan & {
  code: 'nl-unknown-return' | 'nl-unknown-parameter' | 'nl-not-called' | 'nl-not-tag' | 'nl-shadowed' | 'nl-ambiguous-signature' | 'nl-sync-callback' |
    'nl-parameter-collision' | 'nl-capture-parameter-collision' | 'nl-unknown-name' | 'nl-spread' | 'nl-const-capture-write' |
    'forbidden-loop' | 'forbidden-dynamic-code' | 'recursion' | 'callable-scope' | 'reserved-property' |
    'duplicate-site' | 'iterate-step' | 'iterate-predicate' | 'module-collision' | 'typescript' |
    'neuralese-opaque-access' | 'neuralese-condition' | 'neuralese-interpolation' | 'neuralese-untyped-literal' |
    'neuralese-nested' | 'type-recursive-function' | 'neuralese-file' | 'nl-explicit-captures' | 'nl-type-arguments';
  message: string;
  severity: 'error' | 'warning';
};

export type CapturePlan = {
  name: string;
  type: TargetDescriptor;
  mutable: boolean;
  source: 'input' | 'local' | 'block' | 'handle';
  /** Offset of the first mention in the template, relative to the source file. */
  mentionSpan: number;
  /**
   * Explicit captures (`nl.with`): `snapshot` takes the value when the function is created; `live` reads the let
   * binding at each call and writes it back. Implicit captures have no mode: data lets are live, functions by value.
   */
  mode?: 'snapshot' | 'live';
};

export type InlineLambdaPlan = {
  programId?: string;
  sourceSpan: SourceSpan;
  /** Present only for compiler-registered authored source. */
  adaptation?: { label?: string; templateStart: number; templateEnd: number; expressions: string[]; visibleBindings: string[]; slotBindings: string[] };
  /** Source revision + AST span; stable for one source revision, not a user-facing name. */
  definitionId: string;
  /** Cooked template strings; interpolated values are inserted between them at invocation. */
  strings: string[];
  /** Exact template extent and ordered interpolation provenance; never executed during conversion. */
  templateSpan: SourceSpan;
  interpolations: { expression: string; sourceSpan: SourceSpan; type: TargetDescriptor | null; unsupported?: string }[];
  /** Instruction text with interpolations shown as `${…}` for listings and traces. */
  instructions: string;
  parameters: { name: string; type: TargetDescriptor }[];
  /**
   * Saved without parameters and never called where it was created (`const judge = nl<R>`...``, called in a later
   * eval): each call passes whatever arguments it has, typed from their values.
   */
  openParameters?: boolean;
  returns: TargetDescriptor;
  captures: CapturePlan[];
  /** `nl.with({ … })`: the captures are exactly the listed ones; no names are captured by mention. */
  explicitCaptures?: true;
  /** A soft function literal: the template is one model-written Neuralese block, this block ID. */
  softBody?: string;
  inheritedCodebaseRevision: string;
};

export type BindingClassifier = (declaration: ts.Declaration) => CapturePlan['source'] | undefined;

export type InlineAnalysisOptions = {
  /** Files whose declarations may be captured in addition to the analyzed file itself (eval scope declarations). */
  scopeFiles?: readonly ts.SourceFile[];
  authored?: boolean;
  /** Names never captured (the result slot, debug state, plumbing). */
  excludedNames?: ReadonlySet<string>;
  /** Reject type aliases that reach themselves through a function type (model-written eval code). */
  recursiveTypes?: boolean;
  classify?: BindingClassifier;
  sourceRevision?: string;
  codebaseRevision?: string;
  /** Display path used in spans and definition IDs. */
  displayPath?: (file: ts.SourceFile) => string;
};

const IDENTIFIER_TOKEN = /(?<![A-Za-z0-9_$])[A-Za-z_$][A-Za-z0-9_$]*(?![A-Za-z0-9_$])/g;
const BACKTICK_MENTION = /\\`([A-Za-z_$][A-Za-z0-9_$]*)\\`/g;
const IDENTIFIER_NAME = /^[A-Za-z_$][A-Za-z0-9_$]*$/;

/** The block ID when a template is exactly one model-written Neuralese body, `${__neuralese.body("nz1_…")}`. */
export function softBodyOf(template: ts.TemplateLiteral): string | undefined {
  if (!ts.isTemplateExpression(template) || template.templateSpans.length !== 1 || template.head.text !== '' ||
      template.templateSpans[0]!.literal.text !== '') return;
  const expression = template.templateSpans[0]!.expression;
  if (!ts.isCallExpression(expression) || !ts.isPropertyAccessExpression(expression.expression)) return;
  const callee = expression.expression;
  const [argument] = expression.arguments;
  return ts.isIdentifier(callee.expression) && callee.expression.text === '__neuralese' && callee.name.text === 'body' &&
    expression.arguments.length === 1 && argument && ts.isStringLiteralLike(argument) ? argument.text : undefined;
}

/** The `nl.with(…)` call when a tagged template's tag is one. */
export function withCallOf(checker: ts.TypeChecker | undefined, tag: ts.Expression): ts.CallExpression | undefined {
  if (!ts.isCallExpression(tag) || !ts.isPropertyAccessExpression(tag.expression) || tag.expression.name.text !== 'with') return;
  if (checker) return resolveIntrinsic(checker, tag.expression) === 'nl.with' ? tag : undefined;
  return ts.isIdentifier(tag.expression.expression) && tag.expression.expression.text === 'nl' ? tag : undefined;
}

/** The binding a `live(x)` capture names, when `expression` is one. */
export function liveTargetOf(checker: ts.TypeChecker | undefined, expression: ts.Expression): ts.Expression | undefined {
  while (ts.isParenthesizedExpression(expression)) expression = expression.expression;
  if (!ts.isCallExpression(expression)) return;
  const isLive = checker ? resolveIntrinsic(checker, expression.expression) === 'live' :
    ts.isIdentifier(expression.expression) && expression.expression.text === 'live';
  return isLive ? expression.arguments[0] ?? expression : undefined;
}

/** The intrinsic name (`nl`, `iterateOn`) a declaration stands for, via its `@natlangIntrinsic` tag. */
export function intrinsicOf(declaration: ts.Node | undefined): string | undefined {
  if (!declaration) return;
  for (const tag of ts.getJSDocTags(declaration)) {
    if (tag.tagName.text !== 'natlangIntrinsic') continue;
    const comment = typeof tag.comment === 'string' ? tag.comment : ts.getTextOfJSDocComment(tag.comment);
    return comment?.trim().split(/\s+/)[0];
  }
  return;
}

/** Resolve an expression to the intrinsic it references, following import aliases. */
export function resolveIntrinsic(checker: ts.TypeChecker, expression: ts.Expression): string | undefined {
  let symbol = checker.getSymbolAtLocation(ts.isPropertyAccessExpression(expression) ? expression.name : expression);
  if (symbol && symbol.flags & ts.SymbolFlags.Alias) symbol = checker.getAliasedSymbol(symbol);
  for (const declaration of symbol?.declarations ?? []) {
    const name = intrinsicOf(declaration);
    if (name) return name;
  }
  return;
}

export function spanOf(node: ts.Node, displayPath: (file: ts.SourceFile) => string = file => file.fileName,
  start = node.getStart(), end = node.getEnd()): SourceSpan {
  const file = node.getSourceFile();
  const position = file.getLineAndCharacterOfPosition(start);
  return { file: displayPath(file), start, end, line: position.line + 1, column: position.character + 1 };
}

const unwrapParentheses = (node: ts.Node): ts.Node => {
  while (node.parent && (ts.isParenthesizedExpression(node.parent) || ts.isAsExpression(node.parent) ||
    ts.isNonNullExpression(node.parent) || ts.isSatisfiesExpression(node.parent))) node = node.parent;
  return node;
};

type Signature = { parameters?: { name: string; type: ts.Type }[]; returns?: ts.Type; origin: string; open?: boolean };

export function analyzeInlineLambdas(program: ts.Program, files: readonly ts.SourceFile[],
  options: InlineAnalysisOptions = {}): { plans: InlineLambdaPlan[]; diagnostics: NatlangDiagnostic[]; neuralese: NeuraleseLiteral[] } {
  const checker = program.getTypeChecker();
  const plans: InlineLambdaPlan[] = [];
  const labels = new Set<string>();
  const diagnostics: NatlangDiagnostic[] = [];
  const displayPath = options.displayPath ?? (file => file.fileName);
  const excluded = new Set(['result', 'nl', 'iterateOn', 'self', 'live', '__neuralese', ...(options.excludedNames ?? [])]);
  const scopeFiles = new Set(options.scopeFiles ?? []);

  const report = (node: ts.Node, code: NatlangDiagnostic['code'], message: string, severity: 'error' | 'warning' = 'error') =>
    diagnostics.push({ ...spanOf(node, displayPath), code, message, severity });

  const target = (type: ts.Type, node: ts.Node, what: string, allowHost = true,
    within: { program: ts.Program; location: ts.Node } = { program, location: node }): TargetDescriptor | undefined => {
    // A result or parameter typed any or unknown is open, as when nothing says what it is; so is any such part of one.
    if (type.flags & (ts.TypeFlags.Any | ts.TypeFlags.Unknown)) return { text: 'any', natlang: 'unknown', aliases: {} };
    try { return describeTarget(within.program, within.program.getTypeChecker(), type, { allowHost, location: within.location }); }
    catch (error) {
      if (!(error instanceof TargetError)) throw error;
      report(node, what === 'return' ? 'nl-unknown-return' : 'nl-unknown-parameter', what === 'return' ?
        `Return type of this \`nl\` expression is unknown (${error.message}); annotate the target or write \`nl<Verdict>\`.` :
        `Type of ${what} for this \`nl\` expression cannot be used (${error.message}); annotate it.`);
      return;
    }
  };

  const widen = (type: ts.Type): ts.Type => type.flags & ts.TypeFlags.Literal && !(type.flags & ts.TypeFlags.EnumLiteral) ?
    checker.getBaseTypeOfLiteralType(type) : type;

  /** Contextual value type for an expression whose result is awaited or returned as a promise. */
  const resultContext = (outer: ts.Node, awaited: boolean): ts.Type | undefined => {
    // `return await nl`...`(x)` in an async function: the function's awaited return type, not `T | PromiseLike<T>`.
    if (outer.parent && ts.isReturnStatement(outer.parent)) {
      let fn: ts.Node | undefined = outer.parent;
      while (fn && !ts.isFunctionLike(fn)) fn = fn.parent;
      if (fn && ts.isFunctionLike(fn) && ts.getCombinedModifierFlags(fn as ts.Declaration) & ts.ModifierFlags.Async) {
        // An unannotated async callback can still have a declared caller slot,
        // e.g. review_each(..., async item => { return await nl`...`(item); }).
        // Its Promise<boolean> contract must reach the inline child as boolean.
        const contextual = !fn.type && (ts.isArrowFunction(fn) || ts.isFunctionExpression(fn)) ?
          checker.getContextualType(fn)?.getCallSignatures() : undefined;
        const signature = fn.type ? checker.getSignatureFromDeclaration(fn) :
          contextual?.length === 1 ? contextual[0] : undefined;
        const declared = signature && awaitedType(checker, checker.getReturnTypeOfSignature(signature)).type;
        if (declared && !(declared.flags & ts.TypeFlags.Any)) return declared;
      }
    }
    const contextual = checker.getContextualType(outer as ts.Expression);
    if (!contextual || contextual.flags & ts.TypeFlags.Any) return;
    const result = awaited ? contextual : awaitedType(checker, contextual).type;
    // A generic slot still being inferred (`map`'s `U`) says nothing yet; uses decide (step 6).
    const open = (type: ts.Type): boolean => !!(type.flags & (ts.TypeFlags.TypeParameter | ts.TypeFlags.Unknown | ts.TypeFlags.Any)) ||
      (type.isUnion() && type.types.some(open));
    return open(result) ? undefined : result;
  };

  /** Later uses of an unannotated local that holds the result or the callable itself. */
  const laterUses = (declaration: ts.VariableDeclaration): ts.Identifier[] => {
    if (!ts.isIdentifier(declaration.name)) return [];
    const symbol = checker.getSymbolAtLocation(declaration.name);
    const found: ts.Identifier[] = [];
    const visit = (node: ts.Node): void => {
      if (ts.isIdentifier(node) && node !== declaration.name && node.text === declaration.name.getText() &&
          checker.getSymbolAtLocation(node) === symbol) found.push(node);
      ts.forEachChild(node, visit);
    };
    visit(declaration.getSourceFile());
    return found;
  };

  const uniqueType = (types: ts.Type[], node: ts.Node, what: string): ts.Type | undefined => {
    const distinct: ts.Type[] = [];
    for (const type of types) if (!distinct.some(seen => checker.isTypeAssignableTo(seen, type) && checker.isTypeAssignableTo(type, seen)))
      distinct.push(type);
    if (distinct.length > 1) {
      report(node, 'nl-ambiguous-signature', `Uses of this \`nl\` expression disagree about its ${what}; annotate it.`);
      return;
    }
    return distinct[0];
  };

  const analyze = (node: ts.TaggedTemplateExpression, withCall?: ts.CallExpression): void => {
    const outerTag = unwrapParentheses(node);
    const call = outerTag.parent && ts.isCallExpression(outerTag.parent) && outerTag.parent.expression === outerTag ?
      outerTag.parent : undefined;
    const signature: Signature = { origin: 'none' };
    // `await nl`...`` waits on the function itself, which is never a judgment. (Eval inserts the call instead;
    // see scope-compiler.ts. Project source is not rewritten, so it gets this diagnostic.)
    // items.map(r => nl`...`): the arrow hands back the function itself, one per item, and nothing ever runs it.
    const arrow = outerTag.parent;
    if (arrow && ts.isArrowFunction(arrow) && arrow.body === outerTag && arrow.parent && ts.isCallExpression(arrow.parent) &&
        arrow.parent.arguments.includes(arrow)) {
      report(node, 'nl-not-called', 'This arrow returns the `nl` function itself, never called: nl`...` creates a function. ' +
        'Call it, and await the results: `await Promise.all(items.map(item => nl`Does item …?`(item)))`, or ask one ' +
        'question at a time with `await nl(`… ${item.text}`)`.');
      return;
    }
    if (outerTag.parent && ts.isAwaitExpression(outerTag.parent)) {
      report(node, 'nl-not-called', 'This awaits the `nl` function itself instead of calling it: nl`...` creates a function. ' +
        'Call it with the values it should judge, as in `await nl`...`(value)`; names its instructions mention are also visible to it.');
      return;
    }

    // A capture object is a value argument, never a separate type argument.
    // Do not silently lower an invalid two-generic tag to its first schema.
    if ((node.typeArguments?.length ?? 0) > 1 || (withCall?.typeArguments?.length ?? 0) > 1) {
      report(node, 'nl-type-arguments', '`nl` and `nl.with` accept one type argument: the child result type or full callable signature. ' +
        'Put fixed context in `nl.with<Result>({ context })`, then call the function with its current input. ' +
        'Do not use `nl.with<CaptureObject, Result>` or pass the capture object in place of the child input.');
      return;
    }

    // 1. Explicit annotation (`nl<F>`, or `nl.with<F>({ … })`).
    const annotation = node.typeArguments?.[0] ?? withCall?.typeArguments?.[0];
    if (annotation) {
      if (ts.isFunctionTypeNode(annotation)) {
        signature.parameters = [];
        for (const parameter of annotation.parameters) {
          if (!ts.isIdentifier(parameter.name) || parameter.dotDotDotToken) {
            report(parameter, 'nl-spread', 'Annotated `nl` parameters must be named identifiers without rest syntax.');
            return;
          }
          signature.parameters.push({ name: parameter.name.text,
            type: parameter.type ? checker.getTypeFromTypeNode(parameter.type) : checker.getAnyType() });
        }
        signature.returns = awaitedType(checker, checker.getTypeFromTypeNode(annotation.type)).type;
      } else signature.returns = awaitedType(checker, checker.getTypeFromTypeNode(annotation)).type;
      signature.origin = 'annotation';
    }

    // 2a. The step of an iteration, `nl`...`.iterateOn(initial, ...args)` or `iterateOn(nl`...`, initial, ...args)`:
    // the state has the initial value's type, and the step returns the next state.
    const iteration = signature.origin === 'none' ? iterationArguments(checker, outerTag) : undefined;
    if (iteration) {
      const [initial, ...rest] = iteration;
      if (!initial || rest.some(ts.isSpreadElement) || ts.isSpreadElement(initial)) {
        report(node, 'nl-spread', 'An `nl` iteration step needs an initial state and plainly listed arguments, or an explicit signature.');
        return;
      }
      const name = (argument: ts.Expression, index: number) => ts.isIdentifier(argument) ? argument.text : index ? `input${index + 1}` : 'state';
      signature.parameters = [initial, ...rest].map((argument, index) => ({ name: name(argument, index), type: widen(checker.getTypeAtLocation(argument)) }));
      signature.returns = signature.parameters[0]!.type;
      signature.origin = 'iteration';
    }

    // 2. Contextual callable type (callbacks, annotated locals).
    if (!call && !signature.parameters) {
      const contextual = checker.getContextualType(outerTag as ts.Expression);
      // A Neuralese<F> slot (a soft function literal) takes F's own signature, not the branded value's rest call.
      const slots = contextual ? (contextual.isUnion() ? contextual.types : [contextual]) : [];
      const softSlot = slots.some(type => neuraleseParts(checker, type));
      const signatures = slots.flatMap(type => (neuraleseParts(checker, type)?.element ?? type).getCallSignatures());
      if (signatures.length > 1) {
        report(node, 'nl-ambiguous-signature', 'The callback slot for this `nl` expression has several call signatures; annotate it with `nl<(x: T) => R>`.');
        return;
      }
      const only = signatures[0];
      if (only) {
        const returnType = only.getReturnType();
        const members = returnType.isUnion() ? returnType.types : [returnType];
        // A Neuralese<F> is always called asynchronously, whatever F's declared result.
        if (!softSlot && !(returnType.flags & ts.TypeFlags.Any) && !members.some(member => isPromiseLike(checker, member))) {
          report(node, 'nl-sync-callback', 'This callback slot expects a synchronous result, but an `nl` function returns a Promise. ' +
            'Use an async map and then an ordinary filter or loop.');
          return;
        }
        signature.parameters = only.getParameters().map(parameter => {
          const declaration = parameter.valueDeclaration;
          if (declaration && ts.isParameter(declaration) && declaration.dotDotDotToken)
            report(node, 'nl-spread', 'A rest-parameter callback slot needs an explicit `nl<(...) => R>` signature.');
          return { name: parameter.name, type: checker.getTypeOfSymbolAtLocation(parameter, node) };
        });
        signature.returns ??= awaitedType(checker, returnType).type;
        if (signature.origin === 'none') signature.origin = 'context';
      }
    }

    // 3. Immediate-call arguments and 4. result context.
    if (call) {
      const names: string[] = [];
      const types: ts.Type[] = [];
      let generated = 0;
      for (const argument of call.arguments) {
        if (ts.isSpreadElement(argument)) {
          const spread = checker.getTypeAtLocation(argument.expression);
          if (!checker.isTupleType(spread)) {
            report(argument, 'nl-spread', 'A spread argument to an `nl` call needs a statically known tuple or an explicit signature.');
            return;
          }
          for (const element of checker.getTypeArguments(spread as ts.TypeReference)) {
            names.push(generated++ ? `input${generated}` : 'input'); types.push(widen(element));
          }
          continue;
        }
        names.push(ts.isIdentifier(argument) ? argument.text : generated++ ? `input${generated}` : 'input');
        types.push(widen(checker.getTypeAtLocation(argument)));
      }
      if (!signature.parameters) signature.parameters = names.map((name, index) => ({ name, type: types[index]! }));
      else if (signature.parameters.length !== names.length) {
        report(call, 'nl-ambiguous-signature', `This \`nl\` call passes ${names.length} arguments but its signature has ${signature.parameters.length}.`);
        return;
      }
      if (!signature.returns) {
        const outer = unwrapParentheses(call);
        const awaitNode = outer.parent && ts.isAwaitExpression(outer.parent) ? unwrapParentheses(outer.parent) : undefined;
        signature.returns = resultContext(awaitNode ?? outer, !!awaitNode);
      }
    } else if (!signature.parameters) {
      // 5. `const judge = nl`...`` followed by calls: its parameters come from the calls' arguments.
      const holder = outerTag.parent;
      if (holder && ts.isVariableDeclaration(holder) && !holder.type) {
        const calls = laterUses(holder).map(use => unwrapParentheses(use)).map(use => use.parent)
          .filter((parent): parent is ts.CallExpression => !!parent && ts.isCallExpression(parent));
        if (!signature.parameters && calls.length) {
          const arity = new Set(calls.map(item => item.arguments.length));
          if (arity.size > 1) { report(node, 'nl-ambiguous-signature', 'Calls of this `nl` function pass different numbers of arguments; annotate it.'); return; }
          const first = calls[0]!;
          signature.parameters = first.arguments.map((argument, index) => ({
            name: ts.isIdentifier(argument) ? argument.text : index ? `input${index + 1}` : 'input',
            type: uniqueType(calls.map(item => widen(checker.getTypeAtLocation(item.arguments[index]!))), node,
              `parameter ${index + 1} type`) ?? checker.getAnyType() }));
        }
        if (!signature.parameters && !calls.length) signature.open = true;
      }
    }

    // A soft function literal says nothing about its signature but its type: the context must give it.
    if (softBodyOf(node.template) && (!signature.parameters || !signature.returns)) {
      report(node, 'neuralese-untyped-literal', 'Nothing says what this soft function literal is: give it a function type first, ' +
        'as in `const triage: Neuralese<(t: Ticket) => Promise<Label>> = nl.with({ … })`…``, or write `nl.with<F>({ … })`.');
      return;
    }
    // 6. No annotation or context: infer the result from how it is used (see holes.ts).
    if (!signature.returns) { deferred.push({ node, call, signature, withCall }); return; }
    finish(node, call, signature, undefined, undefined, withCall);
  };

  const deferred: { node: ts.TaggedTemplateExpression; call: ts.CallExpression | undefined; signature: Signature;
    withCall?: ts.CallExpression }[] = [];

  /** The listed captures of `nl.with({ … })`: `name` or `name: value` snapshots, `name: live(name)` live lets. */
  const explicitCaptures = (withCall: ts.CallExpression, node: ts.TaggedTemplateExpression): CapturePlan[] | undefined => {
    const [argument] = withCall.arguments;
    if (withCall.arguments.length !== 1 || !argument || !ts.isObjectLiteralExpression(argument)) {
      report(withCall, 'nl-explicit-captures', 'nl.with takes one object literal listing the captures, as in ' +
        '`nl.with({ rubric, count: live(count) })`.');
      return;
    }
    const captures: CapturePlan[] = [];
    for (const property of argument.properties) {
      let name: string, expression: ts.Expression;
      if (ts.isShorthandPropertyAssignment(property)) { name = property.name.text; expression = property.name; }
      else if (ts.isPropertyAssignment(property) && (ts.isIdentifier(property.name) || ts.isStringLiteral(property.name))) {
        name = property.name.text; expression = property.initializer;
      } else {
        report(property, 'nl-explicit-captures', 'List each capture as `name` or `name: value`; spreads, methods and computed names are not captures.');
        return;
      }
      if (!IDENTIFIER_NAME.test(name) || excluded.has(name) || name.startsWith('__natlang')) {
        report(property, 'nl-explicit-captures', `${JSON.stringify(name)} cannot be a capture name.`);
        return;
      }
      let mode: 'snapshot' | 'live' = 'snapshot', mutable = false;
      let declaration: ts.Declaration | undefined;
      const liveTarget = liveTargetOf(checker, expression);
      if (liveTarget) {
        const symbol = ts.isIdentifier(liveTarget) ? checker.getSymbolAtLocation(liveTarget) : undefined;
        declaration = symbol?.valueDeclaration ?? symbol?.declarations?.[0];
        if (!ts.isIdentifier(liveTarget) || liveTarget.text !== name) {
          report(property, 'nl-explicit-captures', `A live capture names its own binding: write \`${name}: live(${name})\`.`);
          return;
        }
        if (!declaration || !isMutableBinding(declaration)) {
          report(property, 'nl-const-capture-write', `live(${name}) needs a let binding; capture a const as a plain snapshot, \`${name}\`.`);
          return;
        }
        mode = 'live'; mutable = true; expression = liveTarget;
      } else if (ts.isIdentifier(expression)) {
        const symbol = checker.getSymbolAtLocation(expression);
        declaration = symbol?.valueDeclaration ?? symbol?.declarations?.[0];
      }
      const type = checker.getTypeAtLocation(expression);
      let described: TargetDescriptor;
      try { described = describeTarget(program, checker, type, { allowHost: true, location: node }); }
      catch (error) {
        if (!(error instanceof TargetError)) throw error;
        described = { text: checker.typeToString(type, node), aliases: {}, host: { kind: 'shape', members: [] } };
      }
      const source = (declaration && options.classify?.(declaration)) ?? (described.host ? 'handle' :
        declaration && ts.isParameter(declaration) ? 'input' : declaration && isTopLevel(declaration) ? 'local' : 'block');
      captures.push({ name, type: described, mutable, source, mentionSpan: property.getStart(), mode });
    }
    return captures;
  };

  const finish = (node: ts.TaggedTemplateExpression, call: ts.CallExpression | undefined, signature: Signature,
    solved?: { program: ts.Program; location: ts.Node }, openReturns?: TargetDescriptor, withCall?: ts.CallExpression): void => {
    const file = node.getSourceFile();
    const parameters = signature.parameters ?? [];
    const seen = new Set<string>();
    for (const parameter of parameters) {
      if (seen.has(parameter.name)) {
        report(call ?? node, 'nl-parameter-collision', `Two inputs of this \`nl\` expression are both named ${JSON.stringify(parameter.name)}; ` +
          'pass distinct variables or annotate the parameters.');
        return;
      }
      seen.add(parameter.name);
    }
    const returns = openReturns ?? target(signature.returns!, node, 'return', true, solved);
    if (!returns) return;
    const parameterTargets: InlineLambdaPlan['parameters'] = [];
    for (const parameter of parameters) {
      const described = target(parameter.type, node, `parameter ${parameter.name}`);
      if (!described) return;
      parameterTargets.push({ name: parameter.name, type: described });
    }

    // Captures: exact mentions of visible value bindings, plus identifiers inside interpolations.
    const literalParts: { text: string; offset: number }[] = [];
    const strings: string[] = [];
    const template = node.template;
    if (ts.isNoSubstitutionTemplateLiteral(template)) {
      literalParts.push({ text: template.rawText ?? template.text, offset: template.getStart() + 1 });
      strings.push(template.text);
    } else {
      literalParts.push({ text: template.head.rawText ?? template.head.text, offset: template.head.getStart() + 1 });
      strings.push(template.head.text);
      for (const span of template.templateSpans) {
        literalParts.push({ text: span.literal.rawText ?? span.literal.text, offset: span.literal.getStart() + 1 });
        strings.push(span.literal.text);
      }
    }
    // Record the checked AST, not a later regex parse of the rendered instruction.
    // An unrepresentable interpolation remains legal JS and is held by converters.
    const templateSpan = spanOf(template, displayPath);
    const interpolations: InlineLambdaPlan['interpolations'] = ts.isNoSubstitutionTemplateLiteral(template) ? [] :
      template.templateSpans.map(({ expression }) => {
        const provenance = { expression: expression.getText(file), sourceSpan: spanOf(expression, displayPath) };
        try { return { ...provenance, type: describeTarget(program, checker, checker.getTypeAtLocation(expression),
          { allowHost: true, location: expression }) }; }
        catch (error) {
          if (!(error instanceof TargetError)) throw error;
          return { ...provenance, type: null, unsupported: error.message };
        }
      });
    const explicit = new Set(parameters.map(parameter => parameter.name));
    const softBody = softBodyOf(template);
    if (withCall || softBody) {
      // Explicit captures: exactly the listed ones (none for a bare soft literal); nothing is captured by mention.
      const captures = withCall ? explicitCaptures(withCall, node) : [];
      if (!captures) return;
      const collision = captures.find(capture => explicit.has(capture.name));
      if (collision) {
        report(withCall ?? node, 'nl-capture-parameter-collision',
          `The \`nl.with\` capture ${JSON.stringify(collision.name)} conflicts with an input parameter of the same name. ` +
          'Rename the capture or the call argument/annotated parameter so captures and inputs have distinct names.');
        return;
      }
      const listed = new Set([...captures.map(capture => capture.name), ...explicit]);
      if (!softBody) for (const part of literalParts) for (const match of part.text.matchAll(BACKTICK_MENTION)) {
        const name = match[1]!;
        if (!listed.has(name)) diagnostics.push({ ...spanOf(node, displayPath, part.offset + match.index!, part.offset + match.index! + match[0].length),
          code: 'nl-unknown-name', severity: 'error',
          message: `\`${name}\` is quoted as a name in this instruction, but it is neither a parameter nor listed in nl.with({ … }).` });
      }
      const sourceSpan = spanOf(node, displayPath);
      const definitionId = `nl:${hexDigest(`${options.sourceRevision ?? ''}\0${sourceSpan.file}\0${sourceSpan.start}\0${sourceSpan.end}`).slice(0, 16)}`;
      // An authored site is adaptable by its instructions; its visible bindings are exactly the listed captures.
      const adaptation: InlineLambdaPlan['adaptation'] = options.authored ? { templateStart: template.getStart(),
        templateEnd: template.getEnd(), expressions: [], visibleBindings: [...listed].sort(), slotBindings: [] } : undefined;
      plans.push({ sourceSpan, templateSpan, interpolations, definitionId, ...(adaptation ? { adaptation } : {}), strings: softBody ? [''] : strings, instructions: softBody ? '' : strings.join('${…}'),
        parameters: parameterTargets, ...(signature.open ? { openParameters: true } : {}), returns, captures, explicitCaptures: true,
        ...(softBody ? { softBody } : {}), inheritedCodebaseRevision: options.codebaseRevision ?? '' });
      return;
    }
    const candidates = new Map<string, ts.Symbol>();
    for (const symbol of checker.getSymbolsInScope(node, ts.SymbolFlags.Value | ts.SymbolFlags.Alias)) {
      const name = symbol.name;
      if (excluded.has(name) || name.startsWith('__natlang') || explicit.has(name) || candidates.has(name)) continue;
      const declaration = symbol.valueDeclaration ?? symbol.declarations?.[0];
      if (!declaration) continue;
      const declarationFile = declaration.getSourceFile();
      if (declarationFile !== file && !scopeFiles.has(declarationFile)) continue;
      if (declarationFile === file && declaration.getEnd() > node.getStart() && !isEnclosingParameter(declaration, node)) continue;
      if (symbol.flags & ts.SymbolFlags.Alias) {
        const aliased = checker.getAliasedSymbol(symbol);
        if (!(aliased.flags & ts.SymbolFlags.Value)) continue;
      }
      candidates.set(name, symbol);
    }
    const mentions = new Map<string, number>();
    for (const part of literalParts) {
      for (const match of part.text.matchAll(IDENTIFIER_TOKEN)) {
        if (candidates.has(match[0]) && !mentions.has(match[0])) mentions.set(match[0], part.offset + match.index!);
      }
      for (const match of part.text.matchAll(BACKTICK_MENTION)) {
        const name = match[1]!;
        if (!candidates.has(name) && !explicit.has(name) && !excluded.has(name))
          diagnostics.push({ ...spanOf(node, displayPath, part.offset + match.index!, part.offset + match.index! + match[0].length),
            code: 'nl-unknown-name', severity: 'error',
            message: `\`${name}\` is quoted as a name in this instruction, but no binding with that name is visible here.` });
      }
    }
    const slotCaptures = new Set<string>();
    if (!ts.isNoSubstitutionTemplateLiteral(template)) for (const span of template.templateSpans) {
      const visit = (child: ts.Node): void => {
        if (ts.isIdentifier(child) && !(ts.isPropertyAccessExpression(child.parent) && child.parent.name === child)) {
          const symbol = checker.getSymbolAtLocation(child);
          if (symbol && candidates.get(child.text) === symbol) {
            slotCaptures.add(child.text);
            if (!mentions.has(child.text)) mentions.set(child.text, child.getStart());
          }
        }
        ts.forEachChild(child, visit);
      };
      visit(span.expression);
    }
    const captures: CapturePlan[] = [];
    for (const [name, offset] of [...mentions].sort((a, b) => a[1] - b[1])) {
      const symbol = candidates.get(name)!;
      const declaration = (symbol.valueDeclaration ?? symbol.declarations![0]!) as ts.Declaration;
      const type = checker.getTypeOfSymbolAtLocation(symbol, node);
      let described: TargetDescriptor;
      try { described = describeTarget(program, checker, type, { allowHost: true, location: node }); }
      catch (error) {
        if (!(error instanceof TargetError)) throw error;
        described = { text: checker.typeToString(type, node), aliases: {}, host: { kind: 'shape', members: [] } };
      }
      const mutable = isMutableBinding(declaration);
      const source = options.classify?.(declaration) ??
        (described.host ? 'handle' : ts.isParameter(declaration) ? 'input' : isTopLevel(declaration) ? 'local' : 'block');
      captures.push({ name, type: described, mutable, source, mentionSpan: offset });
    }

    let adaptation: InlineLambdaPlan['adaptation'];
    if (options.authored) {
      // Attachment grammar: exactly one block annotation in the trivia immediately before the tag.
      const trivia = file.text.slice(node.getFullStart(), node.getStart());
      const annotations = [...trivia.matchAll(/\/\*\s*@natlangSite\s+([A-Za-z_][A-Za-z0-9_-]*)\s*\*\//g)];
      if ((trivia.match(/@natlangSite/g) ?? []).length !== annotations.length || annotations.length > 1) {
        report(node, 'duplicate-site', 'Ambiguous or malformed @natlangSite annotation.'); return;
      }
      const label = annotations[0]?.[1];
      const labelKey = displayPath(file) + ':' + label;
      if (label && labels.has(labelKey)) { report(node, 'duplicate-site', 'Duplicate @natlangSite label: ' + label); return; }
      if (label) labels.add(labelKey);
      const expressions = ts.isNoSubstitutionTemplateLiteral(template) ? [] : template.templateSpans.map(span => span.expression.getText(file));
      const slotBindings = [...slotCaptures].sort();
      adaptation = { ...(label ? { label } : {}), templateStart: template.getStart(), templateEnd: template.getEnd(),
        expressions, visibleBindings: [...candidates.keys()].sort(), slotBindings };
    }
    const sourceSpan = spanOf(node, displayPath);
    const definitionId = `nl:${hexDigest(`${options.sourceRevision ?? ''}\0${sourceSpan.file}\0${sourceSpan.start}\0${sourceSpan.end}`).slice(0, 16)}`;
    plans.push({ sourceSpan, templateSpan, interpolations, definitionId, ...(adaptation ? { adaptation } : {}), strings, instructions: strings.join('${…}'),
      parameters: parameterTargets, ...(signature.open ? { openParameters: true } : {}), returns, captures,
      inheritedCodebaseRevision: options.codebaseRevision ?? '' });
  };

  // nl exists only as a template tag; anything else it could be mistaken for fails at run time as "nl is not
  // defined", which reads as "there is no nl", so each is named here with the form that works.
  const TAG_FORM = 'write the instructions as a template, nl`...`, and pass the values as arguments, for example ' +
    'await nl<{ amount: number }>`Read the amount paid on receipt.`(receipt). read_code("nl") shows more.';
  const misuse = (node: ts.Identifier): void => {
    const parent = node.parent;
    if (ts.isCallExpression(parent) && parent.expression === node)
      report(parent, 'nl-not-tag', `nl is a template tag, not a function taking a string: ${TAG_FORM}`);
    else report(node, 'nl-not-tag', `nl is a template tag with no value of its own to inspect or pass around: ${TAG_FORM}`);
  };
  for (const file of files) {
    const visit = (node: ts.Node): void => {
      if (ts.isTaggedTemplateExpression(node) && resolveIntrinsic(checker, node.tag) === 'nl') analyze(node);
      else if (ts.isTaggedTemplateExpression(node) && withCallOf(checker, node.tag)) analyze(node, withCallOf(checker, node.tag));
      else if (ts.isIdentifier(node) && node.text === 'nl') {
        const parent = node.parent;
        const declared = (ts.isVariableDeclaration(parent) || ts.isFunctionDeclaration(parent) || ts.isParameter(parent) ||
          ts.isClassDeclaration(parent)) && parent.name === node;
        // Importing or re-exporting nl names it, which is how application code gets it.
        if (ts.isImportSpecifier(parent) || ts.isExportSpecifier(parent) || ts.isImportClause(parent) || ts.isNamespaceImport(parent)) {}
        else if (declared) report(node, 'nl-shadowed', `nl is built in; declaring your own nl hides it. ${TAG_FORM}`);
        else if (ts.isPropertyAccessExpression(parent) && parent.expression === node && parent.name.text === 'with' &&
          ts.isCallExpression(parent.parent) && parent.parent.expression === parent && ts.isTaggedTemplateExpression(parent.parent.parent) &&
          parent.parent.parent.tag === parent.parent) {}
        else if (!(ts.isTaggedTemplateExpression(parent) && parent.tag === node) && !ts.isPropertyAccessExpression(parent) ||
          ts.isPropertyAccessExpression(parent) && parent.expression === node && !ts.isTaggedTemplateExpression(parent.parent)) {
          if (resolveIntrinsic(checker, node) === 'nl') misuse(node);
        }
      }
      ts.forEachChild(node, visit);
    };
    visit(file);
  }
  if (deferred.length) {
    const solutions = solveHoles(program, deferred.map((entry, id) => ({ id, node: entry.node })));
    deferred.forEach((entry, id) => {
      const solution = solutions?.get(id);
      if (solution?.kind === 'solved') {
        finish(entry.node, entry.call, { ...entry.signature, returns: solution.type, origin: 'use' }, solution, undefined, entry.withCall);
      } else if (solution?.kind === 'ambiguous') {
        report(entry.node, 'nl-ambiguous-signature', 'Uses of this `nl` result need different types: ' +
          solution.uses.map(use => `${use.typeText} (${use.why} in \`${use.text}\`)`).join('; ') + '. Write `nl<T>` with the one you mean.');
      } else {
        // Nothing says what the result is: the call runs with an open result instead of being refused. How the code
        // uses it still shapes what it asks for: the fields it reads, or a list.
        const open = solution?.fields.length ? `{ ${solution.fields.map(name => `${name}: unknown`).join(', ')} }` :
          solution?.indexed ? 'unknown[]' : 'unknown';
        finish(entry.node, entry.call, { ...entry.signature, origin: 'use' }, undefined,
          { text: 'any', natlang: open, aliases: {} }, entry.withCall);
      }
    });
    const order = new Map(files.map((file, index) => [displayPath(file), index]));
    plans.sort((a, b) => (order.get(a.sourceSpan.file) ?? 0) - (order.get(b.sourceSpan.file) ?? 0) || a.sourceSpan.start - b.sourceSpan.start);
  }
  // Soft values: opacity, typed literals, and recursive function types (S0 §2, §3, §8).
  const neuralese: NeuraleseLiteral[] = [];
  for (const file of files) neuralese.push(...checkNeuralese(checker, file, report, { recursiveTypes: options.recursiveTypes }).map(literal => ({ ...literal, file: displayPath(file) })));
  return { plans, diagnostics, neuralese };
}

/** The initial state and fixed arguments when an `nl` expression is the step of an `iterateOn` call. */
function iterationArguments(checker: ts.TypeChecker, expression: ts.Node): readonly ts.Expression[] | undefined {
  const parent = expression.parent;
  if (parent && ts.isPropertyAccessExpression(parent) && parent.expression === expression && parent.name.text === 'iterateOn' &&
      ts.isCallExpression(parent.parent) && parent.parent.expression === parent) return parent.parent.arguments;
  if (parent && ts.isCallExpression(parent) && parent.arguments[0] === expression && resolveIntrinsic(checker, parent.expression) === 'iterateOn')
    return parent.arguments.slice(1);
  return;
}

function isEnclosingParameter(declaration: ts.Declaration, node: ts.Node): boolean {
  if (!ts.isParameter(declaration)) return false;
  for (let current: ts.Node | undefined = node.parent; current; current = current.parent)
    if (current === declaration.parent) return true;
  return false;
}

export function isMutableBinding(declaration: ts.Declaration): boolean {
  if (ts.isParameter(declaration)) return true;
  if (ts.isBindingElement(declaration)) {
    let current: ts.Node = declaration;
    while (ts.isBindingElement(current) || ts.isObjectBindingPattern(current) || ts.isArrayBindingPattern(current)) current = current.parent;
    return ts.isParameter(current) || (ts.isVariableDeclaration(current) && isMutableBinding(current));
  }
  if (ts.isVariableDeclaration(declaration)) {
    const list = declaration.parent;
    return ts.isVariableDeclarationList(list) && !(list.flags & (ts.NodeFlags.Const | ts.NodeFlags.Using));
  }
  return false;
}

function isTopLevel(declaration: ts.Declaration): boolean {
  let current: ts.Node = declaration;
  while (current.parent && !ts.isSourceFile(current.parent)) {
    current = current.parent;
    if (ts.isBlock(current) || ts.isFunctionLike(current) || ts.isForStatement(current) || ts.isForOfStatement(current))
      return false;
  }
  return true;
}
