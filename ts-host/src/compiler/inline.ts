import ts from 'typescript';
import { hexDigest } from '../native/hash.js';
import { solveHoles } from './holes.js';
import { awaitedType, describeTarget, isPromiseLike, TargetError, type TargetDescriptor } from './targets.js';
import { checkGenericResults, checkNeuralese, neuraleseParts, type GenericInstantiation, type NeuraleseLiteral,
  type NeuraleseReadout } from './neuralese.js';

export type SourceSpan = { file: string; start: number; end: number; line: number; column: number };
export type InlineRebindSite = { start: number; end: number; templateStart: number; templateEnd: number;
  captureSources: Record<string, CapturePlan['source']> };

export type NatlangDiagnostic = SourceSpan & {
  code: 'nl-unknown-return' | 'nl-unknown-parameter' | 'nl-not-called' | 'nl-not-tag' | 'nl-shadowed' | 'nl-ambiguous-signature' | 'nl-sync-callback' |
    'nl-parameter-collision' | 'nl-unknown-name' | 'nl-spread' | 'nl-const-capture-write' |
    'forbidden-loop' | 'forbidden-dynamic-code' | 'recursion' | 'callable-scope' | 'reserved-property' |
    'duplicate-site' | 'iterate-step' | 'iterate-predicate' | 'module-collision' | 'typescript' | 'nl-undeclared-type' |
    'neuralese-opaque-access' | 'neuralese-condition' | 'neuralese-interpolation' | 'neuralese-untyped-literal' |
    'neuralese-nested' | 'neuralese-readout-sync' | 'neuralese-crisp-result' | 'type-recursive-function' | 'neuralese-file' | 'nl-explicit-captures' | 'nl-type-arguments' | 'undeclared-field' | 'untrusted-instruction';
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
  /** Explicit capture is evaluated at creation but hidden when a callable parameter has the same name. */
  shadowedByParameter?: true;
};

export type InlineLambdaPlan = {
  programId?: string;
  sourceSpan: SourceSpan;
  /** Authenticated prior-eval source for a template recompiled from a persistent eval helper. */
  sourceBackedHelper?: { name: string; sourceHash: string;
    declaredAction?: { toolCallId?: string; actionOrdinal: number; writtenCodeSha256: string };
    declarationSpan: { start: number; end: number; line: number; column: number } };
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

/** Whether `type` (or a member of a union) carries the \`Untrusted<T>\` brand. */
export function isUntrustedType(checker: ts.TypeChecker, type: ts.Type): boolean {
  if (type.isUnion()) return type.types.some(member => isUntrustedType(checker, member));
  return checker.getPropertiesOfType(type).some(property => property.name === '__natlangUntrusted');
}
const isTextType = (type: ts.Type): boolean => type.isUnion() ? type.types.every(isTextType) :
  !!(type.flags & (ts.TypeFlags.StringLike | ts.TypeFlags.TemplateLiteral)) || type.isIntersection() && type.types.some(isTextType);

/**
 * The untrusted part of an interpolated expression, if any. The expression itself being \`Untrusted<T>\` counts. So does a
 * text-valued expression built from one (\`message.slice(0, 20)\`, \`\\\`[\${message}]\\\`\`): text derived from untrusted text is
 * untrusted. A number or boolean computed from one (\`message.length\`) is not text and passes.
 */
function untrustedSplice(checker: ts.TypeChecker, expression: ts.Expression): ts.Expression | undefined {
  const type = checker.getTypeAtLocation(expression);
  if (isUntrustedType(checker, type)) return expression;
  if (!isTextType(type)) return undefined;
  let found: ts.Expression | undefined;
  const visit = (node: ts.Node): void => {
    if (found || ts.isFunctionLike(node)) return;
    if (ts.isExpression(node) && node !== expression && isUntrustedType(checker, checker.getTypeAtLocation(node))) { found = node; return; }
    ts.forEachChild(node, visit);
  };
  visit(expression);
  return found;
}

/** The suffix capture call in `nl<Result>`instructions`.with({ capture })`. */
export function suffixWithCallOf(node: ts.TaggedTemplateExpression): ts.CallExpression | undefined {
  const member = node.parent;
  if (!member || !ts.isPropertyAccessExpression(member) || member.expression !== node || member.name.text !== 'with') return;
  const call = member.parent;
  return call && ts.isCallExpression(call) && call.expression === member ? call : undefined;
}

/** An immediate input call after a suffix `.with(captures)` binding. */
function suffixWithInvocation(call: ts.CallExpression | undefined): ts.CallExpression | undefined {
  if (!call) return;
  const parent = call.parent;
  return parent && ts.isCallExpression(parent) && parent.expression === call ? parent : undefined;
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
  options: InlineAnalysisOptions = {}): { plans: InlineLambdaPlan[]; diagnostics: NatlangDiagnostic[]; neuralese: NeuraleseLiteral[];
    readouts: NeuraleseReadout[]; rebinds: InlineRebindSite[]; instantiations: GenericInstantiation[] } {
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
    const suffixCall = suffixWithCallOf(node);
    const suffixInvocation = suffixWithInvocation(suffixCall);
    let call = suffixCall ? suffixInvocation : outerTag.parent && ts.isCallExpression(outerTag.parent) && outerTag.parent.expression === outerTag ?
      outerTag.parent : undefined;
    withCall ??= suffixCall;
    const signature: Signature = { origin: 'none' };
    // Awaiting an `nl` tag creates its callable; it does not run a judgment. A saved callable may be awaited here and
    // invoked in this or a later eval, so diagnose only when the value is used as a non-callable result.
    // A map may intentionally build a reusable array of callable nl functions; allow it when
    // the array is later invoked through an indexed access. A function array that is never
    // called is still almost always a missing `(item)` after the tag.
    const arrow = outerTag.parent;
    const mappedNl = arrow && ts.isArrowFunction(arrow) && arrow.body === outerTag && arrow.parent &&
      ts.isCallExpression(arrow.parent) && arrow.parent.arguments.includes(arrow) &&
      ts.isPropertyAccessExpression(arrow.parent.expression) && arrow.parent.expression.name.text === 'map' &&
      arrow.parent.parent && ts.isVariableDeclaration(arrow.parent.parent) && arrow.parent.parent.initializer === arrow.parent ?
      arrow.parent.parent : undefined;
    const indexedCalls = mappedNl ? laterUses(mappedNl).flatMap(use => {
      const indexed = use.parent;
      if (!indexed || !ts.isElementAccessExpression(indexed) || indexed.expression !== use) return [];
      const invocation = indexed.parent;
      return invocation && ts.isCallExpression(invocation) && invocation.expression === indexed ? [invocation] : [];
    }) : [];
    if (arrow && ts.isArrowFunction(arrow) && arrow.body === outerTag && arrow.parent && ts.isCallExpression(arrow.parent) &&
        arrow.parent.arguments.includes(arrow)) {
      if (indexedCalls.length) call = indexedCalls[0];
      else {
        report(node, 'nl-not-called', 'This arrow returns the `nl` function itself, never called: nl`...` creates a function. ' +
          'Call it, and await the results: `await Promise.all(items.map(item => nl`Does item …?`(item)))`, or save the ' +
          'callables and invoke them later.');
        return;
      }
    }
    if (outerTag.parent && ts.isAwaitExpression(outerTag.parent)) {
      const awaited = outerTag.parent;
      const declaration = awaited.parent && ts.isVariableDeclaration(awaited.parent) && awaited.parent.initializer === awaited ?
        awaited.parent : undefined;
      const contextual = checker.getContextualType(awaited);
      const savedCallable = !!declaration && (!contextual || contextual.getCallSignatures().length > 0);
      if (!savedCallable) {
        report(node, 'nl-not-called', 'This expression is the callable created by `nl`, not a judgment result. ' +
          'Call it with the values it should judge, as in `await nl`...`(value)`, or store the callable in a variable to call later.');
        return;
      }
    }

    // `nl.with<CaptureRecord, Result>` types the listed capture record and the child result separately;
    // the callable's argument type still comes from its later invocation (or one full callable signature).
    const twoArgumentWith = withCall === node.tag && (withCall?.typeArguments?.length ?? 0) === 2;
    if ((node.typeArguments?.length ?? 0) > 1 || ((withCall?.typeArguments?.length ?? 0) > 1 && !twoArgumentWith)) {
      report(node, 'nl-type-arguments', '`nl` accepts one type argument: the child result type or full callable signature. ' +
        '`nl.with` accepts one result/signature type, or two types as `<CaptureRecord, Result>`. The first type checks only ' +
        'the listed capture record; child arguments are passed separately and inferred from the call. ' +
        'Use one full callable signature when you need to declare child parameter types.');
      return;
    }

    // 1. Explicit annotation (`nl<F>`, or `nl.with<F>({ … })`).
    const annotation = node.typeArguments?.[0] ?? (twoArgumentWith ? withCall?.typeArguments?.[1] : withCall?.typeArguments?.[0]);
    if (annotation) {
      // Inline calls written later by an interpreter run in eval's declared scope. A type alias
      // mentioned only in an earlier natural-language instruction is not in that scope; TypeScript
      // represents that unresolved reference as an open type, which would silently erase the child's
      // declared result contract. Catch it before target() converts unknown/any to `unknown`.
      let unresolved: ts.TypeReferenceNode | undefined;
      const findUnresolved = (typeNode: ts.Node): void => {
        if (unresolved) return;
        if (ts.isTypeReferenceNode(typeNode) && !checker.getSymbolAtLocation(typeNode.typeName)) {
          unresolved = typeNode;
          return;
        }
        ts.forEachChild(typeNode, findUnresolved);
      };
      findUnresolved(annotation);
      if (unresolved) {
        const name = unresolved.typeName.getText(node.getSourceFile());
        report(unresolved, 'nl-undeclared-type', `Type name ${JSON.stringify(name)} is not declared in this eval scope, so the inline result is untyped; declare it in scope or use a type written in this annotation.`);
        return;
      }
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
      for (const invocation of indexedCalls) if (invocation.arguments.length !== call.arguments.length ||
          invocation.arguments.some((argument, index) => !checker.isTypeAssignableTo(widen(checker.getTypeAtLocation(argument)),
            widen(checker.getTypeAtLocation(call!.arguments[index]!))) &&
            !checker.isTypeAssignableTo(widen(checker.getTypeAtLocation(call!.arguments[index]!)), widen(checker.getTypeAtLocation(argument))))) {
        report(invocation, 'nl-ambiguous-signature', 'Calls of this saved `nl` function array disagree about its parameters; annotate the function signature.');
        return;
      }
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
    if (withCall.arguments.length !== 1 || !argument) {
      report(withCall, 'nl-explicit-captures', 'nl.with takes one finite record of captures, as in ' +
        '`nl.with({ rubric })` or `nl.with(context)`, where context has known fields.');
      return;
    }
    const captureSchemaNode = withCall === node.tag && withCall.typeArguments?.length === 2 ? withCall.typeArguments[0] : undefined;
    const fromRecord = !ts.isObjectLiteralExpression(argument);
    const properties: { name: string; declaration?: ts.Declaration }[] = [];
    const actualTypes = new Map<string, ts.Type>();
    if (fromRecord) {
      const recordType = checker.getTypeAtLocation(argument);
      const finite = !!(recordType.flags & ts.TypeFlags.Object) && !checker.getSignaturesOfType(recordType, ts.SignatureKind.Call).length &&
        !checker.getIndexTypeOfType(recordType, ts.IndexKind.String) && !checker.getIndexTypeOfType(recordType, ts.IndexKind.Number);
      const members = checker.getPropertiesOfType(recordType);
      if (!finite || !members.length || members.some(member => !(member.declarations ?? []).some(declaration =>
        ts.isPropertySignature(declaration) || ts.isPropertyDeclaration(declaration) || ts.isPropertyAssignment(declaration)))) {
        report(argument, 'nl-explicit-captures', 'nl.with needs a record expression with a finite set of known fields; ' +
          'use an object literal or give the record a type with named data fields.');
        return;
      }
      for (const member of members) {
        const name = member.getName();
        if (!IDENTIFIER_NAME.test(name) || excluded.has(name) || name.startsWith('__natlang')) {
          report(argument, 'nl-explicit-captures', `Record field ${JSON.stringify(name)} cannot be a capture name.`);
          return;
        }
        properties.push({ name, declaration: member.valueDeclaration ?? member.declarations?.[0] });
        actualTypes.set(name, checker.getTypeOfSymbolAtLocation(member, argument));
      }
    }
    const captures: CapturePlan[] = [];
    const entries = fromRecord ? properties : argument.properties.map(property => ({ property }));
    for (const entry of entries) {
      if (fromRecord) {
        const { name, declaration } = entry as typeof properties[number];
        const type = declaration ? checker.getTypeOfSymbolAtLocation(checker.getPropertyOfType(checker.getTypeAtLocation(argument), name)!, argument) :
          checker.getAnyType();
        let described: TargetDescriptor;
        try { described = describeTarget(program, checker, type, { allowHost: true, location: node }); }
        catch (error) {
          if (!(error instanceof TargetError)) throw error;
          described = { text: checker.typeToString(type, node), aliases: {}, host: { kind: 'shape', members: [] } };
        }
        let recordRoot: ts.Expression = argument;
        while (ts.isParenthesizedExpression(recordRoot) || ts.isAsExpression(recordRoot) || ts.isNonNullExpression(recordRoot) ||
          ts.isSatisfiesExpression(recordRoot)) recordRoot = recordRoot.expression;
        while (ts.isPropertyAccessExpression(recordRoot)) recordRoot = recordRoot.expression;
        const recordSymbol = ts.isIdentifier(recordRoot) ? checker.getSymbolAtLocation(recordRoot) : undefined;
        const recordDeclaration = recordSymbol?.valueDeclaration ?? recordSymbol?.declarations?.[0];
        const source = (recordDeclaration && options.classify?.(recordDeclaration)) ?? (described.host ? 'handle' :
          recordDeclaration && ts.isParameter(recordDeclaration) ? 'input' : recordDeclaration && isTopLevel(recordDeclaration) ? 'local' : 'block');
        captures.push({ name, type: described, mutable: false, source, mentionSpan: argument.getStart(), mode: 'snapshot' });
        continue;
      }
      const property = (entry as { property: ts.ObjectLiteralElementLike }).property;
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
      actualTypes.set(name, type);
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
    if (captureSchemaNode) {
      const schema = checker.getTypeFromTypeNode(captureSchemaNode);
      const schemaMembers = checker.getPropertiesOfType(schema);
      const schemaFinite = !schema.isUnion() && !!(schema.flags & ts.TypeFlags.Object) &&
        !checker.getIndexTypeOfType(schema, ts.IndexKind.String) && !checker.getIndexTypeOfType(schema, ts.IndexKind.Number) &&
        schemaMembers.every(member => (member.declarations ?? []).some(declaration => ts.isPropertySignature(declaration) ||
          ts.isPropertyDeclaration(declaration) || ts.isPropertyAssignment(declaration)));
      if (!schemaFinite) {
        report(captureSchemaNode, 'nl-explicit-captures', 'The first `nl.with` type argument must be a finite record schema with named data fields.');
        return;
      }
      const expectedByName = new Map(schemaMembers.map(member => [member.getName(), member]));
      const unknown = [...actualTypes.keys()].find(name => !expectedByName.has(name));
      if (unknown) {
        report(withCall, 'nl-explicit-captures', `Capture ${JSON.stringify(unknown)} is not declared by the ` +
          '`nl.with` capture schema.');
        return;
      }
      const missing = schemaMembers.find(member => !(member.flags & ts.SymbolFlags.Optional) && !actualTypes.has(member.getName()));
      if (missing) {
        report(withCall, 'nl-explicit-captures', `Required capture ${JSON.stringify(missing.getName())} is missing from the ` +
          "`nl.with` capture record.");
        return;
      }
      for (const [name, actual] of actualTypes) {
        const expectedMember = expectedByName.get(name)!;
        const expected = checker.getTypeOfSymbolAtLocation(expectedMember, captureSchemaNode);
        if (!checker.isTypeAssignableTo(actual, expected)) {
          report(withCall, 'nl-explicit-captures', `Capture ${JSON.stringify(name)} has type ` +
            `\`${checker.typeToString(actual, withCall)}\`, which is not assignable to the declared capture type ` +
            `\`${checker.typeToString(expected, captureSchemaNode)}\`.`);
          return;
        }
      }
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
    // Instruction text is the author's. An `Untrusted<T>` value reaches the model as an argument, shown as data.
    if (!ts.isNoSubstitutionTemplateLiteral(template)) for (const span of template.templateSpans) {
      const culprit = untrustedSplice(checker, span.expression);
      if (culprit) report(culprit, 'untrusted-instruction', `\`${culprit.getText(file)}\` is untrusted data (Untrusted<T>) and cannot be interpolated into the ` +
        'text of an `nl` call; pass it as an argument instead, as in nl`Summarize the message.`(message), so the model reads it as quoted data.');
    }
    const explicit = new Set(parameters.map(parameter => parameter.name));
    const softBody = softBodyOf(template);
    if (withCall || softBody) {
      // Explicit captures: exactly the listed ones (none for a bare soft literal); nothing is captured by mention.
      const listedCaptures = withCall ? explicitCaptures(withCall, node) : [];
      if (!listedCaptures) return;
      const captures = listedCaptures.map(capture => explicit.has(capture.name) ?
        { ...capture, shadowedByParameter: true as const } : capture);
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
      else if (ts.isTaggedTemplateExpression(node) && suffixWithCallOf(node) && resolveIntrinsic(checker, node.tag) === 'nl')
        analyze(node, suffixWithCallOf(node));
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
  const readouts: NeuraleseReadout[] = [];
  for (const file of files) neuralese.push(...checkNeuralese(checker, file, report,
    { recursiveTypes: options.recursiveTypes, readouts }).map(literal => ({ ...literal, file: displayPath(file) })));
  // Representation-generic results: the instance each call site's expected type picks (DECISIONS.md 2026-10-09).
  const instantiations: GenericInstantiation[] = [];
  for (const file of files) checkGenericResults(checker, file, report, instantiations);
  const rebinds: InlineRebindSite[] = [];
  const tagPlans = new Map<ts.Symbol, { tag: ts.TaggedTemplateExpression; plan: InlineLambdaPlan }>();
  const planAt = (tag: ts.TaggedTemplateExpression) => plans.find(plan => plan.sourceSpan.file === displayPath(tag.getSourceFile()) &&
    plan.sourceSpan.start === tag.getStart());
  const captureSource = (expression: ts.Expression): CapturePlan['source'] => {
    let root = expression;
    while (ts.isParenthesizedExpression(root) || ts.isAsExpression(root) || ts.isNonNullExpression(root) || ts.isSatisfiesExpression(root))
      root = root.expression;
    while (ts.isPropertyAccessExpression(root) || ts.isElementAccessExpression(root)) root = root.expression;
    const symbol = ts.isIdentifier(root) ? checker.getSymbolAtLocation(root) : undefined;
    const declaration = symbol?.valueDeclaration ?? symbol?.declarations?.[0];
    let host = false;
    try { host = !!describeTarget(program, checker, checker.getTypeAtLocation(expression), { allowHost: true, location: expression }).host; }
    catch (error) { if (!(error instanceof TargetError)) throw error; }
    return (declaration && options.classify?.(declaration)) ?? (host ? 'handle' :
      declaration && ts.isParameter(declaration) ? 'input' : declaration && isTopLevel(declaration) ? 'local' : 'block');
  };
  for (const file of files) {
    const findTags = (node: ts.Node): void => {
      if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.initializer) {
        let initializer: ts.Expression = node.initializer;
        while (ts.isParenthesizedExpression(initializer) || ts.isAsExpression(initializer) || ts.isNonNullExpression(initializer))
          initializer = initializer.expression;
        const symbol = checker.getSymbolAtLocation(node.name);
        if (symbol && ts.isTaggedTemplateExpression(initializer) && resolveIntrinsic(checker, initializer.tag) === 'nl') {
          const plan = planAt(initializer);
          if (plan) tagPlans.set(symbol, { tag: initializer, plan });
        } else if (symbol && ts.isIdentifier(initializer) &&
            ts.isVariableDeclarationList(node.parent) && (node.parent.flags & ts.NodeFlags.Const)) {
          const original = tagPlans.get(checker.getSymbolAtLocation(initializer)!);
          if (original) tagPlans.set(symbol, original);
        }
      }
      ts.forEachChild(node, findTags);
    };
    findTags(file);
  }
  const tagBindingFor = (expression: ts.Expression): { tag: ts.TaggedTemplateExpression; plan: InlineLambdaPlan } | undefined => {
    while (ts.isParenthesizedExpression(expression) || ts.isAsExpression(expression) || ts.isNonNullExpression(expression))
      expression = expression.expression;
    if (ts.isIdentifier(expression)) return tagPlans.get(checker.getSymbolAtLocation(expression)!);
    if (ts.isCallExpression(expression) && ts.isPropertyAccessExpression(expression.expression) && expression.expression.name.text === 'with')
      return tagBindingFor(expression.expression.expression);
    return undefined;
  };
  for (const file of files) {
    const findRebinds = (node: ts.Node): void => {
      if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) && node.expression.name.text === 'with') {
        const binding = tagBindingFor(node.expression.expression);
        if (binding) {
          const argument = node.arguments[0];
          if (node.arguments.length !== 1 || !argument) {
            report(node, 'nl-explicit-captures', 'Saved inline .with(...) takes one finite record of capture values.');
          } else {
            let valid = true;
            if (ts.isObjectLiteralExpression(argument) && argument.properties.some(property => !ts.isPropertyAssignment(property) &&
                !ts.isShorthandPropertyAssignment(property))) {
              report(argument, 'nl-explicit-captures', 'Saved inline .with(record) uses named data fields; spreads, methods and computed names are not captures.');
              valid = false;
            }
            const recordType = checker.getTypeAtLocation(argument);
            const fields = checker.getPropertiesOfType(recordType);
            const finite = !!(recordType.flags & ts.TypeFlags.Object) && !checker.getSignaturesOfType(recordType, ts.SignatureKind.Call).length &&
              !checker.getIndexTypeOfType(recordType, ts.IndexKind.String) && !checker.getIndexTypeOfType(recordType, ts.IndexKind.Number) &&
              fields.every(field => (field.declarations ?? []).some(declaration => ts.isPropertySignature(declaration) ||
                ts.isPropertyDeclaration(declaration) || ts.isPropertyAssignment(declaration) || ts.isShorthandPropertyAssignment(declaration)));
            if (!finite) {
              report(argument, 'nl-explicit-captures', 'Saved inline .with(record) needs a finite record with named data fields.');
              valid = false;
            }
            const expected = binding.plan.captures.map(capture => capture.name).sort();
            const actual = fields.map(field => field.getName()).sort();
            if (actual.length !== expected.length || actual.some((name, index) => name !== expected[index])) {
              report(node, 'nl-explicit-captures', `Saved inline .with(...) needs exactly the original captures: ${expected.join(', ')}.`);
              valid = false;
            }
            const incompatible = binding.plan.captures.find(capture => {
              const original = checker.getSymbolsInScope(binding.tag, ts.SymbolFlags.Value | ts.SymbolFlags.Alias)
                .find(candidate => candidate.name === capture.name);
              const originalDeclaration = original?.valueDeclaration ?? original?.declarations?.[0];
              const wanted = original && originalDeclaration ? checker.getTypeOfSymbolAtLocation(original, originalDeclaration) : undefined;
              const property = checker.getPropertyOfType(recordType, capture.name);
              const propertyDeclaration = property?.valueDeclaration ?? property?.declarations?.[0];
              const actualType = property && propertyDeclaration ? checker.getTypeOfSymbolAtLocation(property, propertyDeclaration) : undefined;
              if (ts.isObjectLiteralExpression(argument)) {
                const entry = argument.properties.find(item => (ts.isPropertyAssignment(item) || ts.isShorthandPropertyAssignment(item)) &&
                  (ts.isIdentifier(item.name) || ts.isStringLiteral(item.name)) && item.name.text === capture.name);
                if (entry) return !wanted || !checker.isTypeAssignableTo(ts.isPropertyAssignment(entry) ?
                  checker.getTypeAtLocation(entry.initializer) : checker.getTypeAtLocation(entry.name ?? entry), wanted);
              }
              return !wanted || !actualType || !checker.isTypeAssignableTo(actualType, wanted);
            });
            if (incompatible) {
              report(node, 'nl-explicit-captures', `Saved inline .with(...) capture ${JSON.stringify(incompatible.name)} must match the original capture type.`);
              valid = false;
            }
            if (valid) {
              const captureSources = Object.fromEntries(binding.plan.captures.map(capture => {
                let expression: ts.Expression | undefined;
                if (ts.isObjectLiteralExpression(argument)) {
                  const entry = argument.properties.find(item => (ts.isPropertyAssignment(item) || ts.isShorthandPropertyAssignment(item)) &&
                    (ts.isIdentifier(item.name) || ts.isStringLiteral(item.name)) && item.name.text === capture.name);
                  if (entry && ts.isPropertyAssignment(entry)) expression = entry.initializer;
                  else if (entry && ts.isShorthandPropertyAssignment(entry)) expression = entry.name;
                }
                return [capture.name, captureSource(expression ?? argument)];
              })) as Record<string, CapturePlan['source']>;
              rebinds.push({ start: node.getStart(file), end: node.getEnd(),
                templateStart: binding.tag.getStart(file), templateEnd: binding.tag.getEnd(), captureSources });
            }
          }
        }
      }
      ts.forEachChild(node, findRebinds);
    };
    findRebinds(file);
  }
  return { plans, diagnostics, neuralese, readouts, rebinds, instantiations };
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
