import ts from 'typescript';
import { hexDigest } from '../native/hash.js';
import { awaitedType, describeTarget, isPromiseLike, TargetError, type TargetDescriptor } from './targets.js';

export type SourceSpan = { file: string; start: number; end: number; line: number; column: number };

export type NatlangDiagnostic = SourceSpan & {
  code: 'nl-unknown-return' | 'nl-unknown-parameter' | 'nl-ambiguous-signature' | 'nl-sync-callback' |
    'nl-parameter-collision' | 'nl-unknown-name' | 'nl-spread' | 'nl-const-capture-write' |
    'forbidden-loop' | 'forbidden-dynamic-code' | 'recursion' | 'callable-scope' | 'reserved-property' |
    'duplicate-site' | 'iterate-step' | 'iterate-predicate' | 'module-collision' | 'typescript';
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
};

export type InlineLambdaPlan = {
  sourceSpan: SourceSpan;
  /** Source revision + AST span; stable for one source revision, not a user-facing name. */
  definitionId: string;
  /** Cooked template strings; interpolated values are inserted between them at invocation. */
  strings: string[];
  /** Instruction text with interpolations shown as `${…}` for listings and traces. */
  instructions: string;
  parameters: { name: string; type: TargetDescriptor }[];
  returns: TargetDescriptor;
  captures: CapturePlan[];
  inheritedCodebaseRevision: string;
};

export type BindingClassifier = (declaration: ts.Declaration) => CapturePlan['source'] | undefined;

export type InlineAnalysisOptions = {
  /** Files whose declarations may be captured in addition to the analyzed file itself (eval scope declarations). */
  scopeFiles?: readonly ts.SourceFile[];
  /** Names never captured (the result slot, debug state, plumbing). */
  excludedNames?: ReadonlySet<string>;
  classify?: BindingClassifier;
  sourceRevision?: string;
  codebaseRevision?: string;
  /** Display path used in spans and definition IDs. */
  displayPath?: (file: ts.SourceFile) => string;
};

const IDENTIFIER_TOKEN = /(?<![A-Za-z0-9_$])[A-Za-z_$][A-Za-z0-9_$]*(?![A-Za-z0-9_$])/g;
const BACKTICK_MENTION = /\\`([A-Za-z_$][A-Za-z0-9_$]*)\\`/g;

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

type Signature = { parameters?: { name: string; type: ts.Type }[]; returns?: ts.Type; origin: string };

export function analyzeInlineLambdas(program: ts.Program, files: readonly ts.SourceFile[],
  options: InlineAnalysisOptions = {}): { plans: InlineLambdaPlan[]; diagnostics: NatlangDiagnostic[] } {
  const checker = program.getTypeChecker();
  const plans: InlineLambdaPlan[] = [];
  const diagnostics: NatlangDiagnostic[] = [];
  const displayPath = options.displayPath ?? (file => file.fileName);
  const excluded = new Set(['result', 'nl', 'iterateOn', 'self', ...(options.excludedNames ?? [])]);
  const scopeFiles = new Set(options.scopeFiles ?? []);

  const report = (node: ts.Node, code: NatlangDiagnostic['code'], message: string, severity: 'error' | 'warning' = 'error') =>
    diagnostics.push({ ...spanOf(node, displayPath), code, message, severity });

  const target = (type: ts.Type, node: ts.Node, what: string, allowHost = true): TargetDescriptor | undefined => {
    try { return describeTarget(program, checker, type, { allowHost, location: node }); }
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
    const contextual = checker.getContextualType(outer as ts.Expression);
    if (!contextual || contextual.flags & ts.TypeFlags.Any) return;
    return awaited ? contextual : awaitedType(checker, contextual).type;
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

  const analyze = (node: ts.TaggedTemplateExpression): void => {
    const file = node.getSourceFile();
    const outerTag = unwrapParentheses(node);
    const call = outerTag.parent && ts.isCallExpression(outerTag.parent) && outerTag.parent.expression === outerTag ?
      outerTag.parent : undefined;
    const signature: Signature = { origin: 'none' };

    // 1. Explicit annotation.
    const annotation = node.typeArguments?.[0];
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

    // 2. Contextual callable type (callbacks, annotated locals).
    if (!call && !signature.parameters) {
      const contextual = checker.getContextualType(outerTag as ts.Expression);
      const signatures = contextual ? (contextual.isUnion() ? contextual.types : [contextual])
        .flatMap(type => type.getCallSignatures()) : [];
      if (signatures.length > 1) {
        report(node, 'nl-ambiguous-signature', 'The callback slot for this `nl` expression has several call signatures; annotate it with `nl<(x: T) => R>`.');
        return;
      }
      const only = signatures[0];
      if (only) {
        const returnType = only.getReturnType();
        const members = returnType.isUnion() ? returnType.types : [returnType];
        if (!(returnType.flags & ts.TypeFlags.Any) && !members.some(member => isPromiseLike(checker, member))) {
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
    let callSiteNames: string[] | undefined;
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
      callSiteNames = names;
      if (!signature.parameters) signature.parameters = names.map((name, index) => ({ name, type: types[index]! }));
      else if (signature.parameters.length !== names.length) {
        report(call, 'nl-ambiguous-signature', `This \`nl\` call passes ${names.length} arguments but its signature has ${signature.parameters.length}.`);
        return;
      }
      if (!signature.returns) {
        const outer = unwrapParentheses(call);
        const awaitNode = outer.parent && ts.isAwaitExpression(outer.parent) ? unwrapParentheses(outer.parent) : undefined;
        signature.returns = resultContext(awaitNode ?? outer, !!awaitNode);
        // 5. Propagate through a uniquely typed local: `const v = await nl`...`(x)`, later used in a typed position.
        const holder = (awaitNode ?? outer).parent;
        if (!signature.returns && holder && ts.isVariableDeclaration(holder) && !holder.type) {
          const contexts = laterUses(holder).map(use => resultContext(use, true)).filter((type): type is ts.Type => !!type);
          signature.returns = uniqueType(contexts, node, 'return type');
        }
      }
    } else if (!signature.parameters || !signature.returns) {
      // 5. `const judge = nl`...`` followed by calls.
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
        if (!signature.returns && calls.length) {
          const contexts = calls.map(item => {
            const outer = unwrapParentheses(item);
            const awaitNode = outer.parent && ts.isAwaitExpression(outer.parent) ? unwrapParentheses(outer.parent) : undefined;
            return resultContext(awaitNode ?? outer, !!awaitNode);
          }).filter((type): type is ts.Type => !!type);
          signature.returns = uniqueType(contexts, node, 'return type');
        }
      }
    }

    if (!signature.returns) {
      report(node, 'nl-unknown-return', 'Return type of this `nl` expression is unknown; annotate the target or write `nl<Verdict>`.');
      return;
    }
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
    const returns = target(signature.returns, node, 'return');
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
    const explicit = new Set(parameters.map(parameter => parameter.name));
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
    if (!ts.isNoSubstitutionTemplateLiteral(template)) for (const span of template.templateSpans) {
      const visit = (child: ts.Node): void => {
        if (ts.isIdentifier(child) && !(ts.isPropertyAccessExpression(child.parent) && child.parent.name === child)) {
          const symbol = checker.getSymbolAtLocation(child);
          if (symbol && candidates.get(child.text) === symbol && !mentions.has(child.text)) mentions.set(child.text, child.getStart());
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

    const sourceSpan = spanOf(node, displayPath);
    const definitionId = `nl:${hexDigest(`${options.sourceRevision ?? ''}\0${sourceSpan.file}\0${sourceSpan.start}\0${sourceSpan.end}`).slice(0, 16)}`;
    plans.push({ sourceSpan, definitionId, strings, instructions: strings.join('${…}'),
      parameters: parameterTargets, returns, captures, inheritedCodebaseRevision: options.codebaseRevision ?? '' });
    void callSiteNames;
  };

  for (const file of files) {
    const visit = (node: ts.Node): void => {
      if (ts.isTaggedTemplateExpression(node) && resolveIntrinsic(checker, node.tag) === 'nl') analyze(node);
      ts.forEachChild(node, visit);
    };
    visit(file);
  }
  return { plans, diagnostics };
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
