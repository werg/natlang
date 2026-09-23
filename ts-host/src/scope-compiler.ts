import ts from 'typescript';
import type { InlineLambdaPlan, NatlangDiagnostic } from './compiler/inline.js';
import { authoredCallables, checkConstrainedSource, findRecursion, lexicalResolver } from './compiler/policy.js';

/** Stable front-end contract for model-authored scope eval snippets. */
export const SCOPE_COMPILE_VERSION = 2 as const;

export type ScopeBinding = {
  name: string;
  kind: 'const' | 'let' | 'var' | 'function';
  mutable: boolean;
  annotation?: string;
  initializer?: string;
  /** Module namespaces, HTTP responses, and local functions live only for this eval. */
  transient?: boolean;
  start: number;
  end: number;
};

export type ScopeSourceSpan = {
  start: number;
  end: number;
  line: number;
  column: number;
};

export type ScopeCompileDiagnostic = ScopeSourceSpan & {
  code: 'typescript-syntax' | 'forbidden-ambient' | 'forbidden-dynamic-code' |
    'forbidden-control' | 'forbidden-prototype-mutation' | 'invalid-binding' | 'forbidden-loop' | 'recursion' |
    NatlangDiagnostic['code'];
  message: string;
};

export type ScopeCompileOptions = {
  /** Lambda parameters exposed as ordinary mutable lexical bindings. */
  inputBindings?: readonly string[];
  /** Existing persistent locals copied into the transaction before execution. */
  localBindings?: readonly ScopeExistingBinding[];
  /** Checked async callables generated over the runtime's host dispatcher. */
  helperBindings?: readonly string[];
  /** Opaque host values supplied by the runtime outside the portable snapshot. */
  opaqueBindings?: readonly string[];
  /** Enabled only when the evaluator exposes application-scoped module loading. */
  allowModules?: boolean;
  allowNetwork?: boolean;
  /** Host services, injected as immutable named bindings. */
  serviceBindings?: readonly string[];
  /** Live captured bindings of an inline lambda. Const captures are immutable in eval. */
  captureBindings?: readonly { name: string; mutable: boolean }[];
  /** Type-checked analysis of `nl` expressions (plans and diagnostics with snippet-relative spans). */
  analyze?: (source: string) => { plans: InlineLambdaPlan[]; diagnostics: NatlangDiagnostic[] };
  /** Prefix for runtime recursion-guard IDs of functions authored in this eval. */
  guardPrefix?: string;
};

export type ScopeExistingBinding = { name: string; mutable: boolean; annotation?: string };

export type ScopeCompileResult = {
  version: typeof SCOPE_COMPILE_VERSION;
  ok: boolean;
  entrypoint: '__natlang_scope';
  bindings: ScopeBinding[];
  finalExpression?: ScopeSourceSpan;
  /** The snippet has a final expression or a top-level return. */
  producesResult: boolean;
  /** Locals directly returned by this eval; their declared function result type provides context. */
  resultBindings?: string[];
  diagnostics: ScopeCompileDiagnostic[];
  repairs: ScopeCompileDiagnostic[];
  /** TypeScript body after applying final-expression REPL semantics. */
  body?: string;
  /** Inline `nl` plans, referenced by index from the lowered program. */
  plans?: InlineLambdaPlan[];
  /** Standalone ES2022 program defining an async entrypoint returning { result, bindings }.
   * Its third argument is a host dispatcher `(name, positionalArgs) => value | Promise<value>`;
   * helper function objects never enter the portable scope snapshot.
   */
  program?: string;
};

const ENTRYPOINT = '__natlang_scope' as const;
const PREFIX = `async function ${ENTRYPOINT}(__inputs: Readonly<Record<string, unknown>>, __locals: Readonly<Record<string, unknown>>, __captures: Readonly<Record<string, unknown>>) {\n`;

/**
 * Helpers every compiled program relies on. The runtime prepends this to eval and callable-folder
 * TypeScript and supplies `__live` (inputs, locals, captures, callables, and the output sink) by reference.
 */
export const SCOPE_RUNTIME_PRELUDE = `const __natlang_plain = (value: any) => !!value && typeof value === 'object' && !Array.isArray(value) &&
  Object.prototype.toString.call(value) === '[object Object]' &&
  (Object.getPrototypeOf(value) === null || Object.getPrototypeOf(Object.getPrototypeOf(value)) === null);
const __natlang_copy = (value: any): any => Array.isArray(value) ? value.map(__natlang_copy) :
  __natlang_plain(value) ? Object.fromEntries(Object.entries(value).map(([k, v]) => [k, __natlang_copy(v)])) : value;
const __natlang_frozen = (value: any): any => {
  if (Array.isArray(value) || __natlang_plain(value)) { Object.freeze(value); for (const item of Object.values(value)) __natlang_frozen(item); }
  return value;
};
const __natlang_callable = (name: string) => __live.callables[name];
const __natlang_output = (value: unknown) => { __live.finish(value); return null; };
const __natlang_inline = (index: number, values: unknown[], accessors: unknown) => __live.inline(index, values, accessors);
const __natlang_finite = (source: any) => __live.finite(source);
const __natlang_guard = (id: string, fn: () => unknown) => __live.guard(id, fn);
const iterateOn = __live.iterateOn;
`;
const SUFFIX = '\n}\n';
const IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;
/** CommonJS names: eval code imports packages with `import` instead. */
const MODULE_AMBIENTS = new Set(['require', 'module']);

function namesOf(name: ts.BindingName): ts.Identifier[] {
  if (ts.isIdentifier(name)) return [name];
  return name.elements.flatMap(element => ts.isOmittedExpression(element) ? [] : namesOf(element.name));
}

/** Translate ordinary TypeScript annotations to the portable natlang type tree. */
function portableAnnotation(node: ts.TypeNode, file: ts.SourceFile): string | undefined {
  const primitive = new Map<number, string>([
    [ts.SyntaxKind.StringKeyword, 'string'], [ts.SyntaxKind.NumberKeyword, 'number'],
    [ts.SyntaxKind.BooleanKeyword, 'boolean'], [ts.SyntaxKind.NullKeyword, 'null'],
  ]);
  const direct = primitive.get(node.kind);
  if (direct) return direct;
  if (ts.isLiteralTypeNode(node)) return node.literal.getText(file);
  if (ts.isParenthesizedTypeNode(node)) {
    const inner = portableAnnotation(node.type, file); return inner ? `(${inner})` : undefined;
  }
  if (ts.isArrayTypeNode(node)) {
    const element = portableAnnotation(node.elementType, file); return element ? `(${element})[]` : undefined;
  }
  if (ts.isUnionTypeNode(node)) {
    const members = node.types.map(type => portableAnnotation(type, file));
    return members.every(Boolean) ? members.join(' | ') : undefined;
  }
  if (ts.isTypeLiteralNode(node)) {
    const fields: string[] = [];
    for (const member of node.members) {
      if (!ts.isPropertySignature(member) || !member.type || !member.name) return;
      const type = portableAnnotation(member.type, file);
      if (!type) return;
      fields.push(`${member.name.getText(file)}${member.questionToken ? '?' : ''}: ${type}`);
    }
    return `{ ${fields.join(', ')} }`;
  }
  if (ts.isTypeOperatorNode(node) && node.operator === ts.SyntaxKind.ReadonlyKeyword)
    return portableAnnotation(node.type, file);
  if (ts.isTypeReferenceNode(node)) {
    const name = node.typeName.getText(file), args = node.typeArguments ?? [];
    if ((name === 'Array' || name === 'ReadonlyArray') && args.length === 1) {
      const element = portableAnnotation(args[0]!, file); return element ? `(${element})[]` : undefined;
    }
    if (name === 'Record' && args.length === 2 && args[0]!.kind === ts.SyntaxKind.StringKeyword) {
      const value = portableAnnotation(args[1]!, file); return value ? `Record<string, ${value}>` : undefined;
    }
    if (!args.length) return name;
  }
  return;
}

function isPropertyName(node: ts.Identifier): boolean {
  const parent = node.parent;
  return (ts.isPropertyAccessExpression(parent) && parent.name === node) ||
    ((ts.isPropertyAssignment(parent) || ts.isMethodDeclaration(parent) || ts.isPropertyDeclaration(parent) ||
      ts.isPropertySignature(parent) || ts.isMethodSignature(parent) || ts.isBindingElement(parent)) &&
      parent.name === node) ||
    (ts.isLabeledStatement(parent) && parent.label === node) ||
    ((ts.isBreakStatement(parent) || ts.isContinueStatement(parent)) && parent.label === node);
}

function isDeclarationName(node: ts.Identifier): boolean {
  const parent = node.parent;
  return (ts.isVariableDeclaration(parent) || ts.isParameter(parent) || ts.isFunctionDeclaration(parent) ||
    ts.isFunctionExpression(parent) || ts.isArrowFunction(parent) || ts.isClassDeclaration(parent) ||
    ts.isClassExpression(parent) || ts.isTypeAliasDeclaration(parent) || ts.isInterfaceDeclaration(parent) ||
    ts.isEnumDeclaration(parent) || ts.isImportClause(parent) || ts.isImportSpecifier(parent) ||
    ts.isNamespaceImport(parent) || ts.isBindingElement(parent)) && parent.name === node;
}

function propertyText(expression: ts.Expression): string | undefined {
  if (ts.isPropertyAccessExpression(expression)) return expression.name.text;
  if (ts.isElementAccessExpression(expression) && expression.argumentExpression &&
      (ts.isStringLiteral(expression.argumentExpression) || ts.isNoSubstitutionTemplateLiteral(expression.argumentExpression)))
    return expression.argumentExpression.text;
  return undefined;
}

function assignmentTarget(node: ts.Node): ts.Expression | undefined {
  if (ts.isBinaryExpression(node) && node.operatorToken.kind >= ts.SyntaxKind.FirstAssignment &&
      node.operatorToken.kind <= ts.SyntaxKind.LastAssignment) return node.left;
  if (ts.isDeleteExpression(node)) return node.expression;
  if ((ts.isPrefixUnaryExpression(node) || ts.isPostfixUnaryExpression(node)) &&
      (node.operator === ts.SyntaxKind.PlusPlusToken || node.operator === ts.SyntaxKind.MinusMinusToken))
    return node.operand;
  return undefined;
}

/** Parse and compile a sandboxed TypeScript snippet without executing it. */
/** End positions of `nl` tagged templates that are awaited without being called. */
function uncalledInlineFunctions(file: ts.SourceFile): number[] {
  const ends: number[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isTaggedTemplateExpression(node) && ts.isIdentifier(node.tag) && node.tag.text === 'nl') {
      let outer: ts.Node = node;
      while (outer.parent && ts.isParenthesizedExpression(outer.parent)) outer = outer.parent;
      if (outer.parent && ts.isAwaitExpression(outer.parent)) ends.push(node.getEnd());
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return ends;
}

export function compileScopeSnippet(source: string, options: ScopeCompileOptions = {}): ScopeCompileResult {
  if (typeof source !== 'string') throw new TypeError('scope source must be a string');
  if (options.allowModules) {
    const module = ts.createSourceFile('imports.ts', source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
    const imports = module.statements.filter(ts.isImportDeclaration);
    if (imports.length) {
      let lowered = source;
      for (const statement of [...imports].reverse()) {
        const clause = statement.importClause;
        let text = '';
        const typeOnlyBindings = clause?.namedBindings && ts.isNamedImports(clause.namedBindings)
          && clause.namedBindings.elements.length > 0 && clause.namedBindings.elements.every(binding => binding.isTypeOnly);
        if (!clause?.isTypeOnly && !(typeOnlyBindings && !clause?.name)) {
          const call = `await import(${statement.moduleSpecifier.getText(module)})`;
          const fields: string[] = [];
          if (clause?.name) fields.push(`default: ${clause.name.text}`);
          const bindings = clause?.namedBindings;
          if (bindings && ts.isNamedImports(bindings)) for (const binding of bindings.elements) {
            if (!binding.isTypeOnly) fields.push(`${binding.propertyName?.text ?? binding.name.text}: ${binding.name.text}`);
          }
          if (bindings && ts.isNamespaceImport(bindings)) {
            text = `const ${bindings.name.text} = ${call};`;
            if (clause?.name) text += ` const {default: ${clause.name.text}} = ${call};`;
          } else text = fields.length ? `const { ${fields.join(', ')} } = ${call};` : `${call};`;
        }
        lowered = lowered.slice(0, statement.getStart(module)) + text + lowered.slice(statement.end);
      }
      return compileScopeSnippet(lowered, options);
    }
  }
  const wrapped = PREFIX + source + SUFFIX;
  const file = ts.createSourceFile('natlang-scope.ts', wrapped, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  // `await nl`...`` awaits the function nl`...` creates, which is never what is meant: it runs the judgment now,
  // on the names its instructions mention and its interpolations, as `await nl`...`()` does.
  const uncalled = uncalledInlineFunctions(file);
  if (uncalled.length) {
    let called = source;
    for (const end of uncalled.sort((a, b) => b - a)) called = called.slice(0, end - PREFIX.length) + '()' + called.slice(end - PREFIX.length);
    return compileScopeSnippet(called, options);
  }
  const fn = file.statements.find(ts.isFunctionDeclaration);
  if (!fn?.body) throw new Error('internal scope compiler wrapper failure');
  const diagnostics: ScopeCompileDiagnostic[] = [];
  const repairs: ScopeCompileDiagnostic[] = [];

  const span = (nodeOrStart: ts.Node | number, length?: number): ScopeSourceSpan => {
    const absolute = typeof nodeOrStart === 'number' ? nodeOrStart : nodeOrStart.getStart(file);
    const rawStart = Math.max(0, Math.min(source.length, absolute - PREFIX.length));
    const rawEnd = Math.max(rawStart, Math.min(source.length,
      typeof nodeOrStart === 'number' ? rawStart + (length ?? 1) : nodeOrStart.getEnd() - PREFIX.length));
    const location = file.getLineAndCharacterOfPosition(Math.max(PREFIX.length, absolute));
    return { start: rawStart, end: rawEnd, line: Math.max(1, location.line), column: location.character + 1 };
  };
  const rawSpan = (start: number, end = start): ScopeSourceSpan => {
    const location = file.getLineAndCharacterOfPosition(PREFIX.length + start);
    return { start, end, line: Math.max(1, location.line), column: location.character + 1 };
  };
  const add = (code: ScopeCompileDiagnostic['code'], message: string, node: ts.Node): void => {
    diagnostics.push({ code, message, ...span(node) });
  };

  const parseDiagnostics = (file as unknown as { parseDiagnostics?: readonly ts.DiagnosticWithLocation[] }).parseDiagnostics ?? [];
  for (const diagnostic of parseDiagnostics) {
    if (diagnostic.start === undefined) continue;
    diagnostics.push({ code: 'typescript-syntax',
      message: ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'),
      ...span(diagnostic.start, diagnostic.length) });
  }

  const inputNames = options.inputBindings ?? [];
  const localOptions = options.localBindings ?? [];
  const localNames = localOptions.map(binding => binding.name);
  const helperNames = options.helperBindings ?? [];
  const opaqueNames = options.opaqueBindings ?? [];
  const captureOptions = options.captureBindings ?? [];
  const captureNames = captureOptions.map(binding => binding.name);
  const serviceNames = options.serviceBindings ?? [];
  const injectedNames = new Set([...inputNames, ...localNames, ...helperNames, ...opaqueNames, ...captureNames, ...serviceNames]);
  const redundantAliases: ScopeSourceSpan[] = [];
  const bindings: ScopeBinding[] = [];
  for (const statement of fn.body.statements) {
    if (ts.isVariableStatement(statement)) {
      const declarations = statement.declarationList.declarations;
      if (declarations.length === 1) {
        const declaration = declarations[0]!;
        if (ts.isIdentifier(declaration.name) && declaration.initializer &&
            ts.isIdentifier(declaration.initializer) && declaration.name.text === declaration.initializer.text &&
            injectedNames.has(declaration.name.text)) {
          const location = span(statement);
          redundantAliases.push(location);
          repairs.push({ code: 'invalid-binding',
            message: `Removed redundant self-alias for injected binding ${JSON.stringify(declaration.name.text)}.`,
            ...location });
          continue;
        }
      }
      const flags = statement.declarationList.flags;
      const kind: ScopeBinding['kind'] = flags & ts.NodeFlags.Const ? 'const' : flags & ts.NodeFlags.Let ? 'let' : 'var';
      for (const declaration of statement.declarationList.declarations) for (const name of namesOf(declaration.name)) {
        let initial = declaration.initializer;
        while (initial && (ts.isAwaitExpression(initial) || ts.isParenthesizedExpression(initial) || ts.isPropertyAccessExpression(initial))) initial = initial.expression;
        const transient = !!(initial && ((ts.isArrowFunction(initial) || ts.isFunctionExpression(initial)) ||
          (ts.isCallExpression(initial) &&
          ((options.allowModules && initial.expression.kind === ts.SyntaxKind.ImportKeyword) ||
           (options.allowNetwork && ts.isIdentifier(initial.expression) && initial.expression.text === 'fetch'))) ||
          false));
        bindings.push({ name: name.text, kind, mutable: kind !== 'const',
          ...(transient ? { transient: true } : {}),
          ...(declaration.type && portableAnnotation(declaration.type, file) ?
            { annotation: portableAnnotation(declaration.type, file) } : {}),
          ...(declaration.initializer ? { initializer: declaration.initializer.getText(file) } : {}),
          start: span(name).start, end: span(name).end });
      }
    } else if (ts.isFunctionDeclaration(statement) && statement.name) {
      bindings.push({ name: statement.name.text, kind: 'function', mutable: false, transient: true,
        ...(statement.type && portableAnnotation(statement.type, file) ?
          { annotation: portableAnnotation(statement.type, file) } : {}),
        start: span(statement.name).start, end: span(statement.name).end });
    }
  }
  // Each eval is a new lexical transaction. A top-level declaration may replace
  // a local from an earlier eval without reusing that earlier const/let binding.
  const shadowedLocals = new Set(bindings.map(binding => binding.name).filter(name => localNames.includes(name)));

  const immutable = new Set([...inputNames, ...helperNames, ...opaqueNames, ...serviceNames,
    ...captureOptions.filter(binding => !binding.mutable).map(binding => binding.name),
    ...localOptions.filter(binding => !binding.mutable && !shadowedLocals.has(binding.name)).map(binding => binding.name),
    ...bindings.filter(binding => !binding.mutable).map(binding => binding.name)]);
  const deeplyReadonly = new Set([...inputNames, ...helperNames, ...opaqueNames, ...serviceNames]);

  const topLevelNames = new Set<string>();
  for (const binding of bindings) {
    if (topLevelNames.has(binding.name))
      diagnostics.push({ code: 'invalid-binding', message: `Top-level binding ${JSON.stringify(binding.name)} is declared more than once.`,
        ...rawSpan(binding.start, binding.end) });
    topLevelNames.add(binding.name);
    if (binding.kind === 'var')
      diagnostics.push({ code: 'forbidden-control', message: 'Persistent eval bindings must use const or let, not var.',
        ...rawSpan(binding.start, binding.end) });
  }

  const visit = (node: ts.Node): void => {
    if (ts.isImportDeclaration(node) || ts.isImportEqualsDeclaration(node) || ts.isExportDeclaration(node) ||
        ts.isExportAssignment(node) || ts.isMetaProperty(node))
      add('forbidden-dynamic-code', 'Modules are resolved by the codebase loader; imports and exports are unavailable in eval.', node);
    else if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword && !options.allowModules)
      add('forbidden-dynamic-code', 'Dynamic import is unavailable in eval.', node);
    else if (ts.isIdentifier(node) && (node.text === 'eval' || node.text === 'Function') &&
        !isPropertyName(node) && !isDeclarationName(node))
      add('forbidden-dynamic-code', `${node.text} is unavailable in eval.`, node);
    else if (ts.isIdentifier(node) && (MODULE_AMBIENTS.has(node.text) || (node.text === 'fetch' && !options.allowNetwork)) &&
        !topLevelNames.has(node.text) && !isPropertyName(node) && !isDeclarationName(node))
      add('forbidden-ambient', node.text === 'fetch' ? 'Network access is turned off for this runtime, so fetch is unavailable.' :
        `${node.text} is unavailable in eval; import packages with import instead.`, node);

    const target = assignmentTarget(node);
    const targetProperty = target && propertyText(target);
    if (targetProperty === 'prototype' || targetProperty === '__proto__')
      add('forbidden-prototype-mutation', `Mutation of ${targetProperty} is unavailable in eval.`, node);
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) &&
        ((node.expression.expression.getText(file) === 'Object' || node.expression.expression.getText(file) === 'Reflect') &&
          ['setPrototypeOf', 'defineProperty', 'defineProperties'].includes(node.expression.name.text)))
      add('forbidden-prototype-mutation', 'Prototype and property-descriptor mutation is unavailable in eval.', node);
    if (target) {
      let root = target;
      while (ts.isPropertyAccessExpression(root) || ts.isElementAccessExpression(root)) root = root.expression;
      if (ts.isIdentifier(root) && (deeplyReadonly.has(root.text) || (root === target && immutable.has(root.text))))
        add('invalid-binding', inputNames.includes(root.text) ?
          `${root.text} is a parameter and cannot be changed; declare a new variable for a changed value.` :
          `Binding ${JSON.stringify(root.text)} is immutable in eval.`, target);
    }
    ts.forEachChild(node, visit);
  };
  for (const statement of fn.body.statements) visit(statement);

  // Constrained-source policy: finite iteration and no recursion among functions authored here.
  const toRaw = (item: NatlangDiagnostic): ScopeCompileDiagnostic => {
    const start = Math.max(0, item.start - PREFIX.length), end = Math.max(start, item.end - PREFIX.length);
    return { ...rawSpan(start, end), code: item.code, message: item.message };
  };
  for (const item of checkConstrainedSource(file, { allowDynamicImport: true }))
    if (item.code !== 'forbidden-dynamic-code') diagnostics.push(toRaw(item));
  const authored = authoredCallables(file, options.guardPrefix ?? 'eval').filter(callable => callable.node !== fn);
  for (const item of findRecursion(authored, lexicalResolver(authored))) diagnostics.push(toRaw(item));

  const injected = [...inputNames, ...localNames, ...helperNames, ...opaqueNames, ...captureNames, ...serviceNames];
  const seen = new Set<string>();
  for (const name of injected) {
    if (!IDENTIFIER.test(name) || name === ENTRYPOINT || name === '__inputs' || name === '__locals' ||
        name === '__captures' || name.startsWith('__natlang_')) {
      diagnostics.push({ code: 'invalid-binding', message: `Invalid injected binding ${JSON.stringify(name)}.`,
        start: 0, end: 0, line: 1, column: 1 });
    } else if (seen.has(name)) {
      diagnostics.push({ code: 'invalid-binding', message: `Injected binding ${JSON.stringify(name)} is duplicated.`,
        start: 0, end: 0, line: 1, column: 1 });
    }
    seen.add(name);
  }
  const declared = new Set(bindings.map(binding => binding.name));
  for (const name of injected) if (declared.has(name) && !shadowedLocals.has(name))
    {
      const binding = bindings.find(item => item.name === name)!;
      diagnostics.push({ code: 'invalid-binding', message: `${name} is already defined in this scope; use it directly instead of declaring it again.`,
        ...rawSpan(binding.start, binding.end) });
    }

  const statements = [...fn.body.statements];
  let finalExpression: ScopeSourceSpan | undefined;
  const last = statements.at(-1);
  if (last && ts.isExpressionStatement(last)) finalExpression = span(last.expression);

  // Inline `nl` analysis, only when the snippet mentions it.
  let plans: InlineLambdaPlan[] = [];
  if (options.analyze && /\bnl\s*(?:<[^`]*>)?\s*`/.test(source)) {
    const analysis = options.analyze(source);
    plans = analysis.plans;
    for (const item of analysis.diagnostics) diagnostics.push({ ...rawSpan(item.start, item.end), code: item.code, message: item.message });
  }

  // Lowering edits (snippet-relative). Container edits (returns, final expression) lower their contents recursively.
  type Edit = { start: number; end: number; text: string };
  const primitive: Edit[] = [];
  const rel = (node: ts.Node) => ({ start: node.getStart(file) - PREFIX.length, end: node.getEnd() - PREFIX.length });
  const planAt = new Map(plans.map((plan, index) => [`${plan.sourceSpan.start}:${plan.sourceSpan.end}`, index]));
  const lowerNodes = (node: ts.Node): void => {
    if (ts.isTaggedTemplateExpression(node)) {
      const at = rel(node), index = planAt.get(`${at.start}:${at.end}`);
      if (index !== undefined) {
        const plan = plans[index]!;
        const values = ts.isTemplateExpression(node.template) ?
          node.template.templateSpans.map(item => lowerSpan(rel(item.expression).start, rel(item.expression).end)) : [];
        const accessors = plan.captures.map(capture => capture.mutable && !immutable.has(capture.name) ?
          `${capture.name}: [() => ${capture.name}, (__v: any) => { ${capture.name} = __v; }]` : `${capture.name}: [() => ${capture.name}]`);
        primitive.push({ ...at, text: `__natlang_inline(${index}, [${values.join(', ')}], { ${accessors.join(', ')} })` });
        return;
      }
    }
    if (ts.isForOfStatement(node) && !node.awaitModifier) {
      const at = rel(node.expression);
      primitive.push({ start: at.start, end: at.start, text: '__natlang_finite(' }, { start: at.end, end: at.end, text: ')' });
    }
    if ((ts.isFunctionDeclaration(node) || ts.isFunctionExpression(node) || ts.isArrowFunction(node) || ts.isMethodDeclaration(node)) &&
        node.body && node !== fn) {
      const callable = authored.find(item => item.node === node);
      if (callable && (ts.isFunctionDeclaration(node) || (node.parent && ts.isVariableDeclaration(node.parent)))) {
        const id = JSON.stringify(callable.id);
        const isAsync = !!node.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.AsyncKeyword);
        const body = rel(node.body);
        if (ts.isBlock(node.body)) {
          primitive.push({ start: body.start + 1, end: body.start + 1, text: ` return __natlang_guard(${id}, ${isAsync ? 'async ' : ''}() => {` },
            { start: body.end - 1, end: body.end - 1, text: '}); ' });
        } else {
          primitive.push({ start: body.start, end: body.start, text: `__natlang_guard(${id}, ${isAsync ? 'async ' : ''}() => (` },
            { start: body.end, end: body.end, text: '))' });
        }
      }
    }
    ts.forEachChild(node, lowerNodes);
  };
  const lowerSpan = (start: number, end: number): string => {
    let text = source.slice(start, end);
    const inside = primitive.filter(edit => edit.start >= start && edit.end <= end);
    for (const edit of [...inside].sort((a, b) => b.start - a.start || b.end - a.end))
      text = text.slice(0, edit.start - start) + edit.text + text.slice(edit.end - start);
    return text;
  };
  for (const statement of statements) lowerNodes(statement);

  const returns: ts.ReturnStatement[] = [];
  const findReturns = (node: ts.Node): void => {
    if (ts.isFunctionLike(node)) return;
    if (ts.isReturnStatement(node)) { returns.push(node); return; }
    ts.forEachChild(node, findReturns);
  };
  for (const statement of statements) findReturns(statement);
  const persisted = [
    ...localOptions.filter(binding => !shadowedLocals.has(binding.name)).map(binding => binding.name),
    ...bindings.filter(binding => !binding.transient).map(binding => binding.name)];
  const capture = `{ ${persisted.join(', ')} }`;
  const containers: Edit[] = returns.map(statement => {
    const location = rel(statement);
    const expression = statement.expression ? lowerSpan(rel(statement.expression).start, rel(statement.expression).end) : 'null';
    const available = [
      ...localOptions.map(binding => binding.name),
      ...bindings.filter(binding => !binding.transient && binding.end < location.start).map(binding => binding.name)];
    return { ...location, text: `return __natlang_finish(${expression}, { ${available.join(', ')} }, true);` };
  });
  if (finalExpression) containers.push({ start: finalExpression.start, end: finalExpression.end,
    text: `return __natlang_finish(${lowerSpan(finalExpression.start, finalExpression.end)});` });
  for (const repair of redundantAliases) containers.push({ start: repair.start, end: repair.end, text: '' });
  const edits = [...containers, ...primitive.filter(edit => !containers.some(container =>
    edit.start >= container.start && edit.end <= container.end && container.text !== ''))];
  let body = source;
  for (const edit of edits.sort((a, b) => b.start - a.start || b.end - a.end))
    body = body.slice(0, edit.start) + edit.text + body.slice(edit.end);

  diagnostics.sort((a, b) => a.start - b.start || a.code.localeCompare(b.code));
  const producesResult = !!finalExpression || returns.length > 0;
  const result: ScopeCompileResult = { version: SCOPE_COMPILE_VERSION, ok: diagnostics.length === 0,
    producesResult,
    resultBindings: [...new Set(returns.flatMap(statement => statement.expression && ts.isIdentifier(statement.expression)
      ? [statement.expression.text] : []))],
    entrypoint: ENTRYPOINT, bindings, ...(finalExpression ? { finalExpression } : {}), diagnostics, repairs };
  if (diagnostics.length) return result;
  const mutableCaptures = captureOptions.filter(binding => binding.mutable).map(binding => binding.name);
  const prologue = [
    inputNames.length ? `const { ${inputNames.join(', ')} } = __natlang_frozen(__natlang_copy(__inputs));` : '',
    // Your own locals come back as mutable copies (only parameters are frozen); the eval commits them when it succeeds.
    ...localOptions.filter(binding => !shadowedLocals.has(binding.name)).map(binding => `${binding.mutable ? 'let' : 'const'} ${binding.name}` +
      `${binding.annotation ? `: ${binding.annotation}` : ''} = __natlang_copy(__locals.${binding.name});`),
    ...captureOptions.map(binding => `${binding.mutable ? 'let' : 'const'} ${binding.name} = __captures.${binding.name};`),
    ...helperNames.map(name => `const ${name} = __natlang_callable(${JSON.stringify(name)});`),
    ...serviceNames.map(name => `const ${name} = __live.services[${JSON.stringify(name)}];`),
    `const __natlang_present = (value: Record<string, unknown>) => Object.fromEntries(` +
      `Object.entries(value).filter(([, item]) => item !== undefined));`,
    `const __natlang_finish = (__natlang_result: unknown, __natlang_bindings: Record<string, unknown> = ${capture}, ` +
      `__natlang_returned = false) => __natlang_output({ result: __natlang_result === undefined ? null : __natlang_result, ` +
      `returned: __natlang_returned, ` +
      `bindings: __natlang_present(__natlang_bindings)${mutableCaptures.length ? `, captures: { ${mutableCaptures.join(', ')} }` : ''} });`,
  ].filter(Boolean).join('\n');
  const typescript = `async function ${ENTRYPOINT}(__inputs: Readonly<Record<string, unknown>>, ` +
    `__locals: Readonly<Record<string, unknown>>, ` +
    `__captures: Readonly<Record<string, unknown>>) {\n${prologue}\n${body}\n` +
    `return __natlang_finish(null);\n}\n`;
  const emitted = ts.transpileModule(typescript, { fileName: 'natlang-scope.ts', reportDiagnostics: true,
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None, isolatedModules: true,
      removeComments: false } });
  const emitErrors = (emitted.diagnostics ?? []).filter(item => item.category === ts.DiagnosticCategory.Error);
  if (emitErrors.length) {
    result.ok = false;
    for (const diagnostic of emitErrors) result.diagnostics.push({ code: 'typescript-syntax',
      message: ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'), start: 0, end: 0, line: 1, column: 1 });
    return result;
  }
  result.body = body;
  result.plans = plans;
  result.program = emitted.outputText;
  return result;
}
