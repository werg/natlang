import ts from 'typescript';

/** Stable front-end contract for model-authored scope eval snippets. */
export const SCOPE_COMPILE_VERSION = 1 as const;

export type ScopeBinding = {
  name: string;
  kind: 'const' | 'let' | 'var' | 'function';
  mutable: boolean;
  annotation?: string;
  initializer?: string;
  /** Module namespaces and HTTP response handles live only for this eval. */
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
    'forbidden-control' | 'forbidden-prototype-mutation' | 'invalid-binding';
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
  /** Standalone ES2022 program defining an async entrypoint returning { result, bindings }.
   * Its third argument is a host dispatcher `(name, positionalArgs) => value | Promise<value>`;
   * helper function objects never enter the portable scope snapshot.
   */
  program?: string;
};

const ENTRYPOINT = '__natlang_scope' as const;
const PREFIX = `async function ${ENTRYPOINT}(__inputs: Readonly<Record<string, unknown>>, __locals: Readonly<Record<string, unknown>>, __invoke: (name: string, args: unknown[]) => unknown) {\n`;
const SUFFIX = '\n}\n';
const IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;
const FORBIDDEN_AMBIENTS = new Set([
  'process', 'globalThis', 'require', 'module', 'Buffer',
  'window', 'document', 'navigator', 'location',
  'fetch', 'XMLHttpRequest', 'WebSocket', 'EventSource',
  'setTimeout', 'setInterval', 'queueMicrotask', 'console',
]);

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
  const injectedNames = new Set([...inputNames, ...localNames, ...helperNames, ...opaqueNames]);
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
        const transient = initial && ts.isCallExpression(initial) &&
          ((options.allowModules && initial.expression.kind === ts.SyntaxKind.ImportKeyword) ||
           (options.allowNetwork && ts.isIdentifier(initial.expression) && initial.expression.text === 'fetch'));
        bindings.push({ name: name.text, kind, mutable: kind !== 'const',
          ...(transient ? { transient: true } : {}),
          ...(declaration.type && portableAnnotation(declaration.type, file) ?
            { annotation: portableAnnotation(declaration.type, file) } : {}),
          ...(declaration.initializer ? { initializer: declaration.initializer.getText(file) } : {}),
          start: span(name).start, end: span(name).end });
      }
    } else if (ts.isFunctionDeclaration(statement) && statement.name) {
      bindings.push({ name: statement.name.text, kind: 'function', mutable: false,
        ...(statement.type && portableAnnotation(statement.type, file) ?
          { annotation: portableAnnotation(statement.type, file) } : {}),
        start: span(statement.name).start, end: span(statement.name).end });
      add('forbidden-control', 'Top-level function declarations cannot be persisted; use a local arrow expression within one eval.', statement);
    }
  }

  const immutable = new Set([...helperNames, ...opaqueNames,
    ...localOptions.filter(binding => !binding.mutable).map(binding => binding.name),
    ...bindings.filter(binding => !binding.mutable).map(binding => binding.name)]);
  const deeplyReadonly = new Set([...helperNames, ...opaqueNames]);

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
    else if (ts.isTryStatement(node))
      add('forbidden-control', 'Eval cannot catch runtime failures; let validation bubble to the caller.', node);
    else if (ts.isClassDeclaration(node) || ts.isClassExpression(node) || ts.isYieldExpression(node) ||
        (ts.isFunctionLike(node) && 'asteriskToken' in node && !!node.asteriskToken))
      add('forbidden-control', 'Classes, generators, and yield are unavailable in eval.', node);
    else if (ts.isIdentifier(node) && FORBIDDEN_AMBIENTS.has(node.text) &&
        !(node.text === 'fetch' && options.allowNetwork) &&
        !topLevelNames.has(node.text) &&
        !isPropertyName(node) && !isDeclarationName(node))
      add('forbidden-ambient', `${node.text} is not an injected eval binding.`, node);

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
        add('invalid-binding', `Binding ${JSON.stringify(root.text)} is immutable in eval.`, target);
    }
    ts.forEachChild(node, visit);
  };
  for (const statement of fn.body.statements) visit(statement);

  const injected = [...inputNames, ...localNames, ...helperNames, ...opaqueNames];
  const seen = new Set<string>();
  for (const name of injected) {
    if (!IDENTIFIER.test(name) || name === ENTRYPOINT || name === '__inputs' || name === '__locals' ||
        name === '__invoke' || name === '__natlang_finish' || name === '__natlang_result') {
      diagnostics.push({ code: 'invalid-binding', message: `Invalid injected binding ${JSON.stringify(name)}.`,
        start: 0, end: 0, line: 1, column: 1 });
    } else if (seen.has(name)) {
      diagnostics.push({ code: 'invalid-binding', message: `Injected binding ${JSON.stringify(name)} is duplicated.`,
        start: 0, end: 0, line: 1, column: 1 });
    }
    seen.add(name);
  }
  const declared = new Set(bindings.map(binding => binding.name));
  for (const name of injected) if (declared.has(name))
    {
      const binding = bindings.find(item => item.name === name)!;
      diagnostics.push({ code: 'invalid-binding', message: `Top-level binding ${JSON.stringify(name)} redeclares an injected binding.`,
        ...rawSpan(binding.start, binding.end) });
    }

  const statements = [...fn.body.statements];
  let finalExpression: ScopeSourceSpan | undefined;
  let body = source;
  const last = statements.at(-1);
  if (last && ts.isExpressionStatement(last)) {
    finalExpression = span(last.expression);
  }

  const returns: ts.ReturnStatement[] = [];
  const findReturns = (node: ts.Node): void => {
    if (ts.isFunctionLike(node)) return;
    if (ts.isReturnStatement(node)) { returns.push(node); return; }
    ts.forEachChild(node, findReturns);
  };
  for (const statement of statements) findReturns(statement);
  const captures = [...localOptions.filter(binding => binding.mutable).map(binding => binding.name),
    ...bindings.filter(binding => !binding.transient).map(binding => binding.name)];
  const capture = `{ ${captures.join(', ')} }`;
  const edits: { start: number; end: number; text: string }[] = returns.map(statement => {
    const location = span(statement);
    const expression = statement.expression ? statement.expression.getText(file) : 'null';
    const available = [...localOptions.filter(binding => binding.mutable).map(binding => binding.name),
      ...bindings.filter(binding => !binding.transient && binding.end < location.start).map(binding => binding.name)];
    return { start: location.start, end: location.end,
      text: `return __natlang_finish(${expression}, { ${available.join(', ')} });` };
  });
  if (finalExpression) edits.push({ start: finalExpression.start, end: finalExpression.end,
    text: `return __natlang_finish(${source.slice(finalExpression.start, finalExpression.end)});` });
  for (const repair of redundantAliases) edits.push({ start: repair.start, end: repair.end, text: '' });
  for (const edit of edits.sort((a, b) => b.start - a.start))
    body = body.slice(0, edit.start) + edit.text + body.slice(edit.end);

  diagnostics.sort((a, b) => a.start - b.start || a.code.localeCompare(b.code));
  const producesResult = !!finalExpression || returns.length > 0;
  const result: ScopeCompileResult = { version: SCOPE_COMPILE_VERSION, ok: diagnostics.length === 0,
    producesResult,
    resultBindings: [...new Set(returns.flatMap(statement => statement.expression && ts.isIdentifier(statement.expression)
      ? [statement.expression.text] : []).concat(last && ts.isExpressionStatement(last) && ts.isIdentifier(last.expression)
      ? [last.expression.text] : []))],
    entrypoint: ENTRYPOINT, bindings, ...(finalExpression ? { finalExpression } : {}), diagnostics, repairs };
  if (diagnostics.length) return result;
  const prologue = [inputNames.length ? `let { ${inputNames.join(', ')} } = ` +
      `JSON.parse(JSON.stringify(__inputs));` : '',
    ...localOptions.map(binding => `${binding.mutable ? 'let' : 'const'} ${binding.name}` +
      `${binding.annotation ? `: ${binding.annotation}` : ''} = __locals.${binding.name};`),
    ...helperNames.map(name => `const ${name} = Object.assign((...args: unknown[]) => ` +
      `__invoke(${JSON.stringify(name)}, args), { __natlangFunction: ${JSON.stringify(name)} });`),
    `const __natlang_present = (value: Record<string, unknown>) => Object.fromEntries(` +
      `Object.entries(value).filter(([, item]) => item !== undefined));`,
    `const __natlang_finish = (__natlang_result: unknown, __natlang_bindings: Record<string, unknown> = ${capture}) => ` +
      `({ result: __natlang_result, inputs: __natlang_present({ ${inputNames.join(', ')} }), ` +
      `bindings: __natlang_present(__natlang_bindings) });`,
  ].filter(Boolean).join('\n');
  const typescript = `async function ${ENTRYPOINT}(__inputs: Readonly<Record<string, unknown>>, ` +
    `__locals: Readonly<Record<string, unknown>>, ` +
    `__invoke: (name: string, args: unknown[]) => unknown) {\n${prologue}\n${body}\n` +
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
  result.program = emitted.outputText;
  return result;
}
