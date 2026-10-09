import ts from 'typescript';
import { createNatlangCompilerHost } from './compiler/host.js';
import type { InlineLambdaPlan, InlineRebindSite, NatlangDiagnostic } from './compiler/inline.js';
import type { NeuraleseLiteral, NeuraleseReadout } from './compiler/neuralese.js';
import { authoredCallables, finiteCounterComparison, loopLabel, checkConstrainedSource, guardArguments, makesCalls } from './compiler/policy.js';

/** Stable front-end contract for model-authored scope eval snippets. */
export const SCOPE_COMPILE_VERSION = 3 as const;

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
  /** Source declarations from earlier successful evals, rechecked and recompiled in this eval. */
  persistentHelpers?: readonly PersistentScopeHelper[];
  /** Names of current scope values and built-ins a saved helper may resolve against. */
  persistentHelperNames?: readonly string[];
  /** Eval's retained type aliases, used to reject saved helpers that capture a transient type. */
  persistentTypeNames?: readonly string[];
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
  analyze?: (source: string) => { plans: InlineLambdaPlan[]; diagnostics: NatlangDiagnostic[]; neuralese?: NeuraleseLiteral[];
    readouts?: NeuraleseReadout[]; rebinds?: InlineRebindSite[] };
  /** Reads of fields a value's declared type does not have (snippet-relative spans), checked in every snippet. */
  checkFields?: (source: string) => NatlangDiagnostic[];
  /** The scope holds Neuralese values: analyze every snippet so their opacity is checked. */
  neuralese?: boolean;
  /** Prefix for runtime recursion-guard IDs of functions authored in this eval. */
  guardPrefix?: string;
};

export type ScopeExistingBinding = { name: string; mutable: boolean; annotation?: string };
export type PersistentScopeHelper = { name: string; source: string; freeNames: string[];
  typeNames: string[];
  sourceHash: string; declaredAction?: { toolCallId?: string; actionOrdinal: number; writtenCodeSha256: string };
  declarationSpan: ScopeSourceSpan };

export type ScopeCompileResult = {
  version: typeof SCOPE_COMPILE_VERSION;
  ok: boolean;
  entrypoint: '__natlang_scope';
  bindings: ScopeBinding[];
  /** Top-level function sources eligible to persist after this eval transaction succeeds. */
  persistentHelpers?: PersistentScopeHelper[];
  /** Helpers usable in this eval only because their source depends on an eval-local function. */
  transientHelpers?: { name: string; reason: string }[];
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
  /** Checked rebinding calls on inline tags created in this snippet. */
  rebinds?: InlineRebindSite[];
  /** Model-written Neuralese literals the snippet holds, typed by the analysis (graph `literal` nodes). */
  literals?: NeuraleseLiteral[];
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
// The public read intrinsic uses the same configured typed readout as text coercion.
const read = __live.readNeuralese;
const __natlang_output = (value: unknown) => { __live.finish(value); return null; };
const __natlang_thenable = (value: any) => !!value && (typeof value === 'object' || typeof value === 'function') &&
  typeof value.then === 'function';
const __natlang_pending = (value: any, depth = 0): boolean => __natlang_thenable(value) || depth < 4 &&
  (Array.isArray(value) || __natlang_plain(value)) && Object.values(value).some(item => __natlang_pending(item, depth + 1));
// What an eval leaves unawaited (a promise, or data holding promises) is awaited before it is kept, as \`await\` would.
const __natlang_settle = async (value: any, depth = 0): Promise<any> => {
  if (__natlang_thenable(value)) return __natlang_settle(await value, depth);
  if (depth >= 4 || !__natlang_pending(value)) return value;
  if (Array.isArray(value)) return Promise.all(value.map(item => __natlang_settle(item, depth + 1)));
  return Object.fromEntries(await Promise.all(Object.entries(value).map(async ([key, item]) =>
    [key, await __natlang_settle(item, depth + 1)])));
};
const __natlang_inline = (index: number, values: unknown[], accessors: unknown) => __live.inline(index, values, accessors);
const __natlang_finite = (source: any, label?: string) => __live.finite(source, label);
const __natlang_finiteArrayIterator = (source: any, method: 'entries' | 'keys' | 'values', label?: string) =>
  __live.finiteArrayIterator(source, method, label);
const __natlang_finiteAsync = (source: any, label?: string) => __live.finiteAsync(source, label);
const __natlang_guard = (id: string, fn: () => unknown, args?: unknown[]) => __live.guard(id, fn, args);
// A counted loop's bound is read once, when the loop starts; the counter must advance toward it every iteration.
const __natlang_counted = (upward: boolean) => {
  let bound: number | undefined, previous: number | undefined;
  // No globals here: this runs every iteration, and global lookups are slow in an eval context.
  const advance = (current: number) => {
    if (typeof current !== 'number' || current - current !== 0 ||
        (previous !== undefined && !(upward ? current > previous : current < previous)))
      throw new RangeError('the loop counter did not advance toward its bound');
    previous = current;
  };
  return {
    at: (current: number) => { if (bound === undefined) return undefined; advance(current); return bound; },
    fix: (current: number, value: unknown) => {
      if (typeof value !== 'number' || value - value !== 0)
        throw new RangeError('a counted for loop needs a finite number as its bound, but got ' + String(value) +
          '; use iterateOn(step, initial) for an open-ended loop');
      bound = value; advance(current); return value;
    },
  };
};
const iterateOn = __live.iterateOn;
// step.iterateOn(initial) is iterateOn(step, initial) for any function, as it is a method of natural-language ones.
const __natlang_iterate = (step: any, ...rest: any[]) =>
  typeof step?.iterateOn === 'function' ? step.iterateOn(...rest) : iterateOn(step, ...rest);
`;
const SUFFIX = '\n}\n';
const IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;
/** CommonJS names: eval code imports packages with `import` instead. */
const MODULE_AMBIENTS = new Set(['require', 'module']);

function namesOf(name: ts.BindingName): ts.Identifier[] {
  if (ts.isIdentifier(name)) return [name];
  return name.elements.flatMap(element => ts.isOmittedExpression(element) ? [] : namesOf(element.name));
}

/** Static type-only projections for destructured bindings; never run these expressions at runtime. */
function bindingInitializerProjections(name: ts.BindingName, initializer: string): Map<string, string> {
  const projections = new Map<string, string>();
  const file = ts.createSourceFile('initializer.ts', initializer, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const root = file.statements.find(ts.isExpressionStatement)?.expression;
  const staticScopePath = (node: ts.Expression): boolean => {
    if (ts.isIdentifier(node)) return true;
    if (ts.isPropertyAccessExpression(node)) return staticScopePath(node.expression);
    if (ts.isElementAccessExpression(node)) return !!node.argumentExpression &&
      (ts.isStringLiteral(node.argumentExpression) || ts.isNumericLiteral(node.argumentExpression)) &&
      staticScopePath(node.expression);
    return false;
  };
  // scopeInitializerType can follow these static paths. Calls, awaits, and other expressions
  // must be inferred from the extracted runtime value, since the whole call's return type is not
  // the type of a destructured field.
  if (!root || !staticScopePath(root)) return projections;
  const access = (base: string, key: string): string | undefined => {
    if (/^[A-Za-z_$][\w$]*$/.test(key)) return `${base}.${key}`;
    if (/^\d+$/.test(key)) return `${base}[${key}]`;
    return;
  };
  const propertyKey = (element: ts.BindingElement): string | undefined => {
    const property = element.propertyName;
    if (!property) return ts.isIdentifier(element.name) ? element.name.text : undefined;
    if (ts.isIdentifier(property) || ts.isStringLiteral(property) || ts.isNumericLiteral(property)) return property.text;
    if (ts.isComputedPropertyName(property) && (ts.isStringLiteral(property.expression) || ts.isNumericLiteral(property.expression)))
      return property.expression.text;
    return;
  };
  const visit = (pattern: ts.BindingName, base: string): void => {
    if (ts.isIdentifier(pattern)) { projections.set(pattern.text, base); return; }
    if (ts.isObjectBindingPattern(pattern)) {
      for (const element of pattern.elements) {
        if (element.dotDotDotToken) continue; // Object rest is inferred from its actual extracted value.
        if (element.initializer) continue; // A default may supply a different type when the projected field is absent.
        const key = propertyKey(element), child = key === undefined ? undefined : access(base, key);
        if (child) visit(element.name, child);
      }
      return;
    }
    let index = 0;
    for (const element of pattern.elements) {
      if (ts.isOmittedExpression(element)) { index++; continue; }
      if (element.dotDotDotToken) {
        if (ts.isIdentifier(element.name)) projections.set(element.name.text, `${base}.slice(${index})`);
        continue;
      }
      if (element.initializer) { index++; continue; }
      visit(element.name, `${base}[${index}]`);
      index++;
    }
  };
  visit(name, initializer);
  return projections;
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
    if (name === 'Is' && args.length === 2 && ts.isLiteralTypeNode(args[1]!) && ts.isStringLiteral(args[1]!.literal)) {
      const base = portableAnnotation(args[0]!, file); return base ? `Is<${base}, ${args[1]!.literal.getText(file)}>` : undefined;
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
    // `typeof task.passes`: the right side of a qualified name in a type query is a property, not a variable.
    (ts.isQualifiedName(parent) && parent.right === node) ||
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

const PERSISTENT_HELPER_TYPES = new Set(['any', 'unknown', 'never', 'void', 'string', 'number', 'boolean', 'bigint', 'symbol',
  'object', 'Function', 'Array', 'ReadonlyArray', 'Promise', 'PromiseLike', 'Record', 'Partial', 'Required', 'Readonly',
  'Pick', 'Omit', 'Exclude', 'Extract', 'NonNullable', 'ReturnType', 'Parameters', 'Awaited', 'Neuralese', 'FileHandle',
  'Folder', 'FolderSnapshot', 'FolderProposal', 'Entry', 'Date', 'Map', 'Set', 'WeakMap', 'WeakSet', 'RegExp', 'Error',
  'Uint8Array', 'Uint16Array', 'Uint32Array', 'Int8Array', 'Int16Array', 'Int32Array', 'Float32Array', 'Float64Array',
  'ArrayBuffer', 'Promise']);

/** Free value names from checker-resolved lexical scopes, including parameter defaults and nested scopes. */
function persistentHelperFreeNames(fn: ts.FunctionDeclaration | ts.ArrowFunction | ts.FunctionExpression,
  checker: ts.TypeChecker, sourceFile: ts.SourceFile): { names: string[]; nested: boolean } {
  const names = new Set<string>();
  let nested = false;
  const insideHelper = (node: ts.Node): boolean => {
    for (let current: ts.Node | undefined = node; current; current = current.parent) if (current === fn) return true;
    return false;
  };
  const typePosition = (node: ts.Identifier): boolean => {
    for (let parent: ts.Node | undefined = node.parent; parent; parent = parent.parent) {
      if (ts.isTypeQueryNode(parent)) return false;
      if (ts.isTypeNode(parent)) return true;
      if (ts.isExpression(parent) || ts.isStatement(parent)) return false;
    }
    return false;
  };
  const visit = (node: ts.Node): void => {
    if (ts.isArrowFunction(node) || ts.isFunctionExpression(node) || ts.isFunctionDeclaration(node)) if (node !== fn) nested = true;
    if (ts.isIdentifier(node) && !isDeclarationName(node) && !isPropertyName(node) && !typePosition(node)) {
      const symbol = checker.getSymbolAtLocation(node);
      const declarations = symbol?.declarations ?? [];
      const lexical = declarations.some(declaration => declaration.getSourceFile() === sourceFile && insideHelper(declaration));
      const sameFileOuter = declarations.some(declaration => declaration.getSourceFile() === sourceFile && !insideHelper(declaration));
      // A resolved symbol without declarations is an intrinsic such as `undefined`: global.
      const global = symbol !== undefined && declarations.every(declaration => declaration.getSourceFile() !== sourceFile);
      if (!lexical && !global || sameFileOuter) names.add(node.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(fn);
  return { names: [...names].sort(), nested };
}

/**
 * A checker over one eval source and the standard library. The host works without `ts.sys` (browsers read the
 * embedded library declarations), and hands back this exact SourceFile, so declarations compare by identity.
 */
function checkerFor(sourceFile: ts.SourceFile): ts.TypeChecker {
  const compilerOptions: ts.CompilerOptions = { target: ts.ScriptTarget.ES2022, noEmit: true, skipLibCheck: true };
  const path = `/__natlang__/eval/${sourceFile.fileName.replace(/\\/g, '/').replace(/^\.?\//, '')}`;
  const host = createNatlangCompilerHost({ options: compilerOptions, virtual: new Map([[path, sourceFile.text]]), currentDirectory: '/' });
  const getSourceFile = host.getSourceFile.bind(host);
  host.getSourceFile = (fileName, languageVersion, onError, shouldCreateNewSourceFile) =>
    fileName === path ? sourceFile : getSourceFile(fileName, languageVersion, onError, shouldCreateNewSourceFile);
  return ts.createProgram([path], compilerOptions, host).getTypeChecker();
}

function persistentHelperTypeNames(node: ts.Node): string[] {
  const names = new Set<string>();
  const typeParameters = new Set<string>();
  const visit = (item: ts.Node): void => {
    if (ts.isTypeParameterDeclaration(item)) typeParameters.add(item.name.text);
    if (ts.isTypeReferenceNode(item) && ts.isIdentifier(item.typeName) &&
        !typeParameters.has(item.typeName.text) && !PERSISTENT_HELPER_TYPES.has(item.typeName.text)) names.add(item.typeName.text);
    ts.forEachChild(item, visit);
  };
  visit(node);
  return [...names].sort();
}

function helperFunction(statement: ts.Statement): { name: string; fn: ts.FunctionDeclaration | ts.ArrowFunction | ts.FunctionExpression } | undefined {
  if (ts.isFunctionDeclaration(statement) && statement.name)
    return { name: statement.name.text, fn: statement };
  if (!ts.isVariableStatement(statement) || statement.declarationList.declarations.length !== 1) return;
  const declaration = statement.declarationList.declarations[0]!;
  if (!ts.isIdentifier(declaration.name) || !declaration.initializer) return;
  let initializer = declaration.initializer;
  while (ts.isAwaitExpression(initializer) || ts.isParenthesizedExpression(initializer) || ts.isAsExpression(initializer) ||
      ts.isSatisfiesExpression(initializer)) initializer = initializer.expression;
  return ts.isArrowFunction(initializer) || ts.isFunctionExpression(initializer) ?
    { name: declaration.name.text, fn: initializer } : undefined;
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
  const snippetSource = source;
  const persistentHelpers = options.persistentHelpers ?? [];
  const helperPrefix = persistentHelpers.map(helper => helper.source).join('\n') + (persistentHelpers.length ? '\n' : '');
  const analysisSource = helperPrefix + snippetSource;
  const wrapped = PREFIX + analysisSource + SUFFIX;
  const file = ts.createSourceFile('natlang-scope.ts', wrapped, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  // `await nl`...`` awaits the function nl`...` creates, which is never what is meant: it runs the judgment now,
  // on the names its instructions mention and its interpolations, as `await nl`...`()` does.
  const userStart = PREFIX.length + helperPrefix.length;
  const uncalled = uncalledInlineFunctions(file).filter(end => end >= userStart);
  if (uncalled.length) {
    let called = snippetSource;
    for (const end of uncalled.sort((a, b) => b - a)) called = called.slice(0, end - userStart) + '()' + called.slice(end - userStart);
    return compileScopeSnippet(called, options);
  }
  const fn = file.statements.find(ts.isFunctionDeclaration);
  if (!fn?.body) throw new Error('internal scope compiler wrapper failure');
  const diagnostics: ScopeCompileDiagnostic[] = [];
  const repairs: ScopeCompileDiagnostic[] = [];

  const sourceFile = ts.createSourceFile('eval-source.ts', snippetSource, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  const span = (nodeOrStart: ts.Node | number, length?: number): ScopeSourceSpan => {
    const absolute = typeof nodeOrStart === 'number' ? nodeOrStart : nodeOrStart.getStart(file);
    if (absolute < userStart) {
      const inHelpers = absolute - PREFIX.length;
      let offset = 0;
      for (const helper of persistentHelpers) {
        if (inHelpers >= offset && inHelpers < offset + helper.source.length) return helper.declarationSpan;
        offset += helper.source.length + 1;
      }
      return rawSpan(0, 0);
    }
    const rawStart = Math.max(0, Math.min(source.length, absolute - userStart));
    const rawEnd = Math.max(rawStart, Math.min(source.length,
      typeof nodeOrStart === 'number' ? rawStart + (length ?? 1) : nodeOrStart.getEnd() - userStart));
    const location = sourceFile.getLineAndCharacterOfPosition(rawStart);
    return { start: rawStart, end: rawEnd, line: location.line + 1, column: location.character + 1 };
  };
  const rawSpan = (start: number, end = start): ScopeSourceSpan => {
    const location = sourceFile.getLineAndCharacterOfPosition(start);
    return { start, end, line: location.line + 1, column: location.character + 1 };
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
  const helperCandidates: { statement: ts.Statement; helper: NonNullable<ReturnType<typeof helperFunction>> }[] = [];
  for (const statement of fn.body.statements) {
    if (statement.getStart(file) < userStart) continue;
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
        const initializerText = declaration.initializer?.getText(file);
        const bindingInitializer = declaration.initializer && initializerText ? ts.isIdentifier(declaration.name) ? initializerText :
          bindingInitializerProjections(declaration.name, initializerText).get(name.text) : undefined;
        let initial = declaration.initializer;
        while (initial && (ts.isAwaitExpression(initial) || ts.isParenthesizedExpression(initial) || ts.isPropertyAccessExpression(initial))) initial = initial.expression;
        const transient = !!(initial && ((ts.isArrowFunction(initial) || ts.isFunctionExpression(initial)) ||
          (ts.isCallExpression(initial) &&
          ((options.allowModules && initial.expression.kind === ts.SyntaxKind.ImportKeyword) ||
           (options.allowNetwork && ts.isIdentifier(initial.expression) && initial.expression.text === 'fetch'))) ||
          false));
        bindings.push({ name: name.text, kind, mutable: kind !== 'const',
          ...(transient ? { transient: true } : {}),
          ...(ts.isIdentifier(declaration.name) && declaration.type && portableAnnotation(declaration.type, file) ?
            { annotation: portableAnnotation(declaration.type, file) } : {}),
          ...(bindingInitializer ? { initializer: bindingInitializer } : {}),
          start: span(name).start, end: span(name).end });
      }
    } else if (ts.isFunctionDeclaration(statement) && statement.name) {
      bindings.push({ name: statement.name.text, kind: 'function', mutable: false, transient: true,
        ...(statement.type && portableAnnotation(statement.type, file) ?
          { annotation: portableAnnotation(statement.type, file) } : {}),
          start: span(statement.name).start, end: span(statement.name).end });
    }
    const helper = helperFunction(statement);
    if (helper) helperCandidates.push({ statement, helper });
  }
  // Each eval is a new lexical transaction. A top-level declaration may replace
  // a local from an earlier eval without reusing that earlier const/let binding.
  const shadowedLocals = new Set(bindings.map(binding => binding.name).filter(name => localNames.includes(name)));

  const knownHelperNames = new Set([...inputNames, ...localNames, ...helperNames, ...opaqueNames, ...captureNames,
    ...serviceNames, ...bindings.map(binding => binding.name), ...(options.persistentHelperNames ?? []),
    // Eval built-ins present in every call, and the binding model-written Neuralese literals lower to.
    'nl', 'iterateOn', 'transcript', 'decide', '__neuralese']);
  const transientNames = new Set(bindings.filter(binding => binding.transient).map(binding => binding.name));
  const persistentTypeNames = new Set(options.persistentTypeNames ?? []);
  const helperChecker = helperCandidates.length ? checkerFor(file) : undefined;
  for (const helper of persistentHelpers) {
    const missing = helper.freeNames.filter(name => !knownHelperNames.has(name));
    const missingTypes = helper.typeNames.filter(name => !persistentTypeNames.has(name));
    if (missing.length || missingTypes.length) diagnostics.push({ ...helper.declarationSpan, code: 'invalid-binding',
      message: `Saved helper ${JSON.stringify(helper.name)} is unavailable in this scope because ${[...missing, ...missingTypes].join(', ')} is not visible.` });
  }
  const savedHelpers: PersistentScopeHelper[] = [];
  const transientHelpers: { name: string; reason: string }[] = [];
  for (const { statement, helper } of helperCandidates) {
    // A helper that calls itself refers to its own binding, which persists with it.
    const found = persistentHelperFreeNames(helper.fn, helperChecker!, file);
    const free = { ...found, names: found.names.filter(name => name !== helper.name) };
    const spanOfDeclaration = span(statement);
    const hidden = free.names.filter(name => !knownHelperNames.has(name) && !bindings.some(binding => binding.name === name));
    const ephemeral = free.names.filter(name => transientNames.has(name));
    const missingTypes = persistentHelperTypeNames(statement).filter(name => !persistentTypeNames.has(name));
    if (ephemeral.length) {
      const details = [ephemeral.length ? `it refers to eval-local function${ephemeral.length > 1 ? 's' : ''} ${ephemeral.join(', ')}` : '']
        .filter(Boolean).join(' and ');
      transientHelpers.push({ name: helper.name, reason: details });
      continue;
    }
    if (hidden.length || missingTypes.length) {
      diagnostics.push({ ...spanOfDeclaration, code: 'invalid-binding',
        message: `Saved helper ${JSON.stringify(helper.name)} cannot persist because it refers to unavailable ` +
          [...hidden, ...missingTypes].join(', ') + '; declare those names in the call scope first.' });
      continue;
    }
    const source = statement.getText(file);
    savedHelpers.push({ name: helper.name, source, freeNames: free.names,
      typeNames: persistentHelperTypeNames(statement), sourceHash: '', declarationSpan: spanOfDeclaration });
  }
  // Helpers may refer to another helper declared in the same successful transaction.
  const transactionHelperNames = new Set(savedHelpers.map(helper => helper.name));
  for (const helper of savedHelpers) {
    const unknown = helper.freeNames.filter(name => !knownHelperNames.has(name) && !transactionHelperNames.has(name));
    if (unknown.length) diagnostics.push({ ...helper.declarationSpan, code: 'invalid-binding',
      message: `Saved helper ${JSON.stringify(helper.name)} cannot persist because it refers to unavailable ${unknown.join(', ')}; declare those names in the call scope first.` });
  }

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
        `${node.text} is unavailable in eval: eval code works with this call's scope, not the host's files, processes or ` +
        'system (npm packages of the project can be imported; Node\'s own modules cannot).', node);

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
  const toRaw = (item: NatlangDiagnostic, wrappedOffsets = false): ScopeCompileDiagnostic => {
    const base = (wrappedOffsets ? PREFIX.length : 0) + helperPrefix.length;
    const start = item.start - base, end = item.end - base;
    if (start < 0) {
      const helperOffset = item.start - (wrappedOffsets ? PREFIX.length : 0);
      let offset = 0, helper: PersistentScopeHelper | undefined;
      for (const candidate of persistentHelpers) {
        if (helperOffset >= offset && helperOffset < offset + candidate.source.length) { helper = candidate; break; }
        offset += candidate.source.length + 1;
      }
      const location = helper?.declarationSpan ?? rawSpan(0, 0);
      return { ...location, code: item.code,
        message: `Saved helper ${JSON.stringify(helper?.name ?? '<unknown>')} declared at its recorded source span is no longer valid here: ${item.message}` };
    }
    return { ...rawSpan(start, Math.max(start, end)), code: item.code, message: item.message };
  };
  for (const item of checkConstrainedSource(file, { allowDynamicImport: true }))
    if (item.code !== 'forbidden-dynamic-code') diagnostics.push(toRaw(item, true));
  // Recursion is checked when it happens: a function may call itself on a smaller argument (runtime/context.ts guard).
  const authored = authoredCallables(file, options.guardPrefix ?? 'eval').filter(callable => callable.node !== fn);

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
      // An input redeclared is usually test data about to replace the caller's real values; say whose they are.
      const whose = options.inputBindings?.includes(name) ? `${name} is this call's input and already holds the caller's value` :
        `${name} is already defined in this scope`;
      diagnostics.push({ code: 'invalid-binding', message: `${whose}; use it directly instead of declaring it again.`,
        ...rawSpan(binding.start, binding.end) });
    }

  const statements = [...fn.body.statements];
  let finalExpression: ScopeSourceSpan | undefined;
  const last = statements.at(-1);
  if (last && ts.isExpressionStatement(last)) finalExpression = span(last.expression);

  // Inline `nl` analysis, only when the snippet mentions it.
  let plans: InlineLambdaPlan[] = [];
  const planCoordinates = new Map<InlineLambdaPlan, { start: number; end: number }>();
  // nl written as code is analyzed: besides the template sites, uses of nl as a value are reported with the form that
  // works. A mention in a file name ("x.nl") is not code.
  // Model-written Neuralese literals (`__neuralese("nz1_…")`) are typed by the analysis, and a scope holding soft values
  // is always analyzed so their opacity is checked.
  const literalCalls = /(?<![.\w$])__neuralese\(/.test(analysisSource);
  let literals: NeuraleseLiteral[] = [];
  let readouts: NeuraleseReadout[] = [];
  let rebinds: InlineRebindSite[] = [];
  if (literalCalls && !options.analyze) diagnostics.push({ ...rawSpan(0, snippetSource.length), code: 'neuralese-untyped-literal',
    message: 'Neuralese literals need the typed eval checker, which this scope does not have.' });
  if (options.analyze && (options.neuralese || literalCalls ||
      /(?<![.\w$])nl\s*[`<(.]|\btypeof\s+nl\b|\b(?:const|let|var|function|class)\s+nl\b/.test(analysisSource))) {
    const analysis = options.analyze(analysisSource);
    plans = analysis.plans;
    for (const plan of plans) planCoordinates.set(plan, { start: plan.sourceSpan.start, end: plan.sourceSpan.end });
    let helperOffset = 0;
    for (const helper of persistentHelpers) {
      const sourceFile = ts.createSourceFile(`saved-helper-${helper.name}.ts`, helper.source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
      const fileName = `saved-helper:${helper.name}@${helper.sourceHash}`;
      const localSpan = (item: { file: string; start: number; end: number; line: number; column: number }) => {
        const start = Math.max(0, item.start - helperOffset), end = Math.max(0, item.end - helperOffset);
        const location = sourceFile.getLineAndCharacterOfPosition(Math.min(sourceFile.text.length, start));
        return { file: fileName, start, end, line: location.line + 1, column: location.character + 1 };
      };
      for (const plan of plans) {
        const coordinate = planCoordinates.get(plan)!;
        if (coordinate.start < helperOffset || coordinate.end > helperOffset + helper.source.length) continue;
        plan.sourceBackedHelper = { name: helper.name, sourceHash: helper.sourceHash,
          ...(helper.declaredAction ? { declaredAction: helper.declaredAction } : {}), declarationSpan: helper.declarationSpan };
        plan.sourceSpan = localSpan(plan.sourceSpan);
        plan.templateSpan = localSpan(plan.templateSpan);
        plan.interpolations = plan.interpolations.map(item => ({ ...item, sourceSpan: localSpan(item.sourceSpan) }));
        plan.captures = plan.captures.map(capture => ({ ...capture, mentionSpan: Math.max(0, capture.mentionSpan - helperOffset) }));
      }
      helperOffset += helper.source.length + 1;
    }
    literals = analysis.neuralese ?? [];
    readouts = analysis.readouts ?? [];
    rebinds = analysis.rebinds ?? [];
    for (const item of analysis.diagnostics) diagnostics.push(toRaw(item));
  }
  if (options.checkFields) for (const item of options.checkFields(analysisSource)) diagnostics.push(toRaw(item));

  // Lowering edits (snippet-relative). Container edits (returns, final expression) lower their contents recursively.
  type Edit = { start: number; end: number; text: string; composed?: boolean };
  const primitive: Edit[] = [];
  let loops = 0;
  const rel = (node: ts.Node) => ({ start: node.getStart(file) - PREFIX.length, end: node.getEnd() - PREFIX.length });
  const planAt = new Map(plans.map((plan, index) => {
    const coordinate = planCoordinates.get(plan)!;
    return [`${coordinate.start}:${coordinate.end}`, index];
  }));
  const rebindAt = new Map(rebinds.map(site => [`${site.start}:${site.end}`, site]));
  for (const readout of readouts) if (!readout.kind) primitive.push({ start: readout.start, end: readout.end,
    text: `(await __live.${readout.conditional ? 'readNeuraleseIfReference' : 'readNeuralese'}((${analysisSource.slice(readout.start, readout.end)})))` });
  const joins = new Set(readouts.filter(readout => readout.kind === 'join').map(readout => `${readout.start}:${readout.end}`));
  const arrayStrings = new Set(readouts.filter(readout => readout.kind === 'array-string').map(readout => `${readout.start}:${readout.end}`));
  const concats = new Set(readouts.filter(readout => readout.kind === 'concat').map(readout => `${readout.start}:${readout.end}`));
  const jsonReadouts = new Map(readouts.filter(readout => readout.kind === 'json')
    .map(readout => [`${readout.start}:${readout.end}`, readout]));
  const errorReadouts = new Map(readouts.filter(readout => readout.kind === 'error')
    .map(readout => [`${readout.start}:${readout.end}`, readout]));
  const stringArguments = new Map(readouts.filter(readout => readout.kind === 'string-argument')
    .map(readout => [`${readout.start}:${readout.end}`, readout]));
  const scalarConversions = new Map(readouts.filter(readout => readout.kind === 'scalar-conversion' && readout.conversion)
    .map(readout => [`${readout.start}:${readout.end}`, readout]));
  const stringReplaces = new Map(readouts.filter(readout => readout.kind === 'string-replace')
    .map(readout => [`${readout.start}:${readout.end}`, readout]));
  const arrayMapReadouts = new Map(readouts.filter(readout => readout.kind === 'array-map')
    .map(readout => [`${readout.start}:${readout.end}`, readout]));
  const asyncMapCallbacks = new Set([...arrayMapReadouts.values()].flatMap(readout => readout.callback ?
    [`${readout.callback.start}:${readout.callback.end}`] : []));
  const lowerNodes = (node: ts.Node): void => {
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) &&
        arrayMapReadouts.has(`${rel(node).start}:${rel(node).end}`) && node.arguments[0]) {
      const readout = arrayMapReadouts.get(`${rel(node).start}:${rel(node).end}`)!;
      lowerNodes(node.expression.expression);
      for (const argument of node.arguments) lowerNodes(argument);
      if (readout.callback) primitive.push({ start: readout.callback.start,
        end: readout.callback.start, text: 'async ' });
      const receiver = lowerSpan(rel(node.expression.expression).start, rel(node.expression.expression).end);
      const args = node.arguments.map(argument => lowerSpan(rel(argument).start, rel(argument).end)).join(', ');
      const receiverName = `__natlang_map_receiver_${rel(node).start}`;
      const methodName = `__natlang_map_method_${rel(node).start}`;
      const argsName = `__natlang_map_args_${rel(node).start}`;
      primitive.push({ ...rel(node), composed: true,
        text: `(await ((${receiverName}: any) => { const ${methodName} = ${receiverName}.map; ` +
          `const ${argsName} = [${args}]; return __live.mapNeuraleseReadout(${receiverName}, ${methodName}, ${argsName}); })(${receiver}))` });
      return;
    }
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) &&
        stringReplaces.has(`${rel(node).start}:${rel(node).end}`)) {
      const readout = stringReplaces.get(`${rel(node).start}:${rel(node).end}`)!;
      lowerNodes(node.expression.expression);
      for (const argument of node.arguments) lowerNodes(argument);
      const receiver = lowerSpan(rel(node.expression.expression).start, rel(node.expression.expression).end);
      const args = node.arguments.map(argument => lowerSpan(rel(argument).start, rel(argument).end)).join(', ');
      const receiverName = `__natlang_replace_receiver_${rel(node).start}`;
      const textName = `__natlang_replace_text_${rel(node).start}`;
      const methodName = `__natlang_replace_method_${rel(node).start}`;
      const argsName = `__natlang_replace_args_${rel(node).start}`;
      const reader = readout.conditional ? 'readNeuraleseIfReference' : 'readNeuralese';
      primitive.push({ ...rel(node), text: `(await (async (${receiverName}: any) => { ` +
        `const ${textName} = await __live.${reader}(${receiverName}); ` +
        `const ${methodName} = ${textName}.replace; const ${argsName} = [${args}]; ` +
        `return { value: __live.invokeWithReceiver(${methodName}, ${textName}, ${argsName}) }; })(${receiver})).value` });
      return;
    }
    if (ts.isCallExpression(node) && scalarConversions.has(`${rel(node).start}:${rel(node).end}`) && node.arguments.length) {
      const readout = scalarConversions.get(`${rel(node).start}:${rel(node).end}`)!;
      for (const argument of node.arguments) lowerNodes(argument);
      lowerNodes(node.expression);
      const callee = lowerSpan(rel(node.expression).start, rel(node.expression).end);
      const args = node.arguments.map(argument => lowerSpan(rel(argument).start, rel(argument).end)).join(', ');
      const calleeName = `__natlang_scalar_constructor_${rel(node).start}`;
      const argsName = `__natlang_scalar_args_${rel(node).start}`;
      const index = readout.argument ?? 0;
      const reader = readout.conditional ? 'readNeuraleseIfReference' : 'readNeuralese';
      primitive.push({ ...rel(node), text: `(await ((${calleeName}: any) => (async (${argsName}: any[]) => { ` +
        `${argsName}[${index}] = await __live.${reader}(${argsName}[${index}]); ` +
        `return ${calleeName}(...${argsName}); })([${args}]))(${callee}))` });
      return;
    }
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) && node.expression.name.text === 'with') {
      const site = rebindAt.get(`${rel(node).start}:${rel(node).end}`);
      if (site && node.arguments[0]) {
        // Lower the record's contents first so nested typed readouts or callable rebinding keep their normal semantics.
        lowerNodes(node.expression.expression);
        for (const argument of node.arguments) lowerNodes(argument);
        primitive.push({ ...rel(node), composed: true, text: `__live.rebindInline(${lowerSpan(rel(node.expression.expression).start, rel(node.expression.expression).end)}, ` +
          `${lowerSpan(rel(node.arguments[0]).start, rel(node.arguments[0]).end)}, ${JSON.stringify(site.captureSources)})` });
        return;
      }
    }
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) &&
        stringArguments.has(`${rel(node).start}:${rel(node).end}`)) {
      const readout = stringArguments.get(`${rel(node).start}:${rel(node).end}`)!;
      for (const argument of node.arguments) lowerNodes(argument);
      lowerNodes(node.expression.expression);
      const receiver = lowerSpan(rel(node.expression.expression).start, rel(node.expression.expression).end);
      const args = node.arguments.map(argument => lowerSpan(rel(argument).start, rel(argument).end)).join(', ');
      const receiverName = `__natlang_text_receiver_${rel(node).start}`;
      const methodName = `__natlang_text_method_${rel(node).start}`;
      const argsName = `__natlang_text_args_${rel(node).start}`;
      const index = readout.argument ?? 0;
      const reader = readout.conditional ? 'readNeuraleseIfReference' : 'readNeuralese';
      primitive.push({ ...rel(node), text: `(await ((${receiverName}: any) => ((${methodName}: any) => (async (${argsName}: any[]) => { ` +
        `${argsName}[${index}] = await __live.${reader}(${argsName}[${index}]); ` +
        `return ${methodName}.call(${receiverName}, ...${argsName}); })([${args}]))(${receiverName}.${node.expression.name.text}))(${receiver}))` });
      return;
    }
    if ((ts.isCallExpression(node) || ts.isNewExpression(node)) && node.arguments?.length &&
        errorReadouts.has(`${rel(node).start}:${rel(node).end}`)) {
      const readout = errorReadouts.get(`${rel(node).start}:${rel(node).end}`)!;
      for (const argument of node.arguments) lowerNodes(argument);
      const callee = lowerSpan(rel(node.expression).start, rel(node.expression).end);
      const args = node.arguments.map(argument => lowerSpan(rel(argument).start, rel(argument).end)).join(', ');
      const calleeName = `__natlang_error_constructor_${rel(node).start}`;
      const argsName = `__natlang_error_args_${rel(node).start}`;
      const reader = readout.conditional ? 'readNeuraleseIfReference' : 'readNeuralese';
      const invoke = ts.isNewExpression(node) ? `new ${calleeName}(await __live.${reader}(${argsName}[0]), ...${argsName}.slice(1))` :
        `${calleeName}(await __live.${reader}(${argsName}[0]), ...${argsName}.slice(1))`;
      primitive.push({ ...rel(node), text: `(await ((${calleeName}: any) => (async (${argsName}: any[]) => ${invoke})` +
        `([${args}]))(${callee}))` });
      return;
    }
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) &&
        jsonReadouts.has(`${rel(node).start}:${rel(node).end}`)) {
      const readout = jsonReadouts.get(`${rel(node).start}:${rel(node).end}`)!;
      // Capture the callable and evaluate every argument left-to-right before awaiting the typed value's readout.
      // This preserves JSON.stringify's ordinary call boundary while letting its first value be a crisp union arm.
      for (const argument of node.arguments) lowerNodes(argument);
      const receiver = lowerSpan(rel(node.expression.expression).start, rel(node.expression.expression).end);
      const args = node.arguments.map(argument => lowerSpan(rel(argument).start, rel(argument).end));
      const reader = readout.conditional ? 'readNeuraleseIfReference' : 'readNeuralese';
      const fresh = (base: string): string => {
        let name = base, suffix = 0;
        while (analysisSource.includes(name)) name = `${base}_${++suffix}`;
        return name;
      };
      const receiverName = fresh(`__natlang_json_receiver_${rel(node).start}`);
      const callName = fresh(`__natlang_json_call_${rel(node).start}`);
      const argsName = fresh(`__natlang_json_args_${rel(node).start}`);
      primitive.push({ ...rel(node), text: `(await ((${receiverName}: any) => (async (${callName}: [any, any], ${argsName}: any[]) => ` +
        `${callName}[1].call(${callName}[0], await __live.${reader}(${argsName}[0])` +
        `${args.slice(1).map((_arg, index) => `, ${argsName}[${index + 1}]`).join('')}` +
        `))([${receiverName}, ${receiverName}.stringify], [${args.join(', ')}]))(${receiver}))` });
      return;
    }
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) &&
        concats.has(`${rel(node).start}:${rel(node).end}`)) {
      const receiver = lowerSpan(rel(node.expression.expression).start, rel(node.expression.expression).end);
      const values = node.arguments.map(argument => lowerSpan(rel(argument).start, rel(argument).end)).join(', ');
      const temp = `__natlang_concat_receiver_${rel(node).start}`;
      primitive.push({ ...rel(node), text: `await ((${temp}: any) => { const __method = ${temp}.concat; const __values = [${values}]; ` +
        `return __live.concatNeuralese(${temp}, __method, __values, async (__value: any) => ` +
        `await __live.readNeuralese(__value)); })(${receiver})` });
      return;
    }
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) &&
        arrayStrings.has(`${rel(node).start}:${rel(node).end}`)) {
      const receiver = rel(node.expression.expression);
      lowerNodes(node.expression.expression);
      for (const argument of node.arguments) lowerNodes(argument);
      const receiverName = `__natlang_array_string_receiver_${rel(node).start}`;
      const methodName = `__natlang_array_string_method_${rel(node).start}`;
      const argsName = `__natlang_array_string_args_${rel(node).start}`;
      const args = node.arguments.map(argument => lowerSpan(rel(argument).start, rel(argument).end)).join(', ');
      primitive.push({ ...rel(node), text: `(await ((${receiverName}: any) => (async (${methodName}: any, ${argsName}: any[]) => ` +
        `(await __live.arrayToStringNeuralese(${receiverName}, ${methodName}, ${argsName}, async (__natlang_array_string_value: any) => ` +
        `await __live.readNeuralese(__natlang_array_string_value))).value)(${receiverName}.toString, [${args}]))(${lowerSpan(receiver.start, receiver.end)}))` });
      return;
    }
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) &&
        joins.has(`${rel(node).start}:${rel(node).end}`)) {
      const receiver = rel(node.expression.expression);
      lowerNodes(node.expression.expression);
      for (const argument of node.arguments) lowerNodes(argument);
      const receiverName = `__natlang_join_receiver_${rel(node).start}`;
      const methodName = `__natlang_join_method_${rel(node).start}`;
      const argsName = `__natlang_join_args_${rel(node).start}`;
      const args = node.arguments.map(argument => lowerSpan(rel(argument).start, rel(argument).end)).join(', ');
      primitive.push({ ...rel(node), text: `(await ((${receiverName}: any) => (async (${methodName}: any, ${argsName}: any[]) => ` +
        `(await __live.joinNeuralese(${receiverName}, ${methodName}, ${argsName}, async (__natlang_join_value: any) => ` +
        `await __live.readNeuralese(__natlang_join_value))).value)(${receiverName}.join, [${args}]))(${lowerSpan(receiver.start, receiver.end)}))` });
      return;
    }
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) && node.expression.name.text === 'with' &&
        ts.isTaggedTemplateExpression(node.expression.expression)) {
      const tagged = node.expression.expression, at = rel(tagged), index = planAt.get(`${at.start}:${at.end}`);
      if (index !== undefined && plans[index]!.explicitCaptures) {
        const plan = plans[index]!, listing = node.arguments[0];
        const properties = listing && ts.isObjectLiteralExpression(listing) ? [...listing.properties] : [];
        const recordTemp = listing && !ts.isObjectLiteralExpression(listing) ? `__natlang_capture_record_${index}_${at.start}` : undefined;
        const accessors = plan.captures.map(capture => {
          const property = properties.find(item => (ts.isShorthandPropertyAssignment(item) || ts.isPropertyAssignment(item)) &&
            (ts.isIdentifier(item.name) || ts.isStringLiteral(item.name)) && item.name.text === capture.name);
          if (capture.mode === 'live') return immutable.has(capture.name) ? `${capture.name}: [() => ${capture.name}]` :
            `${capture.name}: [() => ${capture.name}, (__v: any) => { ${capture.name} = __v; }]`;
          const value = recordTemp ? `${recordTemp}[${JSON.stringify(capture.name)}]` : !property ? 'undefined' : ts.isShorthandPropertyAssignment(property) ? capture.name :
            lowerSpan(rel((property as ts.PropertyAssignment).initializer).start, rel((property as ts.PropertyAssignment).initializer).end);
          return `${capture.name}: [() => (${value})]`;
        });
        const values = !plan.softBody && ts.isTemplateExpression(tagged.template) ?
          tagged.template.templateSpans.map(item => lowerSpan(rel(item.expression).start, rel(item.expression).end)) : [];
        const accessorObject = recordTemp && listing ?
          `((${recordTemp}: any) => ({ ${accessors.join(', ')} }))(${lowerSpan(rel(listing).start, rel(listing).end)})` :
          `{ ${accessors.join(', ')} }`;
        const text = `__natlang_inline(${index}, [${values.join(', ')}], ${accessorObject})`;
        primitive.push({ start: at.start, end: rel(node).end, text });
        return;
      }
    }
    if (ts.isTaggedTemplateExpression(node)) {
      const at = rel(node), index = planAt.get(`${at.start}:${at.end}`);
      if (index !== undefined && plans[index]!.explicitCaptures) {
        // nl.with({ a, b: expr, n: live(n) }): snapshot getters are read once, when the function is created.
        const plan = plans[index]!;
        const withCall = ts.isCallExpression(node.tag) ? node.tag : undefined;
        const listing = withCall?.arguments[0];
        const properties = listing && ts.isObjectLiteralExpression(listing) ? [...listing.properties] : [];
        const recordTemp = listing && !ts.isObjectLiteralExpression(listing) ? `__natlang_capture_record_${index}_${at.start}` : undefined;
        const accessors = plan.captures.map(capture => {
          const property = properties.find(item => (ts.isShorthandPropertyAssignment(item) || ts.isPropertyAssignment(item)) &&
            (ts.isIdentifier(item.name) || ts.isStringLiteral(item.name)) && item.name.text === capture.name);
          if (capture.mode === 'live') return immutable.has(capture.name) ? `${capture.name}: [() => ${capture.name}]` :
            `${capture.name}: [() => ${capture.name}, (__v: any) => { ${capture.name} = __v; }]`;
          const value = recordTemp ? `${recordTemp}[${JSON.stringify(capture.name)}]` : !property ? 'undefined' : ts.isShorthandPropertyAssignment(property) ? capture.name :
            lowerSpan(rel((property as ts.PropertyAssignment).initializer).start, rel((property as ts.PropertyAssignment).initializer).end);
          return `${capture.name}: [() => (${value})]`;
        });
        const values = !plan.softBody && ts.isTemplateExpression(node.template) ?
          node.template.templateSpans.map(item => lowerSpan(rel(item.expression).start, rel(item.expression).end)) : [];
        const accessorObject = recordTemp && listing ?
          `((${recordTemp}: any) => ({ ${accessors.join(', ')} }))(${lowerSpan(rel(listing).start, rel(listing).end)})` :
          `{ ${accessors.join(', ')} }`;
        const text = `__natlang_inline(${index}, [${values.join(', ')}], ${accessorObject})`;
        primitive.push({ ...rel(node), text });
        return;
      }
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
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === '__neuralese') {
      // A typed literal becomes its reference value; an untyped one was reported by the analysis.
      const at = rel(node), literal = literals.find(item => item.start === at.start && item.end === at.end);
      if (literal) {
        primitive.push({ ...at, text: `({ $neuralese: { type: ${JSON.stringify(literal.type)}, id: ${JSON.stringify(literal.id)} } })` });
        return;
      }
    }
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) && node.expression.name.text === 'iterateOn' &&
        !ts.isTaggedTemplateExpression(node.expression.expression)) {
      // From the end of the receiver through the call's opening parenthesis (type arguments included).
      const receiver = rel(node.expression.expression), argumentsStart = node.arguments.pos - PREFIX.length;
      primitive.push({ start: receiver.start, end: receiver.start, text: '__natlang_iterate(' },
        { start: receiver.end, end: argumentsStart, text: node.arguments.length ? ', ' : '' });
    }
    if (ts.isForOfStatement(node)) {
      const at = rel(node.expression);
      let iterableExpression = node.expression;
      while (ts.isParenthesizedExpression(iterableExpression) || ts.isAsExpression(iterableExpression) ||
          ts.isTypeAssertionExpression(iterableExpression) || ts.isNonNullExpression(iterableExpression))
        iterableExpression = iterableExpression.expression;
      if (!node.awaitModifier && ts.isCallExpression(iterableExpression) && iterableExpression.arguments.length === 0 &&
          ts.isPropertyAccessExpression(iterableExpression.expression) &&
          ['entries', 'keys', 'values'].includes(iterableExpression.expression.name.text)) {
        const method = iterableExpression.expression.name.text as 'entries' | 'keys' | 'values';
        const receiver = iterableExpression.expression.expression;
        lowerNodes(receiver);
        primitive.push({ ...at, text: `__natlang_finiteArrayIterator(${lowerSpan(rel(receiver).start, rel(receiver).end)}, ` +
          `${JSON.stringify(method)}, ${JSON.stringify(loopLabel(analysisSource.slice(at.start, at.end)))})` });
        lowerNodes(node.statement);
        return;
      }
      primitive.push({ start: at.start, end: at.start, text: node.awaitModifier ? '__natlang_finiteAsync(' : '__natlang_finite(' },
        { start: at.end, end: at.end, text: `, ${JSON.stringify(loopLabel(analysisSource.slice(at.start, at.end)))})` });
    }
    // A counted loop reads its bound once, when the loop starts, and checks that the counter advances toward it.
    if (ts.isForStatement(node) && node.initializer && ts.isVariableDeclarationList(node.initializer) &&
        node.initializer.declarations.length === 1 && node.condition) {
      const declaration = node.initializer.declarations[0]!;
      const boundedCondition = ts.isIdentifier(declaration.name) ?
        finiteCounterComparison(node.condition, declaration.name.text) : undefined;
      if (ts.isIdentifier(declaration.name) && boundedCondition) {
        const counter = declaration.name.text;
        const comparison = boundedCondition.comparison;
        const left = ts.isIdentifier(comparison.left) && comparison.left.text === counter;
        const kind = comparison.operatorToken.kind;
        const upward = left ? kind === ts.SyntaxKind.LessThanToken || kind === ts.SyntaxKind.LessThanEqualsToken :
          kind === ts.SyntaxKind.GreaterThanToken || kind === ts.SyntaxKind.GreaterThanEqualsToken;
        const loop = `__natlang_loop_${++loops}`, bound = rel(left ? comparison.right : comparison.left);
        primitive.push({ start: rel(declaration).end, end: rel(declaration).end, text: `, ${loop} = __natlang_counted(${upward})` },
          { start: bound.start, end: bound.start, text: `(${loop}.at(${counter}) ?? ${loop}.fix(${counter}, ` },
          { start: bound.end, end: bound.end, text: '))' });
      }
    }
    if ((ts.isFunctionDeclaration(node) || ts.isFunctionExpression(node) || ts.isArrowFunction(node) || ts.isMethodDeclaration(node) ||
        ts.isGetAccessorDeclaration(node) || ts.isSetAccessorDeclaration(node)) && node.body && node !== fn && makesCalls(node)) {
      const callable = authored.find(item => item.node === node);
      if (callable) {
        const id = JSON.stringify(callable.id);
        const isAsync = !!node.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.AsyncKeyword) ||
          asyncMapCallbacks.has(`${rel(node).start}:${rel(node).end}`);
        const body = rel(node.body), args = `[${guardArguments(node).join(', ')}]`;
        // The openings replace a token so that edits starting at the same place stay inside the guard.
        if (ts.isBlock(node.body)) {
          primitive.push({ start: body.start, end: body.start + 1, text: `{ return __natlang_guard(${id}, ${isAsync ? 'async ' : ''}() => {` },
            { start: body.end - 1, end: body.end, text: `}, ${args}); }` });
        } else {
          const arrow = rel((node as ts.ArrowFunction).equalsGreaterThanToken);
          primitive.push({ start: arrow.start, end: arrow.end, text: `=> __natlang_guard(${id}, ${isAsync ? 'async ' : ''}() => (` },
            { start: body.end, end: body.end, text: `), ${args})` });
        }
      }
    }
    ts.forEachChild(node, lowerNodes);
  };
  const lowerSpan = (start: number, end: number): string => {
    let text = analysisSource.slice(start, end);
    const contained = primitive.filter(edit => edit.start >= start && edit.end <= end);
    const inside = contained.filter(edit => !contained.some(parent => parent.composed && parent !== edit &&
      parent.start <= edit.start && parent.end >= edit.end && (parent.start < edit.start || parent.end > edit.end)));
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
      ...bindings.filter(binding => !binding.transient && binding.end + helperPrefix.length < location.start).map(binding => binding.name)];
    return { ...location, text: `return __natlang_finish((${expression}), { ${available.join(', ')} }, true);` };
  });
  if (finalExpression) {
    const start = finalExpression.start + helperPrefix.length, end = finalExpression.end + helperPrefix.length;
    containers.push({ start, end, text: `return __natlang_finish((${lowerSpan(start, end)}));` });
  }
  for (const repair of redundantAliases) containers.push({ start: repair.start + helperPrefix.length,
    end: repair.end + helperPrefix.length, text: '' });
  const topPrimitive = primitive.filter(edit => !primitive.some(parent => parent.composed && parent !== edit &&
    parent.start <= edit.start && parent.end >= edit.end && (parent.start < edit.start || parent.end > edit.end)));
  const edits = [...containers, ...topPrimitive.filter(edit => !containers.some(container =>
    edit.start >= container.start && edit.end <= container.end && container.text !== ''))];
  let body = analysisSource;
  for (const edit of edits.sort((a, b) => b.start - a.start || b.end - a.end))
    body = body.slice(0, edit.start) + edit.text + body.slice(edit.end);

  diagnostics.sort((a, b) => a.start - b.start || a.code.localeCompare(b.code));
  const producesResult = !!finalExpression || returns.length > 0;
  const result: ScopeCompileResult = { version: SCOPE_COMPILE_VERSION, ok: diagnostics.length === 0,
    producesResult,
    resultBindings: [...new Set(returns.flatMap(statement => statement.expression && ts.isIdentifier(statement.expression)
      ? [statement.expression.text] : []))],
    entrypoint: ENTRYPOINT, bindings, ...(savedHelpers.length ? { persistentHelpers: savedHelpers } : {}),
    ...(transientHelpers.length ? { transientHelpers } : {}), ...(finalExpression ? { finalExpression } : {}), diagnostics, repairs,
    ...(literals.length ? { literals } : {}) };
  if (diagnostics.length) return result;
  const mutableCaptures = captureOptions.filter(binding => binding.mutable).map(binding => binding.name);
  const prologue = [
    inputNames.length ? `const { ${inputNames.join(', ')} } = __natlang_frozen(__natlang_copy(__inputs));` : '',
    // Your own locals come back as mutable copies (only parameters are frozen); the eval commits them when it succeeds.
    ...localOptions.filter(binding => !shadowedLocals.has(binding.name)).map(binding => `${binding.mutable ? 'let' : 'const'} ${binding.name}` +
      `${binding.annotation ? `: ${binding.annotation}` : ''} = __natlang_copy(__locals.${binding.name});`),
    ...localOptions.filter(binding => binding.mutable && !shadowedLocals.has(binding.name)).map(binding =>
      `__live.bindLocal?.(${JSON.stringify(binding.name)}, () => ${binding.name}, (__v: any) => { ${binding.name} = __v; });`),
    ...captureOptions.map(binding => `${binding.mutable ? 'let' : 'const'} ${binding.name} = __captures.${binding.name};`),
    ...helperNames.map(name => `const ${name} = __natlang_callable(${JSON.stringify(name)});`),
    ...serviceNames.map(name => `const ${name} = __live.services[${JSON.stringify(name)}];`),
    `const __natlang_present = (value: Record<string, unknown>) => Object.fromEntries(` +
      `Object.entries(value).filter(([, item]) => item !== undefined));`,
    `const __natlang_finish = async (__natlang_result: unknown, __natlang_bindings: Record<string, unknown> = ${capture}, ` +
      `__natlang_returned = false) => { const __natlang_settled = await __natlang_settle(` +
      `{ result: __natlang_result, bindings: __natlang_bindings }); ` +
      `return __natlang_output({ result: __natlang_settled.result === undefined ? null : __natlang_settled.result, ` +
      `returned: __natlang_returned, bindings: __natlang_present(__natlang_settled.bindings)` +
      `${mutableCaptures.length ? `, captures: { ${mutableCaptures.join(', ')} }` : ''} }); };`,
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
  result.rebinds = rebinds;
  result.program = emitted.outputText;
  return result;
}
