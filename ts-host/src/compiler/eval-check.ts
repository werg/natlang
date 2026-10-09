/**
 * Type-checked analysis of one lambda `eval` snippet: the same plan machinery as project builds,
 * over a virtual program whose declarations describe the lambda's typed scope.
 */
import ts from 'typescript';
import { createVirtualProgram, EVAL_COMPILER_OPTIONS } from './host.js';
import { analyzeInlineLambdas, type InlineLambdaPlan, type InlineRebindSite, type NatlangDiagnostic } from './inline.js';
import { hexDigest } from '../native/hash.js';
import { NEURALESE_LITERAL_INTRINSIC, type GenericInstantiation, type NeuraleseLiteral, type NeuraleseReadout } from './neuralese.js';
import { ITERATE_ON_SIGNATURE, RESERVED_CALLABLE_PROPERTIES } from './intrinsics.js';

export type EvalImport = { name: string; params: { name: string; type: string; optional?: boolean }[];
  returns: string; async: boolean; kind: 'natural language' | 'TypeScript' | 'directory reducer' | 'module';
  children: EvalImport[];
  /** A representation-generic result: `returns` is its crisp instance, `constraint` what a call may instantiate it as. */
  generic?: { constraint: string } };

export type EvalScopeDeclarations = {
  /** Stable parent call identity keeps independently created inline functions distinct. */
  scopeIdentity?: string;
  types: Record<string, string>;
  inputs: { name: string; type: string }[];
  locals: { name: string; type: string; mutable: boolean }[];
  captures: { name: string; type: string; mutable: boolean }[];
  imports: EvalImport[];
  services?: string[];
  /** Declarations of host services by name (TypeScript module text), so their methods and results are typed. */
  serviceDeclarations?: Record<string, string>;
  /** The call's return type: the type of the snippet's top-level return. */
  returns?: string;
  opaque?: string[];
};

/** The snippet is checked as the body of the call it runs in, so a top-level return has the call's return type. */
const evalWrapperPrefix = (returns?: string) => `async function __natlang_scope()${returns ? `: Promise<${returns}>` : ''} {\n`;
const SCOPE_FILE = '/__natlang__/eval/scope.ts';
const SNIPPET_FILE = '/__natlang__/eval/snippet.ts';

/** Natlang type text to TypeScript declaration text. `Live<"T", ...>` keeps its TypeScript name when resolvable. */
export function typeScriptText(natlang: string, known: ReadonlySet<string>): string {
  return natlang.replace(/Live<\s*"((?:[^"\\]|\\.)*)"(?:\s*,\s*"(?:[^"\\]|\\.)*")*\s*>/g, (_, text: string) => {
    const name = /^[A-Za-z_$][\w$]*/.exec(text)?.[0];
    return name && (known.has(name) || GLOBAL_TYPES.has(name)) ? text : 'any';
  });
}
const GLOBAL_TYPES = new Set(['Date', 'Map', 'Set', 'WeakMap', 'WeakSet', 'RegExp', 'Error', 'Promise', 'Uint8Array',
  'ArrayBuffer', 'Folder', 'FileHandle', 'IterationTrajectory']);

function importType(item: EvalImport, known: ReadonlySet<string>): string {
  const params = item.params.map(parameter => `${parameter.name}${parameter.optional ? '?' : ''}: ${typeScriptText(parameter.type, known)}`);
  const returns = typeScriptText(item.returns, known);
  const children = item.children.map(child => `readonly ${child.name}: ${importType(child, known)}`);
  const members = children.length ? ` & { ${children.join('; ')} }` : '';
  if (item.kind === 'module') return `{ ${children.join('; ')} }`;
  if (item.kind === 'TypeScript' && !item.async) return `((${params.join(', ')}) => ${returns})${members}`;
  if (item.generic) return `NatlangGenericFunction<[${params.join(', ')}], ${typeScriptText(item.generic.constraint, known)}, ${returns}>${members}`;
  return `NatlangFunction<[${params.join(', ')}], ${returns}>${members}`;
}

/** Declarations for the virtual scope file. */
export function scopeDeclarations(scope: EvalScopeDeclarations, iterationHelper?: string): string {
  const known = new Set(Object.keys(scope.types));
  const lines: string[] = [];
  for (const [name, text] of Object.entries(scope.types)) lines.push(`type ${name} = ${typeScriptText(text, known)};`);
  for (const input of scope.inputs) lines.push(`declare const ${input.name}: ${typeScriptText(input.type, known)};`);
  for (const local of scope.locals) lines.push(`declare ${local.mutable ? 'let' : 'const'} ${local.name}: ${typeScriptText(local.type, known)};`);
  for (const capture of scope.captures) lines.push(`declare ${capture.mutable ? 'let' : 'const'} ${capture.name}: ${typeScriptText(capture.type, known)};`);
  for (const item of scope.imports) lines.push(`declare const ${item.name}: ${importType(item, known)};`);
  for (const name of scope.services ?? []) {
    const declaration = scope.serviceDeclarations?.[name];
    if (name === 'neuralese') lines.push('declare const neuralese: Readonly<{ read<T>(value: Neuralese<T>): Promise<T> }>;');
    else if (declaration !== undefined) lines.push(serviceNamespace(name, declaration));
    else lines.push(`declare const ${name}: any;`);
  }
  for (const name of scope.opaque ?? []) lines.push(`declare const ${name}: any;`);
  if (iterationHelper) lines.push(`declare function ${iterationHelper}${ITERATE_ON_SIGNATURE};`);
  const named = new Set([...scope.inputs, ...scope.locals, ...scope.captures, ...scope.imports].map(item => item.name)
    .concat(scope.services ?? [], scope.opaque ?? []));
  if (!named.has('decide')) lines.push('declare function decide<A extends unknown[], T>(fn: (...args: A) => Promise<T>, ...args: A): ' +
    'Promise<{ value: T; probabilities: { value: T; probability: number }[]; confidence: number; scored: boolean }>;');
  // The runtime writes a model-written Neuralese literal as this call; its contextual type types it.
  lines.push(`declare function ${NEURALESE_LITERAL_INTRINSIC}<T>(id: string): T;`);
  // ...and a model-written soft function body, the whole template of `nl.with({ … })`...``, as this one.
  lines.push(`declare namespace ${NEURALESE_LITERAL_INTRINSIC} { function body(id: string): NeuraleseBody; }`);
  return lines.join('\n') + '\n';
}

/**
 * A service's declaration as the runtime holds it (`declare namespace name { … }` or `declare const name: …`), or
 * module text, wrapped as an ambient namespace. Imports are dropped: what they name resolves in the scope file or stays
 * unresolved, which the checker treats as unknown.
 */
function serviceNamespace(name: string, declaration: string): string {
  const body = declaration.split('\n').filter(line => !/^\s*import\s/.test(line)).join('\n');
  const namespace = /^\s*declare (?:namespace|const) /.test(body) ? body.trim() :
    `declare namespace ${name} {\n${body.replace(/^(\s*export\s+)declare\s+/gm, '$1')}\n}`;
  // A declaration that exports a constant named after the service (`export const calendar: CalendarService`) says the
  // service is that value: the binding takes the constant's type, and the declaration's other names stay its types.
  const own = new RegExp(`^\\s*export\\s+(?:declare\\s+)?const\\s+${name}\\s*:`, 'm');
  if (!namespace.startsWith(`declare namespace ${name} `) || !own.test(namespace)) return namespace;
  const types = `__natlang_service_${name}`;
  return `${namespace.replace(`declare namespace ${name} `, `declare namespace ${types} `)}\ndeclare const ${name}: typeof ${types}.${name};`;
}

/** Cheap test for whether a snippet needs the checked pass. */
export const needsEvalCheck = (source: string) => /\bnl\s*(?:<[^`]*>)?\s*`|\bnl\.with\b|\biterateOn\b|\b__neuralese[(.]/.test(source);

/**
 * Analyze `nl` expressions in an eval snippet. Spans in the returned plans and diagnostics are
 * relative to the snippet text.
 */
export function analyzeEvalSnippet(source: string, scope: EvalScopeDeclarations): { plans: InlineLambdaPlan[];
  diagnostics: NatlangDiagnostic[]; neuralese: NeuraleseLiteral[]; readouts: NeuraleseReadout[]; rebinds: InlineRebindSite[];
  instantiations: GenericInstantiation[] } {
  const prefix = evalWrapperPrefix(scope.returns === undefined ? undefined : typeScriptText(scope.returns, new Set(Object.keys(scope.types))));
  const virtualProgram = (text: string, iterationHelper?: string) => createVirtualProgram({
    [SCOPE_FILE]: scopeDeclarations(scope, iterationHelper), [SNIPPET_FILE]: `${prefix}${text}\n}\n` }, EVAL_COMPILER_OPTIONS);
  let analyzedSource = source;
  let program = virtualProgram(analyzedSource);
  // The eval lowerer makes `fn.iterateOn(initial)` equivalent to the typed free intrinsic.
  // Mirror that equivalence for type analysis only when the receiver is a callable without its own
  // declared `iterateOn` method. A declared custom method keeps its actual result type.
  if (/\.\s*iterateOn\b/.test(source)) {
    const initialSnippet = program.getSourceFile(SNIPPET_FILE)!;
    const checker = program.getTypeChecker();
    const helperNames = new Set([...scope.inputs, ...scope.locals, ...scope.captures, ...scope.imports]
      .map(item => item.name).concat(scope.services ?? [], scope.opaque ?? [], Object.keys(scope.types)));
    let ordinal = 0, helper = '';
    do { helper = `__i${String(ordinal++).padStart(6, '0')}`; } while (source.includes(helper) || helperNames.has(helper));
    const edits: { start: number; end: number; text: string }[] = [];
    const visit = (node: ts.Node): void => {
      if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) &&
          node.expression.name.text === 'iterateOn' && !ts.isTaggedTemplateExpression(node.expression.expression)) {
        const receiver = node.expression.expression;
        const declaredMethod = checker.getSymbolAtLocation(node.expression.name);
        const receiverType = checker.getTypeAtLocation(receiver);
        const isOpaque = (receiverType.flags & (ts.TypeFlags.Any | ts.TypeFlags.Unknown)) !== 0;
        const callable = checker.getSignaturesOfType(receiverType, ts.SignatureKind.Call).length > 0;
        if (!declaredMethod && !isOpaque && callable) {
          const start = receiver.getStart(initialSnippet), end = node.arguments.pos;
          const generic = node.typeArguments?.length ? `<${node.typeArguments.map(item => item.getText(initialSnippet)).join(', ')}>` : '';
          const replacement = `${helper}${generic}(${receiver.getText(initialSnippet)},`;
          const original = source.slice(start - prefix.length, end - prefix.length);
          if (replacement.length <= original.length)
            edits.push({ start: start - prefix.length, end: end - prefix.length,
              text: replacement + ' '.repeat(original.length - replacement.length) });
        }
      }
      ts.forEachChild(node, visit);
    };
    visit(initialSnippet);
    if (edits.length) {
      for (const edit of edits.sort((a, b) => b.start - a.start))
        analyzedSource = analyzedSource.slice(0, edit.start) + edit.text + analyzedSource.slice(edit.end);
      program = virtualProgram(analyzedSource, helper);
    }
  }
  const snippet = program.getSourceFile(SNIPPET_FILE)!;
  const scopeFile = program.getSourceFile(SCOPE_FILE)!;
  const inputs = new Set(scope.inputs.map(input => input.name));
  const { plans, diagnostics, neuralese, readouts, rebinds, instantiations } = analyzeInlineLambdas(program, [snippet], {
    sourceRevision: hexDigest(`${scope.scopeIdentity ?? ''}\0${source}`),
    scopeFiles: [scopeFile], displayPath: () => 'eval', recursiveTypes: true,
    classify: declaration => declaration.getSourceFile() === scopeFile ?
      (ts.isVariableDeclaration(declaration) && ts.isIdentifier(declaration.name) && inputs.has(declaration.name.text) ? 'input' : 'local') :
      isSnippetTopLevel(declaration) ? 'local' : 'block',
  });
  const offset = prefix.length;
  const shift = <T extends { start: number; end: number; line: number }>(item: T): T =>
    ({ ...item, start: item.start - offset, end: item.end - offset, line: Math.max(1, item.line - 1) });
  return { plans: plans.map(plan => ({ ...plan, sourceSpan: shift(plan.sourceSpan),
    templateSpan: shift(plan.templateSpan),
    interpolations: plan.interpolations.map(item => ({ ...item, sourceSpan: shift(item.sourceSpan) })),
    captures: plan.captures.map(capture => ({ ...capture, mentionSpan: capture.mentionSpan - offset })) })),
    diagnostics: diagnostics.map(shift), neuralese: neuralese.map(shift), readouts: readouts.map(item => ({ ...shift(item),
      ...(item.callback ? { callback: { start: item.callback.start - offset, end: item.callback.end - offset } } : {}) })),
    rebinds: rebinds.map(site => ({ ...site, start: site.start - offset, end: site.end - offset,
      templateStart: site.templateStart - offset, templateEnd: site.templateEnd - offset })),
    instantiations: instantiations.map(shift) };
}

/**
 * Reads of a field the value's declared type does not have (`entry.content` where `EntryRecord` has no `content`).
 * Such a read is `undefined` at run time, which code then takes for "none" and acts on. Only declared types count: a
 * named type of the program, a service or the eval, or the declared type of a call input or service. A local whose
 * type the runtime inferred from its value may lack optional fields its source had, so its anonymous type is not
 * held against a read, nor is a field one member of a union declares. Writes are not checked.
 */
export function checkEvalFields(source: string, scope: EvalScopeDeclarations): NatlangDiagnostic[] {
  const prefix = evalWrapperPrefix(scope.returns === undefined ? undefined : typeScriptText(scope.returns, new Set(Object.keys(scope.types))));
  const program = createVirtualProgram({ [SCOPE_FILE]: scopeDeclarations(scope), [SNIPPET_FILE]: `${prefix}${source}\n}\n` }, EVAL_COMPILER_OPTIONS);
  const snippet = program.getSourceFile(SNIPPET_FILE)!, scopeFile = program.getSourceFile(SCOPE_FILE)!;
  const checker = program.getTypeChecker();
  const inferred = new Set([...scope.locals, ...scope.captures].map(item => item.name));
  const declared = (declaration: ts.Declaration): boolean => {
    const file = declaration.getSourceFile();
    if (file === snippet) return ts.isTypeAliasDeclaration(declaration) || ts.isInterfaceDeclaration(declaration) ||
      !!ts.findAncestor(declaration, node => ts.isTypeAliasDeclaration(node) || ts.isInterfaceDeclaration(node));
    if (file !== scopeFile) return false;
    const variable = ts.findAncestor(declaration, ts.isVariableDeclaration);
    return !(variable && ts.isIdentifier(variable.name) && inferred.has(variable.name.text));
  };
  const declaredType = (type: ts.Type): boolean => {
    const symbol = type.aliasSymbol ?? type.getSymbol();
    return !!symbol?.declarations?.length && symbol.declarations.every(declared);
  };
  const found: NatlangDiagnostic[] = [];
  for (const diagnostic of program.getSemanticDiagnostics(snippet)) {
    if ((diagnostic.code !== 2339 && diagnostic.code !== 2551) || diagnostic.start === undefined) continue;
    const name = findNode(snippet, diagnostic.start);
    const access = name?.parent;
    if (!name || !access || !ts.isPropertyAccessExpression(access) || access.name !== name) continue;
    if (ts.isBinaryExpression(access.parent) && access.parent.left === access &&
        access.parent.operatorToken.kind >= ts.SyntaxKind.FirstAssignment && access.parent.operatorToken.kind <= ts.SyntaxKind.LastAssignment) continue;
    const receiver = checker.getNonNullableType(checker.getTypeAtLocation(access.expression));
    // A callable has the runtime's intrinsic members (iterateOn, in, with) whatever its declared type lists.
    if (RESERVED_CALLABLE_PROPERTIES.has(name.text) && receiver.getCallSignatures().length) continue;
    const members = receiver.isUnion() ? receiver.types : [receiver];
    if (!members.length || !members.every(declaredType)) continue;
    // A field one member of a union declares is how code tells the members apart (`if (result.error)`): reading it is
    // the test, as at run time, where the guard lets it through.
    if (members.some(member => checker.getPropertyOfType(member, name.text))) continue;
    const fields = members.map(member => {
      const names = checker.getPropertiesOfType(member).map(property => property.name);
      return `${checker.typeToString(member)} has ${names.length ? names.join(', ') : 'no fields'}`;
    }).join('; ');
    const start = access.name.getStart(snippet), end = access.name.getEnd();
    const location = snippet.getLineAndCharacterOfPosition(start);
    found.push({ code: 'undeclared-field', severity: 'error', file: 'eval', start: start - prefix.length, end: end - prefix.length,
      line: Math.max(1, location.line), column: location.character + 1,
      message: `${access.expression.getText(snippet)} has no field ${name.text}: ${fields}. Read the field that holds what you need.` });
  }
  return found;
}

/** The innermost identifier at `position`. */
function findNode(file: ts.SourceFile, position: number): ts.Identifier | undefined {
  let found: ts.Identifier | undefined;
  const visit = (node: ts.Node): void => {
    if (position < node.getStart(file) || position >= node.getEnd()) return;
    if (ts.isIdentifier(node)) found = node;
    ts.forEachChild(node, visit);
  };
  visit(file);
  return found;
}

function isSnippetTopLevel(declaration: ts.Declaration): boolean {
  let current: ts.Node = declaration;
  while (current.parent) {
    const parent: ts.Node = current.parent;
    if (ts.isBlock(parent) && parent.parent && ts.isFunctionDeclaration(parent.parent) && parent.parent.name?.text === '__natlang_scope')
      return true;
    if (ts.isBlock(parent) || ts.isFunctionLike(parent)) return false;
    current = parent;
  }
  return false;
}

export const EVAL_OPTIONS = EVAL_COMPILER_OPTIONS;
