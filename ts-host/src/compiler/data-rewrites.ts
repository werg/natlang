/**
 * Training-data rewrites for Neuralese (spec/NEURALESE_DATA.md): compiler passes over model-written eval code.
 *
 * - Eager typing: every unannotated `const`/`let` gets the type the checker infers, in natlang type syntax.
 * - Explicit captures: an inline ``nl`…` `` that captures by mention becomes ``nl.with({ … })`…` `` listing exactly
 *   the analysed captures; a captured `let` becomes `live(x)`, a function-typed capture stays a snapshot.
 *
 * Each pass checks itself by recompiling: the rewritten code must have the original's diagnostics (and, for explicit
 * captures, the original's capture sets). A rewrite that fails its check is not applied. Replaying the recorded
 * execution is the caller's check (it needs the runtime and the record's outcomes).
 *
 * Code is checked as the body of the call, after `prelude` (earlier code of the same call, such as the opening eval
 * and earlier evals), with `declarations` (type aliases, ambient declarations) and `scope` at top level. Only
 * declarations and sites inside `code` are rewritten.
 */
import ts from 'typescript';
import { createVirtualProgram, EVAL_COMPILER_OPTIONS } from './host.js';
import { scopeDeclarations, typeScriptText, type EvalScopeDeclarations } from './eval-check.js';
import { analyzeInlineLambdas, type CapturePlan, type InlineLambdaPlan } from './inline.js';
import { describeTarget, TargetError } from './targets.js';

export const DATA_REWRITE_MIGRATION = 'neuralese-language';
export const DATA_REWRITE_VERSION = 'natlang.data-rewrite/1';

export type RewriteInput = {
  code: string;
  /** Statements of the same call that ran before `code` (they are context; they are not rewritten). */
  prelude?: string;
  /** Top-level TypeScript declarations: type aliases and `declare` lines of the call's opening. */
  declarations?: string;
  scope?: Partial<EvalScopeDeclarations>;
};

export type RewriteCheck = { ok: boolean; before: string[]; after: string[]; reason?: string };
export type EagerTypingResult = { code: string; annotated: { name: string; type: string }[];
  skipped: { name: string; reason: string }[]; check: RewriteCheck };
export type CaptureSite = { start: number; end: number; captures: { name: string; mode: 'snapshot' | 'live' }[] };
export type ExplicitCapturesResult = { code: string; sites: CaptureSite[]; check: RewriteCheck };

const SCOPE_FILE = '/__natlang__/rewrite/scope.ts';
const SNIPPET_FILE = '/__natlang__/rewrite/snippet.ts';

type Compiled = { program: ts.Program; checker: ts.TypeChecker; file: ts.SourceFile; scopeFile: ts.SourceFile; offset: number;
  end: number; diagnostics: string[]; plans: InlineLambdaPlan[] };

function fullScope(scope: Partial<EvalScopeDeclarations> = {}): EvalScopeDeclarations {
  return { types: {}, inputs: [], locals: [], captures: [], imports: [], ...scope };
}

/** Compile `code` in its call's context; plan spans are relative to `code`. */
function compile(input: RewriteInput, code: string): Compiled {
  const scope = fullScope(input.scope);
  const returns = scope.returns === undefined ? '' : `: Promise<${typeScriptText(scope.returns, new Set(Object.keys(scope.types)))}>`;
  const declarations = input.declarations ?? '';
  const scopeText = scopeDeclarations(scope) + declarations + '\n' +
    (/\bread_inputs\b/.test(declarations) ? '' : 'declare function read_inputs(): any;\n');
  const prefix = `async function __natlang_scope()${returns} {\n${input.prelude ? input.prelude.replace(/\n*$/, '\n') : ''}`;
  const program = createVirtualProgram({ [SCOPE_FILE]: scopeText, [SNIPPET_FILE]: `${prefix}${code}\n}\n` }, EVAL_COMPILER_OPTIONS);
  const file = program.getSourceFile(SNIPPET_FILE)!;
  const scopeFile = program.getSourceFile(SCOPE_FILE)!;
  const flatten = (diagnostic: ts.Diagnostic) => `TS${diagnostic.code}: ${ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')}`;
  const analysis = analyzeInlineLambdas(program, [file], { scopeFiles: [scopeFile], displayPath: () => 'eval', recursiveTypes: true });
  const diagnostics = [...program.getSyntacticDiagnostics(file), ...program.getSemanticDiagnostics(file)].map(flatten)
    .concat(analysis.diagnostics.map(diagnostic => `${diagnostic.code}: ${diagnostic.message}`)).sort();
  const offset = prefix.length;
  const plans = analysis.plans.filter(plan => plan.sourceSpan.start >= offset).map(plan => ({ ...plan,
    sourceSpan: { ...plan.sourceSpan, start: plan.sourceSpan.start - offset, end: plan.sourceSpan.end - offset } }));
  return { program, checker: program.getTypeChecker(), file, scopeFile, offset, end: offset + code.length, diagnostics, plans };
}

const sameDiagnostics = (before: string[], after: string[]) => before.length === after.length && before.every((item, index) => item === after[index]);

/** Apply span edits (relative to `code`), last first. */
function applyEdits(code: string, edits: readonly { start: number; end: number; text: string }[]): string {
  let result = code;
  for (const edit of [...edits].sort((a, b) => b.start - a.start)) result = result.slice(0, edit.start) + edit.text + result.slice(edit.end);
  return result;
}

// --- Eager typing ------------------------------------------------------------------------------------------------

/** The annotation for one declaration, or why there is none. */
function annotationOf(compiled: Compiled, declaration: ts.VariableDeclaration): { type: string } | { reason: string } {
  const { checker, program } = compiled;
  let type = ts.isIdentifier(declaration.name) ? checker.getTypeAtLocation(declaration.name) :
    checker.getWidenedType(checker.getTypeAtLocation(declaration.initializer!));
  // `const n = 0` is a number to the model, not the literal type 0.
  if (type.flags & (ts.TypeFlags.StringLiteral | ts.TypeFlags.NumberLiteral | ts.TypeFlags.BooleanLiteral | ts.TypeFlags.BigIntLiteral) &&
    declaration.initializer && (ts.isLiteralExpression(declaration.initializer) || declaration.initializer.kind === ts.SyntaxKind.TrueKeyword ||
      declaration.initializer.kind === ts.SyntaxKind.FalseKeyword || ts.isNoSubstitutionTemplateLiteral(declaration.initializer) ||
      ts.isPrefixUnaryExpression(declaration.initializer)))
    type = checker.getBaseTypeOfLiteralType(type);
  if (type.flags & (ts.TypeFlags.Any | ts.TypeFlags.Unknown)) return { reason: type.flags & ts.TypeFlags.Any ? 'any' : 'unknown' };
  const signature = type.getCallSignatures().length > 0 && type.getProperties().length === 0;
  if (signature) {
    // A function-valued declaration gets its full signature.
    const text = checker.typeToString(type, declaration, ts.TypeFormatFlags.NoTruncation | ts.TypeFormatFlags.UseAliasDefinedOutsideCurrentScope);
    if (/\bany\b|\bunknown\b|import\(|typeof /.test(text)) return { reason: 'inexpressible' };
    return { type: text };
  }
  let described;
  try { described = describeTarget(program, checker, type, { location: declaration }); }
  catch (error) { if (error instanceof TargetError) return { reason: 'inexpressible' }; throw error; }
  const text = described.natlang;
  if (!text || /\bany\b|Live</.test(text)) return { reason: 'inexpressible' };
  if (text === 'unknown') return { reason: 'unknown' };
  return { type: text.replace(/\(([A-Za-z_$][\w$]*)\)\[\]/g, '$1[]') };
}

/** Unannotated declarations inside `code` (statement and nested declarations; not loop heads). */
function declarationsIn(compiled: Compiled): ts.VariableDeclaration[] {
  const found: ts.VariableDeclaration[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isVariableDeclaration(node) && !node.type && node.initializer && ts.isVariableStatement(node.parent.parent) &&
      node.getStart(compiled.file) >= compiled.offset && node.getEnd() <= compiled.end) found.push(node);
    ts.forEachChild(node, visit);
  };
  visit(compiled.file);
  return found;
}

/**
 * Insert inferred annotations. Declarations whose type is any, unknown or inexpressible are skipped and counted. If
 * the annotated code's diagnostics differ, annotations are kept only where each one alone preserves them.
 */
export function eagerTyping(input: RewriteInput): EagerTypingResult {
  const original = compile(input, input.code);
  const annotated: { name: string; type: string }[] = [];
  const skipped: { name: string; reason: string }[] = [];
  const edits: { start: number; end: number; text: string; name: string; type: string }[] = [];
  for (const declaration of declarationsIn(original)) {
    const name = declaration.name.getText(original.file);
    const result = annotationOf(original, declaration);
    if ('reason' in result) { skipped.push({ name, reason: result.reason }); continue; }
    const at = declaration.name.getEnd() - original.offset;
    edits.push({ start: at, end: at, text: `: ${result.type}`, name, type: result.type });
  }
  const check = (chosen: typeof edits) => compile(input, applyEdits(input.code, chosen)).diagnostics;
  let chosen = edits;
  let after = chosen.length ? check(chosen) : original.diagnostics;
  if (!sameDiagnostics(original.diagnostics, after)) {
    chosen = edits.filter(edit => {
      const ok = sameDiagnostics(original.diagnostics, check([edit]));
      if (!ok) skipped.push({ name: edit.name, reason: 'changes-diagnostics' });
      return ok;
    });
    after = check(chosen);
  }
  if (!sameDiagnostics(original.diagnostics, after))
    return { code: input.code, annotated: [], skipped: [...skipped, ...chosen.map(edit => ({ name: edit.name, reason: 'changes-diagnostics' }))],
      check: { ok: false, before: original.diagnostics, after, reason: 'annotations change diagnostics' } };
  annotated.push(...chosen.map(({ name, type }) => ({ name, type })));
  return { code: applyEdits(input.code, chosen), annotated, skipped, check: { ok: true, before: original.diagnostics, after } };
}

// --- Explicit captures -------------------------------------------------------------------------------------------

const isFunctionCapture = (capture: CapturePlan) => /=>/.test(capture.type.natlang ?? capture.type.text) ||
  capture.type.host?.kind === 'function';

/** The capture listing for one site: shorthand snapshots, `live(x)` for a captured `let`. */
export function captureListing(captures: readonly CapturePlan[]): { text: string; captures: CaptureSite['captures'] } {
  const listed = captures.map(capture => ({ name: capture.name,
    mode: capture.mutable && !isFunctionCapture(capture) ? 'live' as const : 'snapshot' as const }));
  return { captures: listed, text: listed.length ?
    `{ ${listed.map(item => item.mode === 'live' ? `${item.name}: live(${item.name})` : item.name).join(', ')} }` : '{}' };
}

function taggedTemplateAt(compiled: Compiled, start: number, end: number): ts.TaggedTemplateExpression | undefined {
  let found: ts.TaggedTemplateExpression | undefined;
  const visit = (node: ts.Node): void => {
    if (found) return;
    if (ts.isTaggedTemplateExpression(node) && node.getStart(compiled.file) === start + compiled.offset && node.getEnd() === end + compiled.offset)
      found = node;
    else ts.forEachChild(node, visit);
  };
  visit(compiled.file);
  return found;
}

/**
 * Rewrite inline nl sites that capture by mention to `nl.with({ … })`. The rewrite is kept only when the new code has
 * the same diagnostics and every site has the same capture names as before.
 */
export function explicitCaptures(input: RewriteInput): ExplicitCapturesResult {
  const original = compile(input, input.code);
  const edits: { start: number; end: number; text: string }[] = [];
  const sites: CaptureSite[] = [];
  for (const plan of original.plans) {
    if (plan.explicitCaptures) continue;
    const node = taggedTemplateAt(original, plan.sourceSpan.start, plan.sourceSpan.end);
    if (!node || !ts.isIdentifier(node.tag)) continue;
    const listing = captureListing(plan.captures);
    const typeArguments = node.typeArguments?.length ? `<${node.typeArguments.map(item => item.getText(original.file)).join(', ')}>` : '';
    edits.push({ start: node.getStart(original.file) - original.offset, end: node.template.getStart(original.file) - original.offset,
      text: `${node.tag.text}.with${typeArguments}(${listing.text})` });
    sites.push({ start: plan.sourceSpan.start, end: plan.sourceSpan.end, captures: listing.captures });
  }
  if (!edits.length) return { code: input.code, sites: [], check: { ok: true, before: original.diagnostics, after: original.diagnostics } };
  const code = applyEdits(input.code, edits);
  const rewritten = compile(input, code);
  const fail = (reason: string) => ({ code: input.code, sites: [], check: { ok: false, before: original.diagnostics,
    after: rewritten.diagnostics, reason } });
  if (!sameDiagnostics(original.diagnostics, rewritten.diagnostics)) return fail('the rewrite changes diagnostics');
  const names = (plans: InlineLambdaPlan[]) => plans.map(plan => plan.captures.map(capture => capture.name).sort().join(',')).join(';');
  const explicitAfter = rewritten.plans.filter(plan => plan.explicitCaptures);
  const before = original.plans.filter(plan => !plan.explicitCaptures);
  if (explicitAfter.length !== original.plans.length || names(rewritten.plans) !== names(original.plans))
    return fail(`capture sets differ: ${names(before)} → ${names(explicitAfter)}`);
  return { code, sites, check: { ok: true, before: original.diagnostics, after: rewritten.diagnostics } };
}

// --- Trajectory records ------------------------------------------------------------------------------------------

type Message = { role: string; content?: unknown; tool_calls?: { id?: string; function: { name: string; arguments: unknown } }[] };
export type TrajectoryRewriteStats = { evals: number; annotated: number; skipped: Record<string, number>; sites: number;
  failed: { eval: number; pass: 'eager-typing' | 'explicit-captures'; reason: string }[] };

const textOf = (content: unknown): string => typeof content === 'string' ? content :
  Array.isArray(content) ? content.map(part => (part as { text?: string }).text ?? '').join('') : '';
const argumentsOf = (raw: unknown): Record<string, unknown> | undefined => {
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) return raw as Record<string, unknown>;
  if (typeof raw === 'string') try { return JSON.parse(raw) as Record<string, unknown>; } catch { return undefined; }
  return undefined;
};

/**
 * The call context of a materialized trajectory: the opening message's type declarations, and the opening eval
 * (`scope_0`) split into top-level `declare` lines and body statements.
 */
export function trajectoryContext(messages: readonly Message[]): { declarations: string; prelude: string } {
  const opening = textOf(messages.find(message => message.role === 'user')?.content);
  const types = opening.includes('\nInstructions:') ? opening.slice(opening.indexOf('\n') + 1, opening.indexOf('\nInstructions:')) : '';
  const scopeCall = messages.flatMap(message => message.tool_calls ?? []).find(call => call.id === 'scope_0');
  const scopeCode = String(argumentsOf(scopeCall?.function.arguments)?.code ?? '');
  const parsed = ts.createSourceFile('scope.ts', scopeCode, ts.ScriptTarget.ES2022, true);
  const top: string[] = [], body: string[] = [];
  for (const statement of parsed.statements) {
    const ambient = ts.canHaveModifiers(statement) && ts.getModifiers(statement)?.some(modifier => modifier.kind === ts.SyntaxKind.DeclareKeyword);
    const declaration = ts.isTypeAliasDeclaration(statement) || ts.isInterfaceDeclaration(statement) || ts.isClassDeclaration(statement);
    (ambient || declaration ? top : body).push(statement.getText(parsed));
  }
  return { declarations: [types, ...top].join('\n'), prelude: body.join('\n') };
}

/**
 * Rewrite every model-written eval of a trajectory (its `messages` after the opening, then `target`): eager typing,
 * then explicit captures. Earlier evals are context for later ones, as they ran.
 */
export function rewriteTrajectory<T extends { messages: Message[]; target?: Message }>(record: T,
  options: { scope?: Partial<EvalScopeDeclarations> } = {}): { record: T; stats: TrajectoryRewriteStats } {
  const context = trajectoryContext(record.messages);
  const stats: TrajectoryRewriteStats = { evals: 0, annotated: 0, skipped: {}, sites: 0, failed: [] };
  const earlier: string[] = [];
  const rewriteMessage = (message: Message): Message => {
    if (message.role !== 'assistant' || !message.tool_calls?.length) return message;
    return { ...message, tool_calls: message.tool_calls.map(call => {
      if (call.function.name !== 'eval' || call.id === 'scope_0') return call;
      const args = argumentsOf(call.function.arguments);
      if (!args || typeof args.code !== 'string') return call;
      const index = stats.evals++;
      const input = { code: args.code, prelude: [context.prelude, ...earlier].join('\n'), declarations: context.declarations, scope: options.scope };
      earlier.push(args.code);
      const typed = eagerTyping(input);
      if (!typed.check.ok) stats.failed.push({ eval: index, pass: 'eager-typing', reason: typed.check.reason ?? '' });
      stats.annotated += typed.annotated.length;
      for (const item of typed.skipped) stats.skipped[item.reason] = (stats.skipped[item.reason] ?? 0) + 1;
      const explicit = explicitCaptures({ ...input, code: typed.code });
      if (!explicit.check.ok) stats.failed.push({ eval: index, pass: 'explicit-captures', reason: explicit.check.reason ?? '' });
      stats.sites += explicit.sites.length;
      if (explicit.code === args.code) return call;
      const rewritten = { ...args, code: explicit.code };
      return { ...call, function: { ...call.function, arguments: typeof call.function.arguments === 'string' ? JSON.stringify(rewritten) : rewritten } };
    }) };
  };
  const messages = record.messages.map(rewriteMessage);
  const target = record.target ? rewriteMessage(record.target) : undefined;
  return { record: { ...record, messages, ...(target ? { target } : {}) }, stats };
}
