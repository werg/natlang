/**
 * Natlang for a host's eval cells: source a host runs as one unit (Vibestudio's agent `eval`: TSX, top-level `await`,
 * `import`, a top-level `return`), where `nl` is ambient rather than imported.
 *
 * `lowerHostCell` plans every ambient `nl` expression in a cell with the project compiler's own analysis
 * (`analyzeInlineLambdas`: contextual type, immediate call arguments, result context, typed holes) over the cell plus
 * the host's ambient declarations, and rewrites each into the lowered form project builds emit
 * (`nl.__inline(plan, values, accessors, context)`, see `lower.ts`). Everything else in the cell stays as written, types
 * and JSX included, so the host's own transform runs next, and a lowered site keeps the lines of the code it replaces.
 *
 * `declarationsForValues` is how a host describes the current contents of ambient bindings (`scope`) to that analysis.
 */
import ts from 'typescript';
import { createNatlangCompilerHost, EVAL_COMPILER_OPTIONS, type CompilerFiles, type LibProvider } from './host.js';
import { analyzeInlineLambdas, resolveIntrinsic, suffixWithCallOf, type InlineLambdaPlan, type NatlangDiagnostic } from './inline.js';
import { explicitInlineCall, plainInlineCall, type InlineSite } from './lower.js';
import { applyNlCallEdits, nlCallEdits } from './nl-call.js';
import { INTRINSICS_FILE } from './intrinsics.js';
import type { ProgramDescriptor } from '../adaptation/types.js';
import { loadCallableFolder, NatlangSourceError, type SourceFiles } from '../runtime/loader.js';
import { declarationNamespace } from '../native/external.js';
import { hexDigest } from '../native/hash.js';
import { isLive, isPlainRecord } from '../native/values.js';
import { inferValueType } from '../native/runtime.js';

export type HostCellOptions = {
  /** Name the cell goes by in diagnostics and plan spans (default `cell.ts`, `cell.tsx` with `jsx`). */
  fileName?: string;
  /** The cell is TSX: parse JSX. */
  jsx?: boolean;
  /**
   * Declaration text (a global `.d.ts`, no top-level `import`/`export`) for the host's ambient free variables, with the
   * current contents of `scope` and the like (see `declarationsForValues`). They give inference its types. They are not
   * captured by mentioning their names in an instruction: authority is passed explicitly, as an argument or `nl.with`.
   * Names the declarations leave out, and modules the cell imports, are `any`.
   */
  ambientDeclarations: string;
  /** Typed host services, declared by name in the same ambient scope (the declarations of a program descriptor). */
  services?: ProgramDescriptor['services'];
  /** The callable folder every `nl` in the cell is bound to (a `natlang.d` directory), embedded in each site as in project builds. */
  context?: { directory: string; files: SourceFiles };
  /** Identity of this cell's source revision, mixed into the plans' definition IDs (default: a hash of the source). */
  revision?: string;
  /** TypeScript library declarations, for hosts without a TypeScript install on disk. */
  libs?: LibProvider;
};

export type HostCellResult = {
  /**
   * The cell with each `nl` expression lowered; the cell unchanged when it uses no ambient `nl` or has an error
   * diagnostic.
   */
  code: string;
  /** One plan per lowered site, with spans in the original cell. */
  plans: InlineLambdaPlan[];
  /** Syntax errors and natlang errors, with 1-based lines and columns in the original cell. */
  diagnostics: NatlangDiagnostic[];
  /** The cell refers to the ambient `nl` (a local declaration of that name is not it). */
  usesNl: boolean;
};

// Not under /__natlang__/: the host caches those files as shared, parsed without the program's module detection.
const CELL_DIRECTORY = '/host-cell';
const AMBIENT_FILE = `${CELL_DIRECTORY}/ambient.d.ts`;
// Every cell is a module (top-level `await`, imports and exports are legal in one); JSX is parsed, not emitted.
const CELL_COMPILER_OPTIONS: ts.CompilerOptions = { ...EVAL_COMPILER_OPTIONS,
  moduleDetection: ts.ModuleDetectionKind.Force, jsx: ts.JsxEmit.Preserve };
// Cells are checked against what the host declares, never against whatever files the process can see.
const NO_FILES: CompilerFiles = { readFile: () => undefined, fileExists: () => false, directoryExists: () => false,
  getDirectories: () => [], realpath: path => path };
// A cell that never writes the name `nl` cannot use it: no program is built for it.
const MENTIONS_NL = /(?<![\w$])nl(?![\w$])/;
const NL_CALL = /\bnl\s*(?:<[^()]*>)?\s*\(/;

function servicesDeclarations(services: ProgramDescriptor['services']): string {
  return Object.entries(services?.declarations ?? {}).map(([name, text]) =>
    /^\s*declare (?:namespace|const) /.test(text) ? text.trim() : declarationNamespace(name, text)).join('\n');
}

export function lowerHostCell(source: string, options: HostCellOptions): HostCellResult {
  if (!MENTIONS_NL.test(source)) return { code: source, plans: [], diagnostics: [], usesNl: false };
  const display = options.fileName ?? (options.jsx ? 'cell.tsx' : 'cell.ts');
  const cellFile = `${CELL_DIRECTORY}/cell${options.jsx ? '.tsx' : '.ts'}`;
  const ambient = `${options.ambientDeclarations}\n${servicesDeclarations(options.services)}\n`;
  const build = (text: string) => {
    const host = createNatlangCompilerHost({ options: CELL_COMPILER_OPTIONS, files: NO_FILES, libs: options.libs,
      virtual: new Map([[AMBIENT_FILE, ambient], [cellFile, text]]) });
    const program = ts.createProgram({ rootNames: [INTRINSICS_FILE, AMBIENT_FILE, cellFile], options: CELL_COMPILER_OPTIONS, host });
    return { program, cell: program.getSourceFile(cellFile)!, checker: program.getTypeChecker() };
  };
  const diagnosticOf = (file: ts.SourceFile, shown: string, diagnostic: ts.Diagnostic): NatlangDiagnostic => {
    const start = diagnostic.start ?? 0, location = file.getLineAndCharacterOfPosition(start);
    return { file: shown, start, end: start + (diagnostic.length ?? 0), line: location.line + 1, column: location.character + 1,
      code: 'typescript', severity: 'error', message: ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n') };
  };

  let { program, cell, checker } = build(source);
  const referencesNl = (file: ts.SourceFile, check: ts.TypeChecker): boolean => {
    let found = false;
    const visit = (node: ts.Node): void => {
      if (found) return;
      if (ts.isIdentifier(node) && node.text === 'nl' && resolveIntrinsic(check, node) === 'nl') found = true;
      else ts.forEachChild(node, visit);
    };
    visit(file);
    return found;
  };
  const usesNl = referencesNl(cell, checker);
  // A cell with no ambient `nl` is the host's own to parse and report.
  if (!usesNl) return { code: source, plans: [], diagnostics: [], usesNl };
  const ambientFile = program.getSourceFile(AMBIENT_FILE)!;
  const syntax = [...program.getSyntacticDiagnostics(ambientFile).map(item => diagnosticOf(ambientFile, 'ambient declarations', item)),
    ...program.getSyntacticDiagnostics(cell).map(item => diagnosticOf(cell, display, item))];
  if (syntax.length) return { code: source, plans: [], diagnostics: syntax, usesNl };

  // `await nl("...")` and `nl<T>("...")` read as the one-shot call ``nl`...`()``: same-length rewrites, so every position
  // stays that of the cell as written. Only the built-in `nl` is rewritten; a local function of that name is its own.
  let text = source;
  if (NL_CALL.test(source)) {
    const edits = nlCallEdits(cell, callee => resolveIntrinsic(checker, callee) === 'nl');
    if (edits.length) { text = applyNlCallEdits(source, edits); ({ program, cell, checker } = build(text)); }
  }

  const analysis = analyzeInlineLambdas(program, [cell], { displayPath: () => display, sourceRevision: options.revision ?? hexDigest(source),
    recursiveTypes: true, allowShadowing: true });
  const diagnostics = [...analysis.diagnostics];
  let context: string | undefined;
  if (options.context && analysis.plans.length) {
    try { context = JSON.stringify(loadCallableFolder(options.context.directory, options.context.files)); }
    catch (error) {
      if (!(error instanceof NatlangSourceError)) throw error;
      diagnostics.push({ file: display, start: 0, end: 0, line: 1, column: 1, code: 'callable-scope', severity: 'error', message: error.message });
    }
  }
  if (diagnostics.some(item => item.severity === 'error')) return { code: source, plans: analysis.plans, diagnostics, usesNl };

  // The sites to rewrite: a tagged template with a plan, or its `.with(...)` suffix call (the whole call is the site).
  const planAt = new Map(analysis.plans.map(plan => [`${plan.sourceSpan.start}:${plan.sourceSpan.end}`, plan]));
  const f = ts.factory, printer = ts.createPrinter();
  type Site = { start: number; end: number; expression: () => ts.Expression };
  const sites: Site[] = [];
  // Text of the cell between two positions, nested sites lowered. A lowered expression stands in the call's arguments
  // as raw text, keeping its formatting; one beginning with `{` is parenthesized so an arrow body cannot read it as a block.
  const render = (from: number, to: number): string => {
    let out = '', cursor = from;
    for (const site of sites) {
      if (site.start < cursor || site.end > to) continue;
      out += text.slice(cursor, site.start) + lowered(site);
      cursor = site.end;
    }
    return out + text.slice(cursor, to);
  };
  const raw = (node: ts.Expression): ts.Expression => {
    const code = render(node.getStart(cell), node.getEnd());
    return f.createIdentifier(code.startsWith('{') ? `(${code})` : code);
  };
  const lowered = (site: Site): string => {
    const printed = printer.printNode(ts.EmitHint.Expression, site.expression(), cell);
    // The cell's later lines stay where they were: the newlines the site's source held are kept inside the call.
    const lost = (text.slice(site.start, site.end).match(/\n/g)?.length ?? 0) - (printed.match(/\n/g)?.length ?? 0);
    return lost > 0 && printed.endsWith(')') ? `${printed.slice(0, -1)}${'\n'.repeat(lost)})` : printed;
  };
  const visit = (node: ts.Node): void => {
    const plan = ts.isTaggedTemplateExpression(node) ? planAt.get(`${node.getStart(cell)}:${node.getEnd()}`) : undefined;
    if (plan && ts.isTaggedTemplateExpression(node)) {
      const site: InlineSite = { plan, context, lower: raw };
      const suffix = plan.explicitCaptures ? suffixWithCallOf(node) : undefined;
      if (suffix) sites.push({ start: node.getStart(cell), end: suffix.getEnd(),
        expression: () => explicitInlineCall(f, site, raw(node.tag), suffix.arguments[0], node.template) });
      else if (plan.explicitCaptures) {
        const withCall = ts.isCallExpression(node.tag) ? node.tag : undefined;
        sites.push({ start: node.getStart(cell), end: node.getEnd(), expression: () => explicitInlineCall(f,
          site, withCall && ts.isPropertyAccessExpression(withCall.expression) ? raw(withCall.expression.expression) : f.createIdentifier('nl'),
          withCall?.arguments[0], node.template) });
      } else sites.push({ start: node.getStart(cell), end: node.getEnd(), expression: () => plainInlineCall(f, site, raw(node.tag), node.template) });
    }
    ts.forEachChild(node, visit);
  };
  visit(cell);
  sites.sort((a, b) => a.start - b.start || b.end - a.end);
  return { code: render(0, text.length), plans: analysis.plans, diagnostics, usesNl };
}

export type ValueDeclarationLimits = {
  /** List items read to infer an item type (default 64). */
  maxItems?: number;
  /** Fields of one record read (default 64). */
  maxFields?: number;
  /** Nesting depth read; deeper values are `unknown[]` or `Record<string, unknown>` (default 8). */
  maxDepth?: number;
};

const RESERVED_WORDS = new Set(['break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do', 'else',
  'enum', 'export', 'extends', 'false', 'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'new', 'null', 'return',
  'super', 'switch', 'this', 'throw', 'true', 'try', 'typeof', 'var', 'void', 'while', 'with', 'yield', 'let', 'static', 'implements',
  'interface', 'package', 'private', 'protected', 'public', 'await']);
const TAGGED_GLOBALS: Record<string, string> = { Map: 'Map<any, any>', Set: 'Set<any>', WeakMap: 'WeakMap<any, any>', WeakSet: 'WeakSet<any>',
  Promise: 'Promise<any>', Array: 'any[]', Date: 'Date', RegExp: 'RegExp', Error: 'Error', Uint8Array: 'Uint8Array', ArrayBuffer: 'ArrayBuffer' };
const LIVE_TYPE = /Live<\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*>/g;

/** The part of a value that type inference reads: bounded in items, fields and depth, and safe for cyclic data. */
function typeSample(value: unknown, limits: Required<ValueDeclarationLimits>, depth: number, path: Set<object>): unknown {
  if (Array.isArray(value)) {
    if (depth >= limits.maxDepth || path.has(value)) return [];
    path.add(value);
    const items = value.slice(0, limits.maxItems).map(item => typeSample(item, limits, depth + 1, path));
    path.delete(value);
    return items;
  }
  if (isPlainRecord(value)) {
    if (depth >= limits.maxDepth || path.has(value)) return {};
    path.add(value);
    const fields = Object.fromEntries(Object.entries(value).slice(0, limits.maxFields)
      .map(([key, item]) => [key, typeSample(item, limits, depth + 1, path)]));
    path.delete(value);
    return fields;
  }
  return value;
}

/** Carried by identity, not as data: a function, a class instance, a built-in, a folder handle. */
const opaque = (value: unknown): boolean => typeof value === 'function' ||
  (value !== null && typeof value === 'object' && !Array.isArray(value) && !isPlainRecord(value));
const holdsOpaque = (value: unknown): boolean => opaque(value) || (Array.isArray(value) ? value.some(holdsOpaque) :
  isPlainRecord(value) && Object.values(value).some(holdsOpaque));

/**
 * TypeScript for a sampled value. Data is typed by natlang's own scope-value rule (`inferValueType`). That rule declares a
 * whole list or record opaque once anything in it is carried by identity; a cell wants the fields around a handle
 * typed, so such containers are typed field by field and only the handle itself is natlang's live type.
 */
function shape(sample: unknown): string {
  if (sample === undefined) return 'undefined';
  if (opaque(sample) || !holdsOpaque(sample)) {
    try {
      return inferValueType(sample).replace(LIVE_TYPE, (_, _label, kind, tag) =>
        kind === 'function' ? '(...args: any[]) => any' : kind === 'tag' ? TAGGED_GLOBALS[tag] ?? 'any' : 'any');
    } catch { return 'any'; }
  }
  const union = (types: string[]) => [...new Set(types)].join(' | ');
  if (Array.isArray(sample)) return `(${union(sample.map(shape))})[]`;
  const entries = Object.entries(sample as Record<string, unknown>);
  if (entries.length > 12 || entries.some(([key]) => !/^[A-Za-z_$][\w$]*$/.test(key)))
    return `Record<string, ${union(entries.map(([, item]) => shape(item)))}>`;
  return `{ ${entries.map(([key, item]) => item === undefined ? `${key}?: unknown` : `${key}: ${shape(item)}`).join('; ')} }`;
}

/**
 * Ambient declarations (`declare const name: Type;`) for the current values of host bindings, to pass as
 * `HostCellOptions.ambientDeclarations`. Types are the ones natlang gives a scope value when it opens a call
 * (`inferValueType`): records by their fields, lists by the union of their items', large keyed objects as
 * `Record<string, T>`. Large and deep values are read to bounded size, so the declaration stays short and a cyclic value
 * terminates. Functions and live objects are declared as natlang declares live values: functions callable with any
 * arguments, built-ins (`Date`, `Map`, ...) by name, other objects `any`; a record holding some keeps the types of its
 * other fields. Names that are not identifiers are left out.
 */
export function declarationsForValues(values: Record<string, unknown>, options: ValueDeclarationLimits = {}): string {
  const limits = { maxItems: options.maxItems ?? 64, maxFields: options.maxFields ?? 64, maxDepth: options.maxDepth ?? 8 };
  return Object.entries(values).filter(([name]) => /^[A-Za-z_$][\w$]*$/.test(name) && !RESERVED_WORDS.has(name))
    .map(([name, value]) => `declare const ${name}: ${shape(typeSample(value, limits, 0, new Set()))};\n`).join('');
}
