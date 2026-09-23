/**
 * `natlang check` and `natlang build`: whole-project analysis and lowering.
 *
 * - Every `.nl` function gets a generated `foo.d.nl.ts` so TypeScript (and plain editors) see typed imports.
 * - Callable folders (`foo/` beside `foo.nl`, and `natlang.d/`) are checked with the constrained-source
 *   policy and embedded as records; they execute at runtime as module instances.
 * - Ordinary project files are type-checked, `nl` expressions are planned against the checker, and the
 *   emitted JavaScript calls the runtime through `__natlang`.
 */
import ts from 'typescript';
import { analyzeInlineLambdas, spanOf, type InlineLambdaPlan, type NatlangDiagnostic } from './inline.js';
import { natlangTransformer } from './lower.js';
import { typeScriptText } from './eval-check.js';
import { INTRINSICS_FILE, NATLANG_COMPILE_VERSION, SURFACE_MODULE_FILE } from './intrinsics.js';
import { createNatlangCompilerHost } from './host.js';
import { compileModule } from '../runtime/modules.js';
import { findApplicationContext, loadCallableFolder, loadNamedFunction, NatlangSourceError,
  type ItemRecord, type NatlangRecord } from '../runtime/loader.js';
import type { SourceFiles } from '../runtime/loader.js';

/** File access for the project compiler: the real filesystem on Node, virtual files in browsers. */
export interface ProjectFiles {
  isFile(path: string): boolean;
  isDirectory(path: string): boolean;
  list(dir: string): string[];
  read(path: string): string;
  write?(path: string, text: string): void;
  /** Underlying compiler access for libraries and dependencies (Node: ts.sys). */
  compiler?: import('./host.js').CompilerFiles;
}

// POSIX path helpers (absolute paths); the Node adapter normalizes platform paths before calling in.
const join = (...parts: string[]) => normalize(parts.filter(Boolean).join('/'));
const normalize = (path: string) => {
  const absolute = path.startsWith('/'), out: string[] = [];
  for (const part of path.split('/')) { if (!part || part === '.') continue; if (part === '..') out.pop(); else out.push(part); }
  return (absolute ? '/' : '') + out.join('/');
};
const dirname = (path: string) => { const at = path.lastIndexOf('/'); return at <= 0 ? '/' : path.slice(0, at); };
const resolve = (...parts: string[]) => { let path = ''; for (const part of parts) path = part.startsWith('/') ? part : `${path}/${part}`; return normalize(path); };
const relative = (from: string, to: string) => {
  const a = normalize(from).split('/').filter(Boolean), b = normalize(to).split('/').filter(Boolean);
  let index = 0; while (index < a.length && a[index] === b[index]) index++;
  return [...a.slice(index).map(() => '..'), ...b.slice(index)].join('/');
};
const sep = '/';
const basename = (path: string, extension?: string) => { const base = path.slice(path.lastIndexOf('/') + 1);
  return extension && base.endsWith(extension) ? base.slice(0, -extension.length) : base; };
const extname = (path: string) => { const base = basename(path); const at = base.lastIndexOf('.'); return at > 0 ? base.slice(at) : ''; };

function sourceFiles(files: ProjectFiles, root: string): SourceFiles {
  return { join, dirname, basename, extname, isFile: path => files.isFile(path), isDirectory: path => files.isDirectory(path),
    read: path => files.read(path), list: path => files.list(path), relative: path => relative(root, path) };
}

export type BuildOptions = {
  /** Project directory or tsconfig path (absolute POSIX path when `files` is supplied). */
  project: string;
  /** File access; defaults to the Node filesystem (see `buildProject` in compiler/node-project.ts). */
  files: ProjectFiles;
  /** Emitted module format; browsers executing a virtual project use `commonjs`. */
  module?: 'esm' | 'commonjs';
  outDir?: string;
  target?: 'node' | 'browser';
  /** Specifier of the runtime package the emitted code imports (default: the one files import `nl` from). */
  runtimeSpecifier?: string;
  /** Write `.d.nl.ts` declarations and emitted JavaScript. */
  write?: boolean;
  /** Emit JavaScript (build) or only check. */
  emit?: boolean;
  /** Write generated `.d.nl.ts` files beside their sources (default true; read-only installed packages use false). */
  writeDeclarations?: boolean;
  /**
   * Bind the project to a specific runtime module: these specifiers type-check against `types` and are
   * emitted as `url`, so the compiled app shares one runtime instance with its launcher.
   */
  runtimeModule?: { specifiers: string[]; url: string; types: string; path?: string };
  /** Type these runtime specifiers against the built-in surface declarations (virtual projects). */
  surfaceSpecifiers?: string[];
  /** Type-check these specifiers against a runtime's declaration file when the project cannot resolve them. */
  runtimeTypes?: { specifiers: string[]; types: string };
  /** Type roots shipped with the runtime (Node's `@types`), used for `types` entries the project cannot resolve. */
  runtimeTypeRoots?: string[];
};

export type DefinitionManifest = { version: typeof NATLANG_COMPILE_VERSION;
  inline: { id: string; source: string; line: number; signature: string }[];
  named: { id: string; source: string; signature: string }[];
  sites: string[] };

export type BuildResult = { ok: boolean; diagnostics: NatlangDiagnostic[]; outputs: Record<string, string>;
  declarations: Record<string, string>; manifest: DefinitionManifest; outDir: string };

const SKIPPED_DIRS = new Set(['node_modules', 'dist', '.git', '.natlang']);

/** Directories that are callable folders: `natlang.d/`, and `foo/` beside `foo.nl`. */
function callableFolders(files: ProjectFiles, root: string): { named: string[]; contexts: string[]; folders: Set<string> } {
  const named: string[] = [], contexts: string[] = [], folders = new Set<string>();
  const mark = (dir: string) => {
    folders.add(dir);
    for (const entry of files.list(dir)) {
      const path = join(dir, entry);
      if (files.isDirectory(path)) mark(path);
    }
  };
  const walk = (dir: string) => {
    for (const entry of files.list(dir).sort()) {
      if (SKIPPED_DIRS.has(entry) || entry.startsWith('.')) continue;
      const path = join(dir, entry);
      if (files.isDirectory(path)) {
        if (folders.has(path)) continue;
        if (entry === 'natlang.d') { contexts.push(path); mark(path); continue; }
        if (files.isFile(`${path}.nl`)) { mark(path); continue; }
        walk(path);
      } else if (entry.endsWith('.nl')) named.push(path);
    }
  };
  walk(root);
  return { named, contexts, folders };
}

const aliasBlock = (types: Record<string, string>) => {
  const known = new Set(Object.keys(types));
  return Object.entries(types).map(([name, text]) => `type ${name} = ${typeScriptText(text, known)};`).join('\n');
};

/** TypeScript declaration text for an item and its callable-folder children. */
function itemType(record: ItemRecord, known: ReadonlySet<string>): string {
  const children = Object.entries(record.codebase).map(([name, child]) => `readonly ${name}: ${itemType(child, known)}`);
  const members = children.length ? ` & { ${children.join('; ')} }` : '';
  if (record.kind === 'namespace') return `{ ${children.join('; ')} }`;
  if (record.kind === 'module') {
    const exports = Object.entries(record.exports).filter(([name]) => name !== 'default').map(([name, spec]) =>
      spec.kind === 'function' ? `readonly ${name}: (${Object.entries(spec.args).map(([raw, type]) =>
        `${raw.replace(/\?$/, '')}${raw.endsWith('?') ? '?' : ''}: ${typeScriptText(type, known)}`).join(', ')}) => ` +
        `${spec.async ? `Promise<${typeScriptText(spec.returns, known)}>` : typeScriptText(spec.returns, known)}` :
        `readonly ${name}: ${spec.type ? typeScriptText(spec.type, known) : 'unknown'}`);
    const main = record.exports.default;
    const body = `{ ${[...exports, ...children].join('; ')} }`;
    if (main?.kind !== 'function') return body;
    const params = Object.entries(main.args).map(([raw, type]) => `${raw.replace(/\?$/, '')}${raw.endsWith('?') ? '?' : ''}: ${typeScriptText(type, known)}`);
    return `((${params.join(', ')}) => ${main.async ? `Promise<${typeScriptText(main.returns, known)}>` : typeScriptText(main.returns, known)}) & ${body}`;
  }
  const params = Object.entries(record.args).map(([raw, type]) => `${raw.replace(/\?$/, '')}${raw.endsWith('?') ? '?' : ''}: ${typeScriptText(type, known)}`);
  if (record.subtype === 'directory-reducer') params.unshift('folder: Folder');
  return `NatlangFunction<[${params.join(', ')}], ${typeScriptText(record.returns, known)}>${members}`;
}

/** Generated declaration for `foo.nl`, read by TypeScript as `foo.d.nl.ts`. */
export function natlangDeclaration(record: NatlangRecord, runtimeSpecifier: string): string {
  const collect: Record<string, string> = {};
  const gather = (item: ItemRecord) => {
    if (item.kind !== 'namespace') Object.assign(collect, item.types);
    for (const child of Object.values(item.codebase)) gather(child);
  };
  gather(record);
  const known = new Set(Object.keys(collect));
  return `// Generated by natlang from ${record.name}.nl; do not edit.\n` +
    `import type { NatlangFunction, Folder as FolderStore, FolderHandle } from ${JSON.stringify(runtimeSpecifier)};\n` +
    '/** A natlang `Folder` accepts a whole folder or a directory handle within one. */\ntype Folder = FolderStore | FolderHandle;\n' +
    `${aliasBlock(collect)}\ndeclare const fn: ${itemType(record, known)};\nexport default fn;\n`;
}

/** Declaration of `natlang:services` falls to the application; this supplies an empty default. */
const SERVICES_FALLBACK = '/__natlang__/services.d.ts';

function findTsconfig(files: ProjectFiles, project: string): { configPath?: string; root: string } {
  const absolute = normalize(project);
  if (files.isFile(absolute)) return { configPath: absolute, root: dirname(absolute) };
  const candidate = join(absolute, 'tsconfig.json');
  return files.isFile(candidate) ? { configPath: candidate, root: absolute } : { root: absolute };
}

/** Check and optionally compile a project. File access comes from `options.files`. */
export function compileProject(options: BuildOptions): BuildResult {
  const fs = options.files;
  const { configPath, root } = findTsconfig(fs, options.project);
  const diagnostics: NatlangDiagnostic[] = [];
  const files = sourceFiles(fs, root);
  const rel = (path: string) => relative(root, path);
  const problem = (file: string, message: string, code: NatlangDiagnostic['code'] = 'callable-scope') =>
    diagnostics.push({ file, start: 0, end: 0, line: 1, column: 1, code, message, severity: 'error' });

  // Callable folders and named functions.
  const layout = callableFolders(fs, root);
  const namedRecords = new Map<string, NatlangRecord>();
  for (const path of layout.named) {
    if ([...layout.folders].some(folder => path.startsWith(folder + sep))) continue;
    try { namedRecords.set(path, loadNamedFunction(path, files)); }
    catch (error) { problem(rel(path), error instanceof NatlangSourceError ? error.message : String(error)); }
  }
  const contextRecords = new Map<string, Record<string, ItemRecord>>();
  for (const dir of layout.contexts) {
    try { contextRecords.set(dir, loadCallableFolder(dir, files)); }
    catch (error) { problem(rel(dir), error instanceof NatlangSourceError ? error.message : String(error)); }
  }
  // Compile every callable-folder module now so errors surface at build time rather than first call.
  const checkModules = (level: Record<string, ItemRecord>) => {
    for (const item of Object.values(level)) {
      if (item.kind === 'module') {
        try { compileModule(item, level); }
        catch (error) { problem(item.source, error instanceof Error ? error.message : String(error)); }
      }
      checkModules(item.codebase);
    }
  };
  for (const record of namedRecords.values()) checkModules(record.codebase);
  for (const records of contextRecords.values()) checkModules(records);

  // TypeScript program over ordinary project files, with generated `.nl` declarations.
  const configHost: ts.ParseConfigFileHost = { useCaseSensitiveFileNames: true, getCurrentDirectory: () => root,
    fileExists: path => fs.isFile(path), readFile: path => fs.isFile(path) ? fs.read(path) : undefined,
    readDirectory: (dir, extensions, excludes, includes, depth) => fs.compiler ?
      ts.sys.readDirectory(dir, extensions, excludes, includes, depth) : walkTs(fs, dir),
    onUnRecoverableConfigFileDiagnostic: diagnostic => problem(rel(configPath!), ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'), 'typescript') };
  const parsed = configPath ? ts.getParsedCommandLineOfConfigFile(configPath, {}, configHost) : undefined;
  const compilerOptions: ts.CompilerOptions = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.NodeNext,
    moduleResolution: ts.ModuleResolutionKind.NodeNext, strict: true, skipLibCheck: true,
    ...(parsed?.options ?? {}), allowArbitraryExtensions: true, noEmit: false, declaration: false,
    ...(options.module === 'commonjs' ? { module: ts.ModuleKind.CommonJS, moduleResolution: ts.ModuleResolutionKind.Node10 } : {}) };
  const outDir = resolve(options.outDir ?? compilerOptions.outDir ?? join(root, 'dist'));
  compilerOptions.outDir = outDir;
  compilerOptions.rootDir ??= root;
  const inFolder = (path: string) => [...layout.folders].some(folder => path.startsWith(folder + sep) || path === folder);
  const rootNames = (parsed?.fileNames?.length ? parsed.fileNames : walkTs(fs, root)).filter(path => !inFolder(path) && !path.endsWith('.d.nl.ts'));
  const bound = options.runtimeModule;
  if (options.runtimeTypeRoots?.length) {
    const resolutionHost = ts.sys ?? { fileExists: (path: string) => fs.isFile(path), readFile: (path: string) => fs.isFile(path) ? fs.read(path) : undefined };
    const missing = (compilerOptions.types ?? ['node']).some(name =>
      !ts.resolveTypeReferenceDirective(name, join(root, 'index.ts'), compilerOptions, resolutionHost).resolvedTypeReferenceDirective);
    if (missing) compilerOptions.typeRoots = [...(compilerOptions.typeRoots ?? []), ...options.runtimeTypeRoots];
  }
  if (options.runtimeTypes) for (const specifier of options.runtimeTypes.specifiers) {
    const resolved = ts.resolveModuleName(specifier, join(root, 'index.ts'), compilerOptions, ts.sys ?? {
      fileExists: path => fs.isFile(path), readFile: path => fs.isFile(path) ? fs.read(path) : undefined });
    if (!resolved.resolvedModule) compilerOptions.paths = { ...(compilerOptions.paths ?? {}), [specifier]: [options.runtimeTypes.types] };
  }
  if (options.surfaceSpecifiers?.length) compilerOptions.paths = { ...(compilerOptions.paths ?? {}),
    ...Object.fromEntries(options.surfaceSpecifiers.map(specifier => [specifier, [SURFACE_MODULE_FILE]])) };
  if (bound) compilerOptions.paths = { ...(compilerOptions.paths ?? {}),
    ...Object.fromEntries(bound.specifiers.map(specifier => [specifier, [bound.types]])) };
  const runtimeSpecifier = bound?.url ?? options.runtimeSpecifier ?? detectRuntimeSpecifier(fs, rootNames) ?? '@natlang/node';
  const declarationSpecifier = bound?.specifiers[0] ?? runtimeSpecifier;
  const rewrite = new Map((bound?.specifiers ?? []).map(specifier => [specifier, bound!.url]));
  const declarations: Record<string, string> = {};
  const virtual = new Map<string, string>([[SERVICES_FALLBACK, "declare module 'natlang:services' { const services: Record<string, any>; export = services; }\n"]]);
  for (const [path, record] of namedRecords) {
    const declaration = natlangDeclaration(record, declarationSpecifier);
    const target = path.replace(/\.nl$/, '.d.nl.ts');
    declarations[target] = declaration;
    virtual.set(target, declaration);
  }
  const host = createNatlangCompilerHost({ options: compilerOptions, virtual, currentDirectory: root,
    files: fs.compiler ?? { readFile: path => fs.isFile(path) ? fs.read(path) : undefined, fileExists: path => fs.isFile(path),
      directoryExists: path => fs.isDirectory(path), getDirectories: path => fs.isDirectory(path) ?
        fs.list(path).filter(entry => fs.isDirectory(join(path, entry))) : [] } });
  const hasServicesDeclaration = rootNames.some(path => /declare\s+module\s+['"]natlang:services['"]/.test(fs.read(path)));
  const program = ts.createProgram({ rootNames: [INTRINSICS_FILE, ...(hasServicesDeclaration ? [] : [SERVICES_FALLBACK]), ...rootNames],
    options: compilerOptions, host });
  for (const diagnostic of ts.getPreEmitDiagnostics(program)) {
    if (diagnostic.category !== ts.DiagnosticCategory.Error) continue;
    const file = diagnostic.file;
    const position = file && diagnostic.start !== undefined ? file.getLineAndCharacterOfPosition(diagnostic.start) : { line: 0, character: 0 };
    diagnostics.push({ file: file ? rel(file.fileName) : '', start: diagnostic.start ?? 0, end: (diagnostic.start ?? 0) + (diagnostic.length ?? 0),
      line: position.line + 1, column: position.character + 1, code: 'typescript', severity: 'error',
      message: ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n') });
  }
  const sources = rootNames.map(path => program.getSourceFile(path)).filter((file): file is ts.SourceFile => !!file);
  const revision = (file: ts.SourceFile) => `${rel(file.fileName)}@${file.text.length}`;
  const plans = new Map<ts.SourceFile, InlineLambdaPlan[]>();
  for (const file of sources) {
    const analysis = analyzeInlineLambdas(program, [file], { displayPath: source => rel(source.fileName), sourceRevision: revision(file) });
    diagnostics.push(...analysis.diagnostics);
    plans.set(file, analysis.plans);
  }
  const manifest: DefinitionManifest = { version: NATLANG_COMPILE_VERSION,
    inline: [...plans.values()].flat().map(plan => ({ id: plan.definitionId, source: plan.sourceSpan.file, line: plan.sourceSpan.line,
      signature: `(${plan.parameters.map(parameter => `${parameter.name}: ${parameter.type.text}`).join(', ')}) => ${plan.returns.text}` })),
    named: [...namedRecords.values()].map(record => ({ id: record.id, source: record.source,
      signature: `(${Object.entries(record.args).map(([name, type]) => `${name}: ${type}`).join(', ')}) => ${record.returns}` })),
    sites: [] };
  const ok = !diagnostics.some(item => item.severity === 'error');
  const outputs: Record<string, string> = {};
  if (options.write !== false && options.writeDeclarations !== false && fs.write) for (const [path, text] of Object.entries(declarations))
    if (!fs.isFile(path) || fs.read(path) !== text) fs.write(path, text);
  if (ok && options.emit !== false) {
    const generated = new Map<string, { record: NatlangRecord; commonjs: boolean }>();
    const natlangImports = (file: ts.SourceFile) => {
      const found = new Set<string>();
      for (const statement of file.statements) {
        if (!ts.isImportDeclaration(statement) || !ts.isStringLiteral(statement.moduleSpecifier)) continue;
        const specifier = statement.moduleSpecifier.text;
        if (!specifier.endsWith('.nl')) continue;
        const target = resolve(dirname(file.fileName), specifier);
        const record = namedRecords.get(target);
        if (!record) { problem(rel(file.fileName), `cannot import ${specifier}: no such natlang function`); continue; }
        found.add(specifier);
        generated.set(join(outDir, relative(compilerOptions.rootDir!, target)) + '.js',
          { record, commonjs: file.impliedNodeFormat === ts.ModuleKind.CommonJS || compilerOptions.module === ts.ModuleKind.CommonJS });
      }
      return found;
    };
    const specifierFor = (commonjs: boolean) => bound ? (commonjs && bound.path ? bound.path : bound.url) : runtimeSpecifier;
    const byFile = new Map(sources.map(file => [file.fileName, file]));
    const transformer: ts.TransformerFactory<ts.SourceFile> = context => file => {
      const source = byFile.get(file.fileName);
      if (!source) return file;
      const contextDir = findApplicationContext(dirname(file.fileName), files);
      const filePlans = plans.get(source) ?? [];
      const commonjs = source.impliedNodeFormat === ts.ModuleKind.CommonJS || compilerOptions.module === ts.ModuleKind.CommonJS;
      const rewriteFor = new Map([...rewrite.keys()].map(specifier => [specifier, specifierFor(commonjs)]));
      return natlangTransformer({ plans: new Map(filePlans.map(plan => [`${plan.sourceSpan.start}:${plan.sourceSpan.end}`, plan])),
        checker: program.getTypeChecker(), runtime: '__natlang',
        context: contextDir && contextRecords.has(contextDir) ? JSON.stringify(contextRecords.get(contextDir)) : undefined,
        constrained: false, guardPrefix: rel(file.fileName), modulePath: rel(file.fileName), browser: options.target === 'browser',
        module: { natlangImports: natlangImports(source), rewrite: rewriteFor } })(context)(file);
    };
    const emit = (fileName: string, text: string) => {
      outputs[fileName] = text;
      if (options.write !== false) fs.write?.(fileName, text);
    };
    const emitted = program.emit(undefined, emit, undefined, false, { before: [transformer] });
    for (const [path, { record, commonjs }] of generated) {
      const specifier = JSON.stringify(specifierFor(commonjs)), json = JSON.stringify(record);
      emit(path, commonjs ?
        `"use strict";\nObject.defineProperty(exports, "__esModule", { value: true });\nexports.default = require(${specifier}).__natlang.named(${JSON.stringify(record.name)}, ${json});\n` :
        `import { __natlang } from ${specifier};\nexport default __natlang.named(${JSON.stringify(record.name)}, ${json});\n`);
    }
    for (const diagnostic of emitted.diagnostics) if (diagnostic.category === ts.DiagnosticCategory.Error)
      problem(diagnostic.file ? rel(diagnostic.file.fileName) : '', ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'), 'typescript');
    if (options.write !== false) fs.write?.(join(outDir, 'natlang-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  }
  diagnostics.sort((a, b) => a.file.localeCompare(b.file) || a.start - b.start);
  return { ok: !diagnostics.some(item => item.severity === 'error'), diagnostics, outputs, declarations, manifest, outDir };
}

function walkTs(fs: ProjectFiles, root: string): string[] {
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const entry of fs.list(dir)) {
      if (SKIPPED_DIRS.has(entry) || entry.startsWith('.')) continue;
      const path = join(dir, entry);
      if (fs.isDirectory(path)) walk(path);
      else if (/\.(ts|tsx|mts)$/.test(entry) && !entry.endsWith('.d.ts')) out.push(path);
    }
  };
  walk(root);
  return out;
}

function detectRuntimeSpecifier(fs: ProjectFiles, files: string[]): string | undefined {
  for (const path of files) {
    const match = /import\s*\{[^}]*\b(?:nl|iterateOn|createNatlangRuntime)\b[^}]*\}\s*from\s*['"]([^'"]+)['"]/.exec(fs.read(path));
    if (match) return match[1];
  }
  return;
}

/** Format diagnostics for a terminal. */
export function formatDiagnostics(diagnostics: readonly NatlangDiagnostic[]): string {
  return diagnostics.map(item => `${item.file}:${item.line}:${item.column} ${item.severity} ${item.code}: ${item.message}`).join('\n');
}
export { spanOf };
