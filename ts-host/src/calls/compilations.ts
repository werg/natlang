/**
 * Compilations at runtime (§7.1, §7.3): a stored `cases.ts` is loaded as callable-folder TypeScript bound to the
 * definition's own context, so it compiles under the same rules, imports the same items and reaches the same
 * services as the function. Loaded compilations are cached per process and reloaded when the store's version moves.
 */
import ts from 'typescript';
import { hexDigest } from '../native/hash.js';
import { parseModule, PATH_ONLY, type ItemRecord, type ModuleRecord } from '../runtime/loader.js';
import { moduleInstance } from '../runtime/modules.js';
import type { CallStoreLike } from './recorder.js';
import type { CaseTier } from './types.js';

/** One crisp case of a compilation, ready to run. */
export type LoadedCase = { hash: string; position: number; tier: CaseTier;
  when: (args: Record<string, unknown>) => unknown; run: (args: Record<string, unknown>) => unknown;
  /** The `natlang:services` imports the case's own text uses; it is admitted only where the task provides them all. */
  services: string[] };
export type LoadedCompilation = { id: string; cases: LoadedCase[] };

/** The source text of each element of `export const cases = [...]`, in order. */
export function caseSources(text: string): string[] {
  const file = ts.createSourceFile('cases.ts', text, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  for (const statement of file.statements) {
    if (!ts.isVariableStatement(statement) || !ts.getModifiers(statement)?.some(item => item.kind === ts.SyntaxKind.ExportKeyword)) continue;
    for (const declaration of statement.declarationList.declarations) {
      if (!ts.isIdentifier(declaration.name) || declaration.name.text !== 'cases' || !declaration.initializer) continue;
      let initializer: ts.Expression = declaration.initializer;
      while (ts.isAsExpression(initializer) || ts.isSatisfiesExpression(initializer) || ts.isParenthesizedExpression(initializer))
        initializer = initializer.expression;
      if (!ts.isArrayLiteralExpression(initializer)) throw new Error('cases.ts: `export const cases` must be an array literal of { when, run } objects');
      return initializer.elements.map(element => element.getText(file));
    }
  }
  throw new Error('cases.ts must export `const cases = [{ when, run }, ...]`');
}

/** The services each case of a cases file uses: names imported from `natlang:services` that its text mentions. */
export function caseServices(text: string): string[][] {
  const file = ts.createSourceFile('cases.ts', text, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  const imported = new Map<string, string>();
  for (const statement of file.statements)
    if (ts.isImportDeclaration(statement) && ts.isStringLiteral(statement.moduleSpecifier) && statement.moduleSpecifier.text === 'natlang:services') {
      const bindings = statement.importClause?.namedBindings;
      if (bindings && ts.isNamedImports(bindings))
        for (const element of bindings.elements) imported.set(element.name.text, (element.propertyName ?? element.name).text);
    }
  return caseSources(text).map(source => [...imported].filter(([local]) => new RegExp(`(?<![\\w$.])${local.replace(/\$/g, '\\$')}(?![\\w$])`).test(source))
    .map(([, name]) => name));
}

/** Content hashes of a cases file's cases: a case keeps its tier and counts while its text is unchanged. */
export function caseHashes(text: string): string[] {
  return caseSources(text).map(source => hexDigest(source.replace(/\s+/g, ' ').trim()).slice(0, 24));
}

/** Load a cases file in a definition's context. Throws a source error when it does not compile or is malformed. */
export function loadCases(id: string, text: string, codebase: Record<string, unknown>, types: Record<string, string>): LoadedCompilation['cases'] {
  const record: ModuleRecord = { ...parseModule(`/__natlang__/compilations/${id}/cases.ts`, text, types, PATH_ONLY), id: `compilation:${id}` };
  const exports = moduleInstance(record, codebase as Record<string, ItemRecord>);
  const cases = exports.cases;
  if (!Array.isArray(cases)) throw new Error(`compilation ${id}: cases.ts does not export an array named cases`);
  const hashes = caseHashes(text), services = caseServices(text);
  if (hashes.length !== cases.length) throw new Error(`compilation ${id}: cases array has ${cases.length} entries, source has ${hashes.length}`);
  return cases.map((item, position) => {
    if (!item || typeof item.when !== 'function' || typeof item.run !== 'function')
      throw new Error(`compilation ${id}: case ${position} needs a when(args) guard and a run(args) body`);
    return { hash: hashes[position]!, position, tier: 'shadow' as CaseTier, when: item.when, run: item.run, services: services[position] ?? [] };
  });
}

/** The process's view of stored compilations. */
export class CompilationCache {
  private version = -1;
  private checked = 0;
  private readonly loaded = new Map<string, LoadedCompilation | null>();
  private readonly failures = new Set<string>();
  constructor(readonly store: CallStoreLike) {}

  /** The current compilation for a definition revision whose context interface matches, or undefined. */
  get(definitionKey: string, interfaceHash: string, codebase: Record<string, unknown>, types: Record<string, string>): LoadedCompilation | undefined {
    if (Date.now() - this.checked > 2000) {
      this.checked = Date.now();
      const version = this.store.version();
      if (version !== this.version) { this.version = version; this.loaded.clear(); }
    }
    const key = `${definitionKey}#${interfaceHash}`;
    if (this.loaded.has(key)) return this.loaded.get(key) ?? undefined;
    let result: LoadedCompilation | null = null;
    const row = this.store.currentCompilation(definitionKey);
    if (row && row.interface_hash === interfaceHash && row.files['cases.ts']) {
      try {
        const cases = loadCases(row.id, row.files['cases.ts'], codebase, types);
        const tiers = new Map(row.cases.map(item => [`${item.position}:${item.hash}`, item.tier]));
        result = { id: row.id, cases: cases.map(item => ({ ...item, tier: tiers.get(`${item.position}:${item.hash}`) ?? 'disabled' })) };
      } catch (error) {
        if (!this.failures.has(row.id)) {
          this.failures.add(row.id);
          console.warn(`natlang: compilation ${row.id} for ${row.definition_name} is not used: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
    }
    this.loaded.set(key, result);
    return result ?? undefined;
  }
  /** Forget loaded compilations (after this process saved one). */
  invalidate(): void { this.loaded.clear(); this.version = -1; this.checked = 0; }
}
