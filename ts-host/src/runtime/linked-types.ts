import ts from 'typescript';
import {sourceLoadMemo} from './source-session.js';
import { hexDigest } from '../native/hash.js';
import { readTypeAliases } from '../native/type-aliases.js';
import type { SourceFiles } from './loader.js';

/** Resolve portable types in their declaring module before flattening them for Natlang's type environment. */
export function readTypesFile(path: string, files: SourceFiles): Record<string, string> {
  return {...sourceLoadMemo(files,'types:'+path,()=>readTypesFileInSession(path,files))};
}
function readTypesFileInSession(path: string, files: SourceFiles): Record<string, string> {
  type Alias = { module: TypeModule; name: string; body: string };
  type Binding = Alias | { module: TypeModule; from: string };
  type TypeModule = {
    path: string;
    local: Map<string, Binding>;
    exported: Map<string, Binding>;
  };
  const modules = new Map<string, TypeModule>();

  function readModule(filePath: string): TypeModule {
    const existing = modules.get(filePath);
    if (existing) return existing;
    const module: TypeModule = { path: filePath, local: new Map(), exported: new Map() };
    modules.set(filePath, module);
    const text = files.read(filePath);
    const aliases = Object.entries(readTypeAliases(text));
    for (const [name, body] of aliases) {
      const alias = { module, name, body };
      module.local.set(name, alias);
      module.exported.set(name, alias);
    }
    if (!/\b(?:import|export)\b[^;]*\bfrom\b/u.test(text)) return module;

    const source = ts.createSourceFile('types.ts', text, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
    for (const statement of source.statements) {
      if (!ts.isImportDeclaration(statement) && !ts.isExportDeclaration(statement)) continue;
      const specifier = statement.moduleSpecifier;
      const clause = ts.isImportDeclaration(statement) ? statement.importClause?.namedBindings : statement.exportClause;
      if (specifier && (!ts.isStringLiteral(specifier) || !specifier.text.startsWith('.'))) continue;
      let target = module;
      if (specifier && ts.isStringLiteral(specifier)) {
        const base = files.join(files.dirname(filePath), specifier.text.replace(/\.(?:[cm]?js|ts)$/u, ''));
        const candidate = [`${base}.ts`, files.join(base, 'index.ts')].find((entry) => files.isFile(entry));
        if (!candidate) continue;
        target = readModule(candidate);
      }
      if (clause && (ts.isNamedImports(clause) || ts.isNamedExports(clause))) {
        const bindings = ts.isNamedImports(clause) ? module.local : module.exported;
        for (const element of clause.elements) {
          const from = (element.propertyName ?? element.name).text;
          if (!bindings.has(element.name.text)) bindings.set(element.name.text, { module: target, from });
        }
      } else if (!clause && specifier && ts.isExportDeclaration(statement)) {
        for (const [name, binding] of target.exported) {
          if (!module.exported.has(name)) module.exported.set(name, binding);
        }
      }
    }
    return module;
  }

  function resolve(binding: Binding | undefined): Alias | undefined {
    const seen = new Set<Binding>();
    while (binding && !('body' in binding)) {
      if (seen.has(binding)) return;
      seen.add(binding);
      binding = binding.module.exported.get(binding.from) ?? binding.module.local.get(binding.from);
    }
    return binding;
  }

  const root = readModule(path);
  const visible = new Map([...root.exported, ...root.local]);
  const names = new Map<Alias, string>();
  for (const [name, binding] of visible) {
    const alias = resolve(binding);
    if (alias && (!names.has(alias) || name === alias.name)) names.set(alias, name);
  }
  const result = new Map<string, string>();
  const rewritten = new Map<Alias, string>();

  function render(alias: Alias): string {
    const cached = rewritten.get(alias);
    if (cached !== undefined) return cached;
    const prefix = 'type Linked = ';
    const source = ts.createSourceFile('types.ts', prefix + alias.body, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
    const replacements: { start: number; end: number; name: string }[] = [];
    function visit(node: ts.Node): void {
      if (ts.isTypeReferenceNode(node) && ts.isIdentifier(node.typeName)) {
        const name = node.typeName.text;
        const dependency = resolve(alias.module.local.get(name) ?? alias.module.exported.get(name));
        if (dependency) {
          if (!names.has(dependency)) {
            const id = hexDigest(files.relative?.(dependency.module.path) ?? dependency.module.path).slice(0, 12);
            const stem = `_Linked_${dependency.name}_${id}`;
            let unique = stem;
            for (let index = 1; visible.has(unique); index++) unique = `${stem}_${index}`;
            names.set(dependency, unique);
            visible.set(unique, dependency);
          }
          const linked = names.get(dependency)!;
          if (name !== linked) replacements.push({
            start: node.typeName.getStart(source) - prefix.length, end: node.typeName.end - prefix.length, name: linked,
          });
        }
      }
      ts.forEachChild(node, visit);
    }
    ts.forEachChild(source, visit);
    let body = alias.body;
    for (const { start, end, name } of replacements.reverse()) body = body.slice(0, start) + name + body.slice(end);
    rewritten.set(alias, body);
    return body;
  }
  // render appends private aliases reached from each body. Map iteration visits those new entries too.
  for (const [name, binding] of visible) {
    const alias = resolve(binding);
    if (alias) result.set(name, render(alias));
  }
  return Object.fromEntries(result);
}
