import { createRequire } from 'node:module';
import ts from 'typescript';
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import type { HostEvent } from './native/evaluator.js';
import { packageNameFromSpecifier } from './package-specifier.js';

export function findPackageWorkspace(start: string): string | undefined {
  let directory = resolve(start);
  for (;;) {
    if (existsSync(join(directory, 'package.json'))) return directory;
    const parent = dirname(directory);
    if (parent === directory) return undefined;
    directory = parent;
  }
}

/** Loads a package the way a module in `workspace` would import it. Package code has host authority. */
export class WorkspaceModules {
  readonly workspace: string;
  private readonly requirer: NodeJS.Require;
  constructor(workspace: string, private readonly observe: (event: HostEvent) => void = () => {}) {
    this.workspace = resolve(workspace);
    // Node's require also loads ES modules; the anchor file need not exist.
    this.requirer = createRequire(join(this.workspace, 'package.json'));
  }
  load(specifier: string): unknown {
    packageNameFromSpecifier(specifier);
    const loaded = this.requirer(specifier);
    this.observe({ operation: 'packages.import', specifier, workspace: this.workspace });
    return loaded;
  }
}

/**
 * What a package declares, as its TypeScript declarations give it: `pkg` lists its exports, `pkg.name` shows one export
 * with its doc comments. Undefined when the name is not an importable package with declarations.
 */
export function packageDeclaration(workspace: string, name: string): string | undefined {
  const dot = name.lastIndexOf('.');
  const tries: [string, string | undefined][] = [[name, undefined], ...(dot > 0 ? [[name.slice(0, dot), name.slice(dot + 1)] as [string, string]] : [])];
  const options: ts.CompilerOptions = { module: ts.ModuleKind.NodeNext, moduleResolution: ts.ModuleResolutionKind.NodeNext,
    target: ts.ScriptTarget.ES2022, noEmit: true, skipLibCheck: true, types: [] };
  for (const [specifier, member] of tries) {
    try { packageNameFromSpecifier(specifier); } catch { continue; }
    const resolved = ts.resolveModuleName(specifier, join(workspace, 'index.ts'), options, ts.sys).resolvedModule;
    if (!resolved?.resolvedFileName.endsWith('.d.ts') && !resolved?.resolvedFileName.endsWith('.d.mts') &&
        !resolved?.resolvedFileName.endsWith('.d.cts')) continue;
    const program = ts.createProgram([resolved.resolvedFileName], options), checker = program.getTypeChecker();
    const file = program.getSourceFile(resolved.resolvedFileName), symbol = file && checker.getSymbolAtLocation(file);
    if (!symbol) continue;
    const exports = checker.getExportsOfModule(symbol);
    const target = (item: ts.Symbol) => item.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(item) : item;
    if (member === undefined) {
      const lines = exports.map(item => {
        const type = checker.typeToString(checker.getTypeOfSymbolAtLocation(target(item), file!), undefined, ts.TypeFormatFlags.NoTruncation);
        return `  ${item.name}: ${type.length > 160 ? `${type.slice(0, 157)}...` : type};`;
      });
      return `declare module "${specifier}" {  // ${exports.length} exports; read_code("${specifier}.<name>") shows one with its docs\n${lines.join('\n')}\n}`;
    }
    const found = exports.find(item => item.name === member);
    if (!found) continue;
    const texts = (target(found).declarations ?? []).map(declaration => declaration.getFullText().replace(/^\s*\n/, '').trimEnd());
    return `// ${member}, from "${specifier}" (${resolved.resolvedFileName.split('/node_modules/').pop()})\n${texts.join('\n')}`;
  }
  return undefined;
}

