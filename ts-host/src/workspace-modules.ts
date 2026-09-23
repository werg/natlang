import { createRequire } from 'node:module';
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
