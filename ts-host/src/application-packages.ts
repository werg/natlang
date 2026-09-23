import { execFile } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { promisify } from 'node:util';
import type { HostEvent } from './native/evaluator.js';
export { applicationCapabilityPrompt } from './application-capabilities.js';

const execute = promisify(execFile);
const queues = new Map<string, Promise<unknown>>();
const hash = (path: string) => existsSync(path) ? createHash('sha256').update(readFileSync(path)).digest('hex') : null;

/** Application-scoped npm state. This runs with host authority, NOT in a security sandbox. */
export class ApplicationPackages {
  readonly workspace: string;
  private loader?: Promise<{ load: (specifier: string) => Promise<unknown> }>;
  constructor(workspace: string, private readonly observe: (event: HostEvent) => void = () => {}) {
    this.workspace = resolve(workspace);
    this.manifest();
  }
  private manifest(): { packageManager?: string } {
    const path = join(this.workspace, 'package.json');
    if (!existsSync(path)) throw new Error(`Application workspace requires package.json: ${this.workspace}`);
    const value = JSON.parse(readFileSync(path, 'utf8'));
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid application package.json');
    return value;
  }
  private dependencyNames(): string[] {
    const manifest = JSON.parse(readFileSync(join(this.workspace, 'package.json'), 'utf8')) as Record<string, unknown>;
    const names = new Set<string>();
    for (const section of ['dependencies', 'devDependencies', 'optionalDependencies', 'peerDependencies']) {
      const values = manifest[section];
      if (values && typeof values === 'object' && !Array.isArray(values))
        for (const name of Object.keys(values)) names.add(name);
    }
    return [...names].sort((a, b) => a.localeCompare(b));
  }
  /** Directly declared dependencies with an installed package directory resolvable from this workspace. */
  listAvailableDependencies(): string[] {
    const available = (name: string) => {
      let directory = this.workspace;
      for (;;) {
        if (existsSync(join(directory, 'node_modules', name))) return true;
        const parent = dirname(directory);
        if (parent === directory) return false;
        directory = parent;
      }
    };
    return this.dependencyNames().filter(available);
  }
  /** Direct package.json dependencies, including declarations that still need installation. */
  listDeclaredDependencies(): string[] { return this.dependencyNames(); }
  async installPackages(specifiers: string[] = [], frozen = false): Promise<{ stdout: string; stderr: string; lockfileSha256: string | null }> {
    if (!Array.isArray(specifiers) || specifiers.some(s => typeof s !== 'string' || !s.trim() || s.startsWith('-')))
      throw new Error('Package installation requires package specifiers, not npm options');
    const manager = this.manifest().packageManager;
    if (manager && !manager.startsWith('npm@')) throw new Error(`Package manager ${manager} is not implemented; refusing to change its lockfile with npm`);
    const prior = queues.get(this.workspace) ?? Promise.resolve();
    const job = prior.catch(() => {}).then(async () => {
      const command = frozen ? 'ci' : 'install';
      this.observe({ operation: 'packages.install', workspace: this.workspace, specifiers, command, status: 'started' });
      try {
        const result = await execute('npm', [command, '--no-audit', '--no-fund', ...specifiers], {
          cwd: this.workspace, timeout: 120000, maxBuffer: 4 * 1024 * 1024,
        });
        const lockfileSha256 = hash(join(this.workspace, 'package-lock.json'));
        this.observe({ operation: 'packages.install', workspace: this.workspace, specifiers, command, status: 'completed',
          packageJsonSha256: hash(join(this.workspace, 'package.json')), lockfileSha256 });
        return { stdout: result.stdout, stderr: result.stderr, lockfileSha256 };
      } catch (error) {
        this.observe({ operation: 'packages.install', workspace: this.workspace, specifiers, command, status: 'failed' });
        throw error;
      }
    });
    queues.set(this.workspace, job);
    try { return await job; } finally { if (queues.get(this.workspace) === job) queues.delete(this.workspace); }
  }
  prepareDependencies(): ReturnType<ApplicationPackages['installPackages']> {
    return this.installPackages([], existsSync(join(this.workspace, 'package-lock.json')));
  }
  async importModule(specifier: string): Promise<unknown> {
    if (typeof specifier !== 'string' || !specifier) throw new Error('Module specifier must be a nonempty string');
    // A real ESM loader rooted below the app preserves Node's import export-condition semantics.
    // Imports never implicitly install a package. Node builtins and installed package code have host authority.
    this.loader ??= (async () => {
      const directory = join(this.workspace, '.natlang'); mkdirSync(directory, { recursive: true });
      const path = join(directory, `module-loader-${randomUUID()}.mjs`);
      writeFileSync(path, 'export const load = specifier => import(specifier);\n', { flag: 'wx' });
      return import(pathToFileURL(path).href) as Promise<{ load: (specifier: string) => Promise<unknown> }>;
    })();
    const resolved = specifier.startsWith('.') || specifier.startsWith('/')
      ? pathToFileURL(resolve(this.workspace, specifier)).href : specifier;
    const result = await (await this.loader).load(resolved);
    this.observe({ operation: 'packages.import', specifier, workspace: this.workspace });
    return result;
  }
}
