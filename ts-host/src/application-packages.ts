import { execFile } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { promisify } from 'node:util';
import type { HostEvent } from './native/evaluator.js';

const execute = promisify(execFile);
const queues = new Map<string, Promise<unknown>>();
const hash = (path: string) => existsSync(path) ? createHash('sha256').update(readFileSync(path)).digest('hex') : null;

/** Shared by live inference and corpus replay so capability instructions do not drift. */
export function applicationCapabilityPrompt(capabilities: { allowModules: boolean; allowNetwork: boolean }): string {
  return (capabilities.allowModules ?
    '\nApplication packages are resolved from the configured app workspace. In eval, use await installPackages(["package@version"]) to update package.json and package-lock.json. Imports never install packages implicitly. Static imports and await import("package") are supported; imported bindings are local to the current eval, so re-import in later calls. Relative imports resolve from the application root. Package installation has external effects that are not rolled back on eval failure.\n' : '') +
    (capabilities.allowNetwork ? '\nNetwork access is available through fetch; consume responses into portable values before returning. HTTP response handles are local to the current eval.\n' : '');
}

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
