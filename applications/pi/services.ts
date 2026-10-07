/**
 * The outside world pi works in, as natlang services: a shell, the file system and the user. Everything pi does with
 * them is natural language (pi.nl and its folder); these only carry out what they are asked.
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, isAbsolute, resolve } from 'node:path';
import { projectFiles, runShell } from './tools.js';

export function piServices(cwd: string, confirm: (question: string) => Promise<boolean> = async () => false) {
  const at = (path: string) => isAbsolute(path) ? path : resolve(cwd, path.replace(/^~(?=\/|$)/, homedir()));
  return {
    shell: { async run(command: string, timeout?: number) {
      const result = await runShell(command, cwd, timeout, undefined, 64 << 20);
      return { output: result.output, exitCode: result.exitCode, timedOut: result.timedOut };
    } },
    files: {
      async read(path: string) { try { return readFileSync(at(path), 'utf8'); } catch { return null; } },
      async write(path: string, text: string) { mkdirSync(dirname(at(path)), { recursive: true }); writeFileSync(at(path), text); },
      async append(path: string, text: string) { mkdirSync(dirname(at(path)), { recursive: true }); appendFileSync(at(path), text); },
      async exists(path: string) { return existsSync(at(path)); },
      async list(dir = '.') { return projectFiles(at(dir)); },
      async home() { return homedir(); },
    },
    user: { confirm },
  };
}

export const PI_SERVICE_DECLARATIONS = {
  shell: `/** Run a command with bash in the project directory: everything it printed (stdout and stderr together), its exit
 * code (null when it was killed) and whether it ran past timeout seconds (no limit without one). */
export function run(command: string, timeout?: number): Promise<{ output: string, exitCode: number | null, timedOut: boolean }>;`,
  files: `/** Paths are relative to the project directory, or absolute; ~ is the home directory. */
/** A file's text, or null when there is none. */
export function read(path: string): Promise<string | null>;
/** Replace a file's text, creating missing directories. */
export function write(path: string, text: string): Promise<void>;
/** Add text at the end of a file, creating it (and missing directories) when needed. */
export function append(path: string, text: string): Promise<void>;
export function exists(path: string): Promise<boolean>;
/** The files under dir (default the project), relative to it: git's list in a repository, else a walk that skips hidden directories, node_modules and dist. */
export function list(dir?: string): Promise<string[]>;
/** The user's home directory. */
export function home(): Promise<string>;`,
  user: `/** Ask the user a yes-or-no question; false when no one can answer. */
export function confirm(question: string): Promise<boolean>;`,
};
