/**
 * What every natlang command shares and what `natlang --version` and `natlang help` need, kept free of the runtime so
 * those answer at once: the version, the argument parser, exit codes and plain output.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { closestName } from '../compiler/suggest.js';

/** The version of the @natlang/node package this CLI runs from: its package.json, or the checkout's npm-packages/node. */
function packageVersion(): string {
  for (const path of ['../../package.json', '../../../npm-packages/node/package.json']) {
    try {
      const manifest = JSON.parse(readFileSync(fileURLToPath(new URL(path, import.meta.url)), 'utf8')) as { name?: string; version?: string };
      if (manifest.name === '@natlang/node' && manifest.version) return manifest.version;
    } catch { /* not this layout */ }
  }
  throw new Error('natlang cannot find the package.json of @natlang/node; reinstall natlang');
}
export const NATLANG_CLI_VERSION = packageVersion();

/**
 * Exit codes of check, call, test and run: 1 the program is wrong (diagnostics, failed examples, a call that failed
 * for model-work reasons), 2 a usage error (flags, arguments, paths), 3 the model or runtime is unavailable.
 */
export const EXIT = { failed: 1, usage: 2, unavailable: 3 } as const;
export class CliError extends Error {
  constructor(message: string, readonly exitCode: number, readonly next?: string) { super(message); this.name = 'CliError'; }
}
export const usageError = (message: string) => new CliError(message, EXIT.usage);

export type Parsed = { words: string[]; options: Map<string, string | true>; rest: string[] };
const BOOLEAN = new Set(['--json', '--plain', '--no-color', '--help', '-h', '--version', '--yes', '--refresh', '--lines', '--jsonl',
  '--filter', '--no-adaptation', '--no-weights', '--fusion', '--default', '--warm', '--open', '--cuda', '--no-cuda', '--claude', '--codex',
  '--manifest']);
const VALUED = new Set(['--progress', '--target', '--export', '--profile', '--provider', '--model', '--escalate', '--workspace', '--state',
  '--traces', '--store', '--adaptation', '--inputs', '--trace', '--returns', '--files', '--jobs', '--out', '--fusion-planner', '--program',
  '--native', '--device', '--port', '--gguf', '--dir', '--host', '--root']);

const unknownOption = (name: string, known: Iterable<string>, command?: string) => {
  const near = closestName(name, known);
  return usageError(`unknown option ${name}${command ? ` for natlang ${command}` : ''}${near ? ` (did you mean ${near}?)` : ''}; ` +
    `natlang help${command ? ` ${command}` : ''} lists the options`);
};

export function parseArgs(args: string[]): Parsed {
  const words: string[] = [], options = new Map<string, string | true>(), rest: string[] = [];
  const set = (name: string, value: string | true) => {
    if (options.has(name)) throw usageError(`duplicate option ${name}`);
    options.set(name, value);
  };
  let separated = false;
  for (let index = 0; index < args.length; index++) {
    const value = args[index]!;
    if (separated) { rest.push(value); continue; }
    if (value === '--') { separated = true; continue; }
    if (!value.startsWith('--') && value !== '-h') { words.push(value); continue; }
    const equal = value.indexOf('='), name = equal >= 0 ? value.slice(0, equal) : value;
    // A name is checked before it can take the next word as its value, so a typo is reported as one.
    if (!BOOLEAN.has(name) && !VALUED.has(name)) throw unknownOption(name, [...BOOLEAN, ...VALUED], words[0]);
    if (BOOLEAN.has(name)) {
      if (equal >= 0) throw usageError(`${name} does not take a value`);
      set(name, true);
    } else if (equal >= 0) set(name, value.slice(equal + 1));
    else {
      const next = args[++index];
      if (next === undefined) throw usageError(`${name} needs a value`);
      set(name, next);
    }
  }
  return { words, options, rest };
}

export const option = (parsed: Parsed, name: string): string | undefined => {
  const value = parsed.options.get(name); return typeof value === 'string' ? value : undefined;
};
export function numericOption(parsed: Parsed, name: string, minimum: number): number | undefined {
  const raw = option(parsed, name);
  if (raw === undefined) return;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum) throw usageError(`${name} must be an integer >= ${minimum}`);
  return value;
}
export function acceptOptions(parsed: Parsed, allowed: readonly string[]): void {
  const accepted = new Set([...allowed, '--help', '-h', '--version']);
  const unknown = [...parsed.options.keys()].find(name => !accepted.has(name));
  if (unknown) throw unknownOption(unknown, allowed, parsed.words[0]);
}
export function noTrailingArguments(parsed: Parsed): void {
  if (parsed.rest.length) throw usageError('arguments after -- are only valid for natlang run');
}

export function output(value: unknown, json: boolean): void {
  process.stdout.write(json ? JSON.stringify(value, null, 2) + '\n' : typeof value === 'string' ? value + '\n' :
    Object.entries(value as Record<string, unknown>).map(([key, item]) => `${key}: ${text(item)}`).join('\n') + '\n');
}
/** One line of plain output: scalars as text, lists joined, records as `key=value` pairs. */
function text(item: unknown): string {
  if (Array.isArray(item)) return item.map(text).join(', ') || '(none)';
  if (item && typeof item === 'object') {
    const entries = Object.entries(item);
    return entries.length ? entries.map(([key, child]) => `${key}=${typeof child === 'object' && child ? JSON.stringify(child) : String(child)}`).join(', ') : '(none)';
  }
  return String(item);
}
