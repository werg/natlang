/**
 * pi's four tools (read, write, edit, bash) with pi's parameters and limits, plus the services a codemode script
 * can use. Paths are relative to the working directory.
 */
import { spawn } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync, existsSync, mkdtempSync, readdirSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, isAbsolute, join, relative, resolve } from 'node:path';

export const MAX_LINES = 2000, MAX_BYTES = 50 * 1024;

const fn = (name: string, description: string, properties: Record<string, unknown>, required: string[]) =>
  ({ type: 'function', function: { name, description, parameters: { type: 'object', properties, required } } });

/** The tool definitions the big model sees (OpenAI function format). */
export const TOOL_DEFINITIONS = [
  fn('read', `Read the contents of a file. Output is truncated to ${MAX_LINES} lines or ${MAX_BYTES / 1024}KB; use offset/limit for large files.`,
    { path: { type: 'string', description: 'Path to the file to read (relative or absolute)' },
      offset: { type: 'number', description: 'Line number to start reading from (1-indexed)' },
      limit: { type: 'number', description: 'Maximum number of lines to read' } }, ['path']),
  fn('write', "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
    { path: { type: 'string', description: 'Path to the file to write (relative or absolute)' },
      content: { type: 'string', description: 'Content to write to the file' } }, ['path', 'content']),
  fn('edit', 'Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit.',
    { path: { type: 'string', description: 'Path to the file to edit (relative or absolute)' },
      edits: { type: 'array', description: 'One or more targeted replacements, each matched against the original file.',
        items: { type: 'object', properties: { oldText: { type: 'string', description: 'Exact text for one targeted replacement; unique in the file.' },
          newText: { type: 'string', description: 'Replacement text for this targeted edit.' } }, required: ['oldText', 'newText'] } } }, ['path', 'edits']),
  fn('bash', `Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to the last ${MAX_LINES} lines or ${MAX_BYTES / 1024}KB; the full output is saved to a file.`,
    { command: { type: 'string', description: 'Shell command to execute' },
      timeout: { type: 'number', description: 'Timeout in seconds (optional, no default timeout)' } }, ['command']),
];

export const CODEMODE_DEFINITION = fn('codemode',
  'Run a TypeScript script (top-level await and return) in a sandbox and get back only what it prints (console.log) or returns. ' +
  'Use it to process many files or long outputs without reading them yourself. Loops are finite: for...of, counted for, array methods (no while). ' +
  "Bindings: shell.run(command) -> { output, exitCode }; files.read(path), files.write(path, text), files.list(dir?) -> paths. " +
  'nl runs a typed natural-language judgment on a small model: `const fixed = await nl<boolean>`Decide whether diff fixes the null check.`(diff)`; ' +
  'run many at once with Promise.all.',
  { script: { type: 'string', description: 'The TypeScript source' } }, ['script']);

const inside = (cwd: string, path: string) => isAbsolute(path) ? path : resolve(cwd, path);

/** Keep the first lines/bytes (read) or the last (bash). */
export function truncate(text: string, keep: 'head' | 'tail'): { text: string, truncated: boolean } {
  const ending = text.endsWith('\n') ? '\n' : '';
  let lines = (ending ? text.slice(0, -1) : text).split('\n'), truncated = false;
  if (lines.length > MAX_LINES) { lines = keep === 'head' ? lines.slice(0, MAX_LINES) : lines.slice(-MAX_LINES); truncated = true; }
  let out = lines.join('\n') + ending;
  if (Buffer.byteLength(out) > MAX_BYTES) { out = keep === 'head' ? out.slice(0, MAX_BYTES) : out.slice(-MAX_BYTES); truncated = true; }
  return { text: out, truncated };
}

export type ShellResult = { output: string, exitCode: number | null, timedOut: boolean, ms: number };
export function runShell(command: string, cwd: string, timeoutSeconds?: number, signal?: AbortSignal): Promise<ShellResult> {
  return new Promise(done => {
    const started = performance.now();
    const child = spawn('bash', ['-c', command], { cwd, stdio: ['ignore', 'pipe', 'pipe'], signal });
    let output = '', timedOut = false;
    const timer = timeoutSeconds ? setTimeout(() => { timedOut = true; child.kill('SIGKILL'); }, timeoutSeconds * 1000) : undefined;
    const take = (chunk: Buffer) => { output += chunk.toString('utf8'); if (output.length > 8 * MAX_BYTES) output = output.slice(-4 * MAX_BYTES); };
    child.stdout.on('data', take); child.stderr.on('data', take);
    child.on('error', error => { clearTimeout(timer); done({ output: `${output}${error.message}`, exitCode: null, timedOut, ms: performance.now() - started }); });
    child.on('close', code => { clearTimeout(timer); done({ output, exitCode: code, timedOut, ms: performance.now() - started }); });
  });
}

export type ToolOutcome = { text: string, ok: boolean, full?: string };

/** Carry out one of pi's tools. */
export async function runTool(name: string, args: Record<string, unknown>, cwd: string, signal?: AbortSignal): Promise<ToolOutcome> {
  try {
    if (name === 'read') {
      const path = inside(cwd, String(args.path));
      const lines = readFileSync(path, 'utf8').split('\n');
      const start = Math.max(1, Number(args.offset ?? 1));
      const selected = lines.slice(start - 1, args.limit ? start - 1 + Number(args.limit) : undefined).join('\n');
      const { text, truncated } = truncate(selected, 'head');
      return { ok: true, text: truncated ? `${text}\n\n[Truncated: use offset=${start + text.split('\n').length} to continue]` : text };
    }
    if (name === 'write') {
      const path = inside(cwd, String(args.path));
      mkdirSync(dirname(path), { recursive: true });
      writeFileSync(path, String(args.content ?? ''));
      return { ok: true, text: `Wrote ${Buffer.byteLength(String(args.content ?? ''))} bytes to ${args.path}` };
    }
    if (name === 'edit') {
      const path = inside(cwd, String(args.path));
      const original = readFileSync(path, 'utf8');
      const edits = (args.edits as { oldText: string, newText: string }[] | undefined) ?? [];
      if (!edits.length) return { ok: false, text: 'edit needs at least one edits[] entry' };
      const spans = edits.map(edit => {
        const at = original.indexOf(edit.oldText);
        if (!edit.oldText || at < 0) throw new Error(`Could not find the exact text in ${args.path}: ${JSON.stringify(edit.oldText.slice(0, 120))}`);
        if (original.indexOf(edit.oldText, at + 1) >= 0) throw new Error(`The text is not unique in ${args.path}; include more context: ${JSON.stringify(edit.oldText.slice(0, 120))}`);
        return { at, end: at + edit.oldText.length, newText: edit.newText };
      }).sort((a, b) => a.at - b.at);
      for (let i = 1; i < spans.length; i++) if (spans[i]!.at < spans[i - 1]!.end) throw new Error('Two edits overlap; merge them into one edit');
      let result = '', cursor = 0;
      for (const span of spans) { result += original.slice(cursor, span.at) + span.newText; cursor = span.end; }
      writeFileSync(path, result + original.slice(cursor));
      return { ok: true, text: `Edited ${args.path}: ${edits.length} replacement${edits.length > 1 ? 's' : ''}` };
    }
    if (name === 'bash') {
      const result = await runShell(String(args.command), cwd, args.timeout ? Number(args.timeout) : undefined, signal);
      const { text, truncated } = truncate(result.output, 'tail');
      let note = '';
      if (truncated) {
        const file = join(mkdtempSync(join(tmpdir(), 'pi-bash-')), 'output.txt');
        writeFileSync(file, result.output);
        note = `\n[Output truncated; full output: ${file}]`;
      }
      const status = result.timedOut ? `\n[Timed out after ${args.timeout}s]` : result.exitCode ? `\n[Exit code ${result.exitCode}]` : '';
      return { ok: !result.timedOut && result.exitCode === 0, text: `${text}${note}${status}` || '(no output)', full: result.output };
    }
    return { ok: false, text: `Unknown tool ${name}` };
  } catch (error) {
    return { ok: false, text: `Error: ${String((error as Error)?.message ?? error)}` };
  }
}

/** Every file of the project (git's view when it is a repository), for the scout. */
export async function projectFiles(cwd: string): Promise<string[]> {
  const listed = await runShell('git ls-files 2>/dev/null', cwd);
  if (listed.exitCode === 0 && listed.output.trim()) return listed.output.trim().split('\n').slice(0, 3000);
  const out: string[] = [];
  const walk = (dir: string, depth: number) => {
    if (depth > 6 || out.length >= 3000) return;
    for (const name of readdirSync(dir)) {
      if (name.startsWith('.') || name === 'node_modules' || name === 'dist') continue;
      const path = join(dir, name);
      if (statSync(path).isDirectory()) walk(path, depth + 1); else out.push(relative(cwd, path));
    }
  };
  walk(cwd, 0);
  return out;
}

/** What a codemode script can use; gate returns why a command may not run, or null. */
export function scriptServices(cwd: string, gate: (command: string) => Promise<string | null> = async () => null) {
  return {
    shell: { async run(command: string) {
      const refusal = await gate(command);
      if (refusal) throw new Error(refusal);
      const r = await runShell(command, cwd, 120);
      return { output: truncate(r.output, 'tail').text, exitCode: r.exitCode };
    } },
    files: {
      async read(path: string) { return readFileSync(inside(cwd, path), 'utf8'); },
      async write(path: string, text: string) { const target = inside(cwd, path); mkdirSync(dirname(target), { recursive: true }); writeFileSync(target, text); },
      async list(dir = '.') { return (await projectFiles(inside(cwd, dir))); },
      async exists(path: string) { return existsSync(inside(cwd, path)); },
    },
  };
}
export const SCRIPT_DECLARATIONS = {
  shell: '/** Run a bash command in the project; output is the last 2000 lines. */\nexport function run(command: string): Promise<{ output: string, exitCode: number | null }>;',
  files: '/** Read a project file. */\nexport function read(path: string): Promise<string>;\n/** Write a project file (parents created). */\nexport function write(path: string, text: string): Promise<void>;\n/** Project files under dir. */\nexport function list(dir?: string): Promise<string[]>;\nexport function exists(path: string): Promise<boolean>;',
};
