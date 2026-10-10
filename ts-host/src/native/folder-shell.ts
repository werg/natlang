/** Bash over the same Folder overlay used by file tools and reducers. */
import { Bash, defineCommand, getCommandNames,
  type IFileSystem, type FsStat, type FileContent, type BufferEncoding,
  type RmOptions, type CpOptions, type MkdirOptions } from 'just-bash';
import { Folder } from './scoped-fs.js';
import { runFolderPython, type PythonHost } from './folder-python.js';
import { executeNatlangAsk, executeNatlangCall, parseAskArgs, type AskHost, type CallHost } from './natlang-command.js';

const ROOT = '/workspace';
const BINARIES = new Set(getCommandNames());
for (const name of ['python', 'python3', 'sqlite3', 'changes', 'natlang']) BINARIES.add(name);
const builtin = (path: string) => /^\/(?:usr\/)?bin\/[^/]+$/.test(path) && BINARIES.has(path.split('/').at(-1)!);
const PINNED_TIME = new Date('2026-01-01T00:00:00.000Z');
const asBytes = (value: FileContent) => typeof value === 'string' ? new TextEncoder().encode(value) : value;
const decodeStdin = (value: unknown) => new TextDecoder().decode(Uint8Array.from(value as string, char => char.charCodeAt(0)));

/** A lazy filesystem adapter: stat and directory traversal use metadata, not file contents. */
export class FolderFs implements IFileSystem {
  private readonly emptyDirs = new Set<string>();
  constructor(readonly folder: Folder) {}
  resolvePath(base: string, path: string): string {
    const full = path.startsWith('/') ? path : `${base}/${path}`;
    const parts: string[] = [];
    for (const part of full.split('/')) {
      if (!part || part === '.') continue;
      if (part === '..') parts.pop(); else parts.push(part);
    }
    const resolved = `/${parts.join('/')}`;
    return resolved;
  }
  private relative(path: string): string {
    const resolved = this.resolvePath(ROOT, path);
    if (resolved !== ROOT && !resolved.startsWith(`${ROOT}/`))
      throw new RangeError(`path escapes folder root ${ROOT}: ${path}`);
    return resolved.slice(ROOT.length).replace(/^\//, '');
  }
  private kind(path: string): 'file' | 'directory' | null {
    if (builtin(path)) return 'file';
    if (path === '/bin' || path === '/usr' || path === '/usr/bin') return 'directory';
    // Nothing exists outside the folder: an unknown command looked up on PATH is "not found", not an escape.
    const resolved = this.resolvePath(ROOT, path);
    if (resolved !== ROOT && !resolved.startsWith(`${ROOT}/`)) return null;
    const relative = this.relative(path);
    return this.folder.isFile(relative) ? 'file' : this.folder.isFolder(relative) || this.emptyDirs.has(relative) ? 'directory' : null;
  }
  async readFileBuffer(path: string): Promise<Uint8Array> { return builtin(path) ? new Uint8Array() : this.folder.readBytesSync(this.relative(path)); }
  async readFile(path: string, options?: { encoding?: BufferEncoding | null } | BufferEncoding): Promise<string> {
    const encoding = typeof options === 'string' ? options : options?.encoding ?? 'utf8';
    const data = await this.readFileBuffer(path);
    if (encoding === 'binary' || encoding === 'latin1') return Array.from(data, byte => String.fromCharCode(byte)).join('');
    if (encoding === 'hex') return Array.from(data, byte => byte.toString(16).padStart(2, '0')).join('');
    if (encoding === 'base64') return btoa(Array.from(data, byte => String.fromCharCode(byte)).join(''));
    return new TextDecoder(encoding === 'ascii' ? 'ascii' : 'utf-8').decode(data);
  }
  async writeFile(path: string, content: FileContent, _options?: { encoding?: BufferEncoding } | BufferEncoding): Promise<void> {
    this.folder.writeBytes(this.relative(path), asBytes(content));
  }
  async appendFile(path: string, content: FileContent, options?: { encoding?: BufferEncoding } | BufferEncoding): Promise<void> {
    const old = await this.exists(path) ? await this.readFileBuffer(path) : new Uint8Array();
    const next = asBytes(content), joined = new Uint8Array(old.length + next.length);
    joined.set(old); joined.set(next, old.length);
    await this.writeFile(path, joined, options);
  }
  async exists(path: string): Promise<boolean> { return this.kind(path) !== null; }
  async stat(path: string): Promise<FsStat> {
    const kind = this.kind(path);
    if (!kind) throw new Error(`ENOENT: ${path}`);
    const size = kind === 'file' && !builtin(path) ? (await this.folder.stat(this.relative(path))).bytes : 0;
    return { isFile: kind === 'file', isDirectory: kind === 'directory', isSymbolicLink: false,
      mode: kind === 'file' ? 0o100644 : 0o40755, size, mtime: PINNED_TIME };
  }
  async lstat(path: string): Promise<FsStat> { return this.stat(path); }
  async mkdir(path: string, options?: MkdirOptions): Promise<void> {
    const relative = this.relative(path);
    if (this.kind(path)) { if (options?.recursive) return; throw new Error(`EEXIST: ${path}`); }
    const parts = relative.split('/');
    if (!options?.recursive && parts.length > 1 && !this.kind(`${ROOT}/${parts.slice(0, -1).join('/')}`))
      throw new Error(`ENOENT: parent of ${path}`);
    for (let i = 1; i <= parts.length; i++) this.emptyDirs.add(parts.slice(0, i).join('/'));
  }
  async readdir(path: string): Promise<string[]> {
    if (this.kind(path) !== 'directory') throw new Error(`ENOTDIR: ${path}`);
    const relative = this.relative(path), prefix = relative ? `${relative}/` : '';
    const children = new Set(this.folder.list(relative).map(entry => entry.path.slice(prefix.length)));
    for (const dir of this.emptyDirs) if (dir.startsWith(prefix) && !dir.slice(prefix.length).includes('/'))
      children.add(dir.slice(prefix.length));
    return [...children].sort();
  }
  async readdirWithFileTypes(path: string) {
    const names = await this.readdir(path);
    return names.map(name => ({ name, isFile: this.kind(`${path}/${name}`) === 'file',
      isDirectory: this.kind(`${path}/${name}`) === 'directory', isSymbolicLink: false }));
  }
  async rm(path: string, options?: RmOptions): Promise<void> {
    const kind = this.kind(path), relative = this.relative(path);
    if (!kind) { if (options?.force) return; throw new Error(`ENOENT: ${path}`); }
    if (kind === 'directory' && !options?.recursive && (await this.readdir(path)).length)
      throw new Error(`ENOTEMPTY: ${path}`);
    if (this.folder.isFile(relative) || this.folder.isFolder(relative) && relative) this.folder.remove(relative);
    for (const dir of [...this.emptyDirs]) if (dir === relative || dir.startsWith(`${relative}/`)) this.emptyDirs.delete(dir);
  }
  async cp(src: string, dest: string, options?: CpOptions): Promise<void> {
    if (this.kind(src) === 'directory') {
      if (!options?.recursive) throw new Error('cp: directory needs -r');
      await this.mkdir(dest, { recursive: true });
      for (const name of await this.readdir(src)) await this.cp(`${src}/${name}`, `${dest}/${name}`, options);
    } else await this.writeFile(dest, await this.readFileBuffer(src));
  }
  async mv(src: string, dest: string): Promise<void> {
    const from = this.relative(src), to = this.relative(dest);
    if (this.kind(dest) === 'file') await this.rm(dest);
    if (this.folder.isFile(from) || this.folder.isFolder(from)) this.folder.move(from, to);
    for (const dir of [...this.emptyDirs]) if (dir === from || dir.startsWith(`${from}/`)) {
      this.emptyDirs.delete(dir); this.emptyDirs.add(`${to}${dir.slice(from.length)}`);
    }
  }
  getAllPaths(): string[] { return [ROOT, ...this.folder.listFiles().map(entry => `${ROOT}/${entry.path}`),
    ...this.folder.listFolders().map(entry => `${ROOT}/${entry.path}`), ...[...this.emptyDirs].map(path => `${ROOT}/${path}`)]; }
  async chmod(path: string): Promise<void> { if (!await this.exists(path)) throw new Error(`ENOENT: ${path}`); }
  async symlink(): Promise<void> { throw new Error('symlinks are not supported in folder bash'); }
  async link(): Promise<void> { throw new Error('hard links are not supported in folder bash'); }
  async readlink(): Promise<string> { throw new Error('not a symlink'); }
  async realpath(path: string): Promise<string> { if (!await this.exists(path)) throw new Error(`ENOENT: ${path}`); return this.resolvePath(ROOT, path); }
  async utimes(path: string): Promise<void> { if (!await this.exists(path)) throw new Error(`ENOENT: ${path}`); }
}

/** Refuse open-ended shell loops and function recursion before any command runs. */
export function checkBashPolicy(command: string): void {
  checkBashAst(new Bash().transform(command).ast);
}
function checkBashAst(tree: unknown): void {
  const calls = new Map<string, Set<string>>();
  const visit = (value: unknown, current?: string): void => {
    if (!value || typeof value !== 'object') return;
    if (Array.isArray(value)) { value.forEach(item => visit(item, current)); return; }
    const node = value as Record<string, unknown>;
    if (['While', 'Until', 'CStyleFor'].includes(String(node.type)))
      throw new SyntaxError(`${node.type} loops are not allowed; use for x in a finite list or iterateOn`);
    if (node.type === 'FunctionDef') { const name = String(node.name); calls.set(name, new Set()); visit(node.body, name); return; }
    if (node.type === 'SimpleCommand') {
      const parts = (node.name as { parts?: Array<{ type: string; value?: string }> } | null)?.parts;
      const name = parts?.every(part => part.type === 'Literal') ? parts.map(part => part.value).join('') : undefined;
      if (name && ['source', '.', 'eval', 'bash', 'sh'].includes(name))
        throw new SyntaxError(`${name} cannot bypass the bash policy; use commands directly`);
      if (current && name) calls.get(current)?.add(name);
    }
    for (const [key, child] of Object.entries(node)) if (key !== 'name' || node.type !== 'SimpleCommand') visit(child, current);
  };
  visit(tree);
  const reaches = (start: string, name: string, seen = new Set<string>()): boolean => {
    if (seen.has(name)) return false;
    seen.add(name);
    return [...calls.get(name) ?? []].some(next => next === start || reaches(start, next, seen));
  };
  for (const name of calls.keys()) if (reaches(name, name))
    throw new SyntaxError(`function ${name} calls itself; use a finite for loop`);
}

export type FolderBashResult = { exitCode: number; stdout: string; stderr: string; changedPaths: string[] };
export type FolderBashOptions = { network?: boolean; python?: PythonHost; ask?: AskHost; call?: CallHost;
  apply?: (name: string, path: string) => Promise<unknown>;
  /** The directory the command starts in: absolute under /workspace, or relative to it (default /workspace). */
  cwd?: string;
  /** Environment variables for this command, over the shell's own. */
  env?: Record<string, string>;
  /** Stops the command at its next statement boundary. */
  signal?: AbortSignal;
  /** Arguments appended to the first command as they are, without shell parsing (an argv run as `command`, `args`). */
  args?: string[] };
export async function runFolderBash(folder: Folder, command: string, options: FolderBashOptions = {}): Promise<FolderBashResult> {
  const before = new Map(folder.diffSync().changes.map(change => [change.path, change.after]));
  const changes = defineCommand('changes', async () => ({ stdout: JSON.stringify(folder.diffSync().changes.map(change =>
    ({ path: change.path, kind: change.kind })), null, 2) + '\n', stderr: '', exitCode: 0 }));
  const pythonCommand = (name: string) => defineCommand(name, async (args, ctx) => {
    try {
      let source: string;
      if (args[0] === '-c') source = args[1] ?? '';
      else if (args[0] === '-' || !args.length) source = decodeStdin(ctx.stdin);
      else source = await folder.readText(args[0]!);
      const result = await runFolderPython(folder, source, options.python);
      return { stdout: result.stdout ? `${result.stdout}\n` : '', stderr: result.stderr, exitCode: 0 };
    } catch (error) { return { stdout: '', stderr: String(error), exitCode: 1 }; }
  });
  const sqlite = defineCommand('sqlite3', async (args, ctx) => {
    const flags: string[] = [], positional: string[] = [];
    let separator = '|';
    for (let index = 0; index < args.length; index++) {
      const arg = args[index]!;
      if (arg === '-separator') {
        if (args[index + 1] === undefined) return { stdout: '', stderr: '-separator needs a value', exitCode: 1 };
        separator = args[++index]!;
      } else if (arg.startsWith('-')) flags.push(arg);
      else positional.push(arg);
    }
    const database = positional[0], sql = positional.slice(1).join(' ') || decodeStdin(ctx.stdin);
    if (!database) return { stdout: '', stderr: 'sqlite3 needs a database path', exitCode: 1 };
    const script = `import sqlite3, csv, io, json\n` +
      `con = sqlite3.connect(${JSON.stringify(database)})\n` +
      `sql = ${JSON.stringify(sql)}\n` +
      `rows = []; columns = []\n` +
      `pending = ''\n` +
      `for char in sql:\n` +
      `    pending += char\n` +
      `    if char == ';' and sqlite3.complete_statement(pending):\n` +
      `        cur = con.execute(pending)\n` +
      `        if cur.description: columns = [item[0] for item in cur.description]; rows = cur.fetchall()\n` +
      `        pending = ''\n` +
      `if pending.strip():\n` +
      `    cur = con.execute(pending)\n` +
      `    if cur.description: columns = [item[0] for item in cur.description]; rows = cur.fetchall()\n` +
      `con.commit(); con.close()\n` +
      `buffer = io.StringIO()\n` +
      (flags.includes('-json') ? `print(json.dumps([dict(zip(columns, row)) for row in rows]))\n` :
        `writer = csv.writer(buffer, delimiter=${JSON.stringify(flags.includes('-csv') ? ',' : separator)}, lineterminator='\\n')\n` +
        `${flags.includes('-header') ? 'writer.writerow(columns)\n' : ''}writer.writerows(rows)\n` +
        `print(buffer.getvalue(), end='')\n`);
    try {
      const result = await runFolderPython(folder, script, options.python);
      return { stdout: result.stdout ? `${result.stdout}\n` : '', stderr: result.stderr, exitCode: 0 };
    } catch (error) { return { stdout: '', stderr: String(error), exitCode: 1 }; }
  });
  const natlang = defineCommand('natlang', async (args, ctx) => {
    try {
      if (args[0] === 'ask') {
        if (!options.ask) throw new Error('natlang calls are unavailable outside a natlang runtime');
        const parsed = parseAskArgs(args.slice(1));
        return { stdout: await executeNatlangAsk(folder, parsed.instruction, decodeStdin(ctx.stdin), parsed.options, options.ask),
          stderr: '', exitCode: 0 };
      }
      if (args[0] === 'call') {
        if (!options.call) throw new Error('natlang calls are unavailable outside a natlang runtime');
        const words = args.slice(1).filter(arg => arg !== '--lines' && arg !== '--jsonl');
        if (words.length !== 1) throw new Error('usage: natlang call NAME|FILE.nl [--lines|--jsonl]');
        const lines = args.includes('--lines'), jsonl = args.includes('--jsonl');
        if (lines && jsonl) throw new Error('--lines and --jsonl are mutually exclusive');
        return { stdout: await executeNatlangCall(words[0]!, decodeStdin(ctx.stdin), lines ? 'line' : jsonl ? 'jsonl' : 'whole', options.call),
          stderr: '', exitCode: 0 };
      }
      if (args[0] === 'apply') {
        if (!options.apply) throw new Error('natlang reducers are unavailable outside a natlang runtime');
        if (args.length !== 3) throw new Error('usage: natlang apply REDUCER DIR');
        const value = await options.apply(args[1]!, args[2]!);
        return { stdout: `${value === null || typeof value === 'object' ? JSON.stringify(value) : String(value)}\n`,
          stderr: '', exitCode: 0 };
      }
      return { stdout: '', stderr: 'usage: natlang ask|call|apply ...', exitCode: 2 };
    } catch (error) { return { stdout: '', stderr: String(error), exitCode: 1 }; }
  });
  const bash = new Bash({ fs: new FolderFs(folder), cwd: ROOT, python: false, javascript: false,
    ...(options.network === false ? {} : { network: { dangerouslyAllowFullInternetAccess: true } }),
    customCommands: [changes, pythonCommand('python'), pythonCommand('python3'), sqlite, natlang],
    executionLimits: { maxCommandCount: 1000, maxLoopIterations: 10000 } });
  checkBashAst(bash.transform(command).ast);
  const result = await bash.exec(command, { ...(options.cwd === undefined ? {} : { cwd: bash.fs.resolvePath(ROOT, options.cwd) }),
    ...(options.env ? { env: options.env } : {}), ...(options.signal ? { signal: options.signal } : {}),
    ...(options.args ? { args: options.args } : {}) });
  const changedPaths = folder.diffSync().changes.filter(change => {
    const prior = before.get(change.path), after = change.after;
    return !before.has(change.path) || prior?.length !== after?.length || prior?.some((byte, i) => byte !== after?.[i]);
  }).map(change => change.path);
  return { exitCode: result.exitCode, stdout: result.stdout, stderr: result.stderr, changedPaths };
}
