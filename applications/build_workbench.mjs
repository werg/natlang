/** Serial, declared-input build adapter for the native TypeScript host.
 * Commands are authored and trusted. This is not a process sandbox or a cache.
 */
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createReadStream } from 'node:fs';
import { lstat, realpath } from 'node:fs/promises';
import { isAbsolute, relative, resolve } from 'node:path';

function within(root, path) {
  const rel = relative(root, path);
  return rel !== '' && rel !== '..' && !rel.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`) && !isAbsolute(rel);
}

async function digestFile(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

export class BuildWorkspace {
  constructor(root, { timeoutMs = 30_000, outputLimit = 4096 } = {}) {
    this.rootPath = resolve(root);
    this.timeoutMs = timeoutMs;
    this.outputLimit = outputLimit;
    this.root = null;
    this.events = [];
  }

  async open() {
    this.root = await realpath(this.rootPath);
    return this;
  }

  async path(name, { existing }) {
    if (!this.root || typeof name !== 'string' || !name || isAbsolute(name))
      throw new Error(`invalid workspace path: ${name}`);
    const lexical = resolve(this.root, name);
    if (!within(this.root, lexical)) throw new Error(`path escapes workspace: ${name}`);
    if (existing) {
      const actual = await realpath(lexical);
      if (!within(this.root, actual)) throw new Error(`input escapes workspace: ${name}`);
      if (!(await lstat(actual)).isFile()) throw new Error(`input is not a file: ${name}`);
      return actual;
    }
    const parent = await realpath(resolve(lexical, '..'));
    if (parent !== this.root && !within(this.root, parent)) throw new Error(`output escapes workspace: ${name}`);
    try { await lstat(lexical); throw new Error(`output already exists: ${name}`); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    return lexical;
  }

  async execute(task) {
    const id = String(task.id);
    let input_sha256 = '';
    const fail = (status, detail, exit_code = -1) => {
      this.events.push({ operation: 'build.execute', task: id, status, detail });
      return { id, status, exit_code, input_sha256, output_sha256: '', detail };
    };
    try {
      if (!Array.isArray(task.argv) || !task.argv.length ||
          !Array.isArray(task.inputs) || !Array.isArray(task.outputs))
        return fail('failed', 'invalid task declaration');
      const inputs = [];
      for (const input of task.inputs) {
        const path = await this.path(input, { existing: true });
        inputs.push([input, await digestFile(path)]);
      }
      input_sha256 = createHash('sha256').update(JSON.stringify(inputs)).digest('hex');
      for (const output of task.outputs) await this.path(output, { existing: false });
      const child = spawn(task.argv[0], task.argv.slice(1), {
        cwd: this.root, shell: false, stdio: ['ignore', 'pipe', 'pipe'],
        env: { PATH: process.env.PATH ?? '', LANG: 'C', LC_ALL: 'C' },
      });
      let output = '';
      for (const stream of [child.stdout, child.stderr]) stream.on('data', chunk => {
        if (output.length < this.outputLimit) output += chunk.toString().slice(0, this.outputLimit - output.length);
      });
      let timedOut = false;
      const timer = setTimeout(() => { timedOut = true; child.kill('SIGKILL'); }, this.timeoutMs);
      const closed = await new Promise(resolveClose => {
        child.once('error', error => resolveClose({ error }));
        child.once('close', (code, signal) => resolveClose({ code, signal }));
      });
      clearTimeout(timer);
      if (timedOut || closed.signal) return fail('unknown', `process interrupted; effects may have occurred: ${output}`);
      if (closed.error) return fail('failed', `${closed.error.message}: ${output}`);
      if (closed.code !== 0) return fail('failed', `exit ${closed.code}: ${output}`, closed.code);
      for (const [name, before] of inputs) {
        const path = await this.path(name, { existing: true });
        if (await digestFile(path) !== before) return fail('failed', `declared input changed: ${name}`);
      }
      const hashes = [];
      for (const name of task.outputs) {
        const path = await this.path(name, { existing: true });
        hashes.push([name, await digestFile(path)]);
      }
      const output_sha256 = createHash('sha256').update(JSON.stringify(hashes)).digest('hex');
      this.events.push({ operation: 'build.execute', task: id, status: 'ok', input_sha256, output_sha256 });
      return { id, status: 'ok', exit_code: 0, input_sha256, output_sha256, detail: output };
    } catch (error) {
      return fail('failed', error instanceof Error ? error.message : String(error));
    }
  }

  drainEvents() { return this.events.splice(0); }
}
