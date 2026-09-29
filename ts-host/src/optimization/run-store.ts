import { existsSync, mkdirSync, readFileSync, writeFileSync, renameSync, unlinkSync, openSync, closeSync, fsyncSync } from 'node:fs';
import { resolve, join, dirname } from 'node:path';
import { canonical, fingerprint } from '../adaptation/identity.js';
import { parseStrictJSON } from '../adaptation/schema.js';
export class RunStore {
  readonly directory: string;
  private readonly lock: string;
  private readonly token = fingerprint({ pid: process.pid, time: Date.now(), nonce: Math.random() });
  constructor(directory: string) {
    this.directory = resolve(directory); mkdirSync(this.directory, { recursive: true }); this.lock = join(this.directory, 'run.lock');
    if (existsSync(this.lock)) {
      const previous = JSON.parse(readFileSync(this.lock, 'utf8')) as { pid: number };
      try { process.kill(previous.pid, 0); throw new Error('adaptation run is locked by process ' + previous.pid); }
      catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ESRCH') throw error; unlinkSync(this.lock); }
    }
    const fd = openSync(this.lock, 'wx'); try { writeFileSync(fd, canonical({ pid: process.pid, token: this.token })); } finally { closeSync(fd); }
  }
  has(name: string): boolean { return existsSync(join(this.directory, name)); }
  read<T>(name: string): T { return parseStrictJSON(readFileSync(join(this.directory, name), 'utf8')) as T; }
  write(name: string, value: unknown): void {
    this.writeText(name, canonical(value) + '\n');
  }
  writeText(name: string, text: string): void {
    const target = join(this.directory, name); const tmp = target + '.' + this.token + '.tmp';
    const fd = openSync(tmp, 'w');
    try { writeFileSync(fd, text); fsyncSync(fd); } finally { closeSync(fd); }
    renameSync(tmp, target);
    const directory = openSync(dirname(target), 'r');
    try { fsyncSync(directory); } finally { closeSync(directory); }
  }
  manifest(value: unknown): void {
    if (this.has('manifest.json')) { if (canonical(this.read('manifest.json')) !== canonical(value)) throw new Error('immutable run manifest differs'); }
    else this.write('manifest.json', value);
  }
  blob(value: unknown): string {
    const hash = fingerprint(value); const dir = join(this.directory, 'blobs'); mkdirSync(dir, { recursive: true });
    const name = 'blobs/' + hash + '.json'; if (!this.has(name)) this.write(name, value); return hash;
  }
  event(value: unknown): void {
    // Each event is independently committed; the index is atomically updated.
    const index = this.has('events.json') ? this.read<string[]>('events.json') : [];
    index.push(this.blob(value)); this.write('events.json', index);
  }
  close(): void { if (existsSync(this.lock) && this.read<{ token: string }>('run.lock').token === this.token) unlinkSync(this.lock); }
}
