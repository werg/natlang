import { createHash, randomUUID } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, open, link, rename, unlink } from 'node:fs/promises';
import { dirname } from 'node:path';
import { createInterface } from 'node:readline';

export function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value).sort().filter(k => value[k] !== undefined).map(k => `${JSON.stringify(k)}:${canonical(value[k])}`).join(',')}}`;
  return JSON.stringify(value);
}
export const digest = value => createHash('sha256').update(canonical(value)).digest('hex');
export async function readJsonl(path, { limit = Infinity } = {}) {
  if (!(limit > 0)) throw new Error('limit must be positive');
  const stream = createReadStream(path, { encoding: 'utf8' });
  const lines = createInterface({ input: stream, crlfDelay: Infinity });
  const rows = [];
  try { for await (const line of lines) {
    if (!line.trim()) continue;
    rows.push(JSON.parse(line));
    if (rows.length >= limit) break;
  } } finally { lines.close(); stream.destroy(); }
  return rows;
}
export async function writeJsonl(path, rows, { replace = false } = {}) {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${randomUUID()}`;
  const handle = await open(temporary, 'wx');
  try {
    try { for await (const row of rows) await handle.writeFile(`${JSON.stringify(row)}\n`); await handle.sync(); }
    finally { await handle.close(); }
    if (replace) await rename(temporary, path);
    else await link(temporary, path);
  } finally { await unlink(temporary).catch(e => { if (e.code !== 'ENOENT') throw e; }); }
}
