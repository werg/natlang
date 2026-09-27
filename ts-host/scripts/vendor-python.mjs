/** Download the fixed Pyodide wheel set once and verify it against Pyodide's lock. */
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const lock = JSON.parse(await readFile(resolve(root, '../node_modules/pyodide/pyodide-lock.json'), 'utf8'));
const names = new Set();
function add(name) {
  if (names.has(name)) return;
  const entry = lock.packages[name];
  if (!entry) throw new Error(`package absent from Pyodide lock: ${name}`);
  names.add(name);
  entry.depends.forEach(add);
}
['numpy', 'pandas'].forEach(add);
const directory = resolve(root, 'vendor/pyodide');
await mkdir(directory, { recursive: true });
for (const name of names) {
  const { file_name: filename, sha256 } = lock.packages[name];
  const target = resolve(directory, filename);
  let data;
  try { data = await readFile(target); } catch { /* first download */ }
  if (!data || createHash('sha256').update(data).digest('hex') !== sha256) {
    const url = `https://cdn.jsdelivr.net/pyodide/v314.0.7/full/${filename}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status}`);
    data = Buffer.from(await response.arrayBuffer());
    if (createHash('sha256').update(data).digest('hex') !== sha256)
      throw new Error(`hash mismatch for ${filename}`);
    await writeFile(target, data);
  }
  process.stdout.write(`${filename} ${data.length} bytes verified\n`);
}
