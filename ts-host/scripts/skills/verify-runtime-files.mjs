import { createHash } from 'node:crypto';
import { open } from 'node:fs/promises';
import { isAbsolute, join, relative, resolve, sep } from 'node:path';

/** Hash all sealed runtime files with bounded concurrency and periodic progress. */
export async function verifyRuntimeFiles(root, files, { concurrency = 2, progressEvery = 1000, onProgress = () => {} } = {}) {
  if (!Number.isSafeInteger(concurrency) || concurrency < 1 || concurrency > 8)
    throw new RangeError('runtime verification concurrency must be an integer from 1 to 8');
  if (!Number.isSafeInteger(progressEvery) || progressEvery < 1)
    throw new RangeError('runtime verification progress interval must be a positive integer');
  if (!files || typeof files !== 'object' || Array.isArray(files)) throw new TypeError('runtime file hashes must be an object');

  const base = resolve(root), entries = Object.entries(files);
  if (!entries.length) throw new Error('sealed runtime file list is empty');
  for (const [path, expected] of entries) {
    if (typeof path !== 'string' || !path || isAbsolute(path) || path.includes('\\') ||
        path.split('/').some(segment => !segment || segment === '.' || segment === '..') ||
        !/^[a-f0-9]{64}$/.test(expected))
      throw new Error(`invalid sealed runtime file entry: ${String(path)}`);
    const absolute = resolve(base, ...path.split('/'));
    const rel = relative(base, absolute);
    if (!rel || rel === '..' || rel.startsWith(`..${sep}`) || isAbsolute(rel))
      throw new Error(`sealed runtime path escapes its root: ${path}`);
  }

  const started = performance.now();
  let next = 0, completed = 0, failure;
  const worker = async () => {
    while (!failure) {
      const index = next++;
      if (index >= entries.length) return;
      const [path, expected] = entries[index];
      try {
        const file = await open(join(base, ...path.split('/')), 'r');
        let actual;
        try {
          const hash = createHash('sha256'), buffer = Buffer.allocUnsafe(64 * 1024);
          while (true) {
            const { bytesRead } = await file.read(buffer, 0, buffer.length, null);
            if (!bytesRead) break;
            hash.update(buffer.subarray(0, bytesRead));
          }
          actual = hash.digest('hex');
        } finally { await file.close(); }
        if (actual !== expected) throw new Error(`Frozen runtime file changed: ${path}`);
        completed++;
        if (completed % progressEvery === 0 || completed === entries.length)
          onProgress({ completed, total: entries.length, elapsedMs: Math.round(performance.now() - started) });
      } catch (error) { failure ??= error; return; }
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, entries.length) }, worker));
  if (failure) throw failure;
  return { files: completed, elapsedMs: Math.round(performance.now() - started) };
}
