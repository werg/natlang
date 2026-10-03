/** A `SkillSource` over a directory on disk (Node only; the browser supplies its own source). */
import { readdir, readFile } from 'node:fs/promises';
import { join, relative, sep } from 'node:path';
import type { SkillSource } from './registry.js';

export function directorySkillSource(root: string): SkillSource {
  return {
    async list() {
      const entries = await readdir(root, { recursive: true, withFileTypes: true });
      return entries.filter(entry => entry.isFile())
        .map(entry => relative(root, join(entry.parentPath, entry.name)).split(sep).join('/')).sort();
    },
    async read(path) {
      if (path.split('/').includes('..')) throw new Error(`path escapes the context: ${path}`);
      const bytes = await readFile(join(root, ...path.split('/')));
      return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    },
  };
}
