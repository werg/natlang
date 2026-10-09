/** Node: the standard-library `.nz` file by path (the shared combinators take and return bytes, for browsers too). */
import { readFileSync, writeFileSync } from 'node:fs';
import type { NeuraleseStore } from '../native/neuralese-store.js';
import * as shared from './combinators.js';
import type { StandardLibrary } from './combinators.js';

/** Read a standard-library `.nz` file from `source` (a path or its bytes), registering its blocks in `store`. */
export async function loadStandardLibrary(source: string | Uint8Array, store?: NeuraleseStore): Promise<StandardLibrary> {
  return shared.loadStandardLibrary(typeof source === 'string' ? new Uint8Array(readFileSync(source)) : source, store);
}

/** Initialise the library on a Neuralese server (see the shared `buildStandardLibrary`), also writing it to `path`. */
export async function buildStandardLibrary(options: Parameters<typeof shared.buildStandardLibrary>[0] & { path?: string }) {
  const { path, ...rest } = options;
  const built = await shared.buildStandardLibrary(rest);
  if (path) writeFileSync(path, built.bytes);
  return built;
}
