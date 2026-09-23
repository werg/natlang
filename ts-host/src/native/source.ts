import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { basename, dirname, extname, join, resolve } from 'node:path';
import { loadAnonymousInstructionSource, loadCodebaseSource, loadFunctionSource,
  type SourceFiles } from './source-core.js';
import type { LambdaNode } from './values.js';

const nodeFiles: SourceFiles = {
  resolve, join, dirname, basename, extname, exists: existsSync,
  isFile: path => existsSync(path) && statSync(path).isFile(),
  isDirectory: path => existsSync(path) && statSync(path).isDirectory(),
  read: path => readFileSync(path, 'utf8'),
  list: readdirSync,
};

/** Load frontmatter, companion functions, lexical types, and explicit uses. */
export function loadFunctionFile(path: string, validateImport?: (specifier: string) => void): LambdaNode {
  return loadFunctionSource(path, nodeFiles, validateImport);
}

/** Load the top-level .nl/.ts functions rooted in a native directory. */
export function loadCodebaseDirectory(path: string): Record<string, unknown> {
  return loadCodebaseSource(path, nodeFiles);
}

/** Build an anonymous string-valued instruction with a native directory as its codebase. */
export function loadAnonymousInstruction(path: string, instructions: string): Record<string, unknown> {
  return loadAnonymousInstructionSource(path, instructions, nodeFiles);
}
