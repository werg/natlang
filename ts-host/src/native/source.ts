import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { basename, dirname, extname, join, resolve } from 'node:path';
import { loadFunctionSource, type SourceFiles } from './source-core.js';
import type { LambdaNode } from './values.js';

const nodeFiles: SourceFiles = {
  resolve, join, dirname, basename, extname, exists: existsSync,
  isFile: path => existsSync(path) && statSync(path).isFile(),
  isDirectory: path => existsSync(path) && statSync(path).isDirectory(),
  read: path => readFileSync(path, 'utf8'),
  list: readdirSync,
};

/** Load frontmatter, companion functions, lexical types, and explicit uses. */
export function loadFunctionFile(path: string): LambdaNode {
  return loadFunctionSource(path, nodeFiles);
}
