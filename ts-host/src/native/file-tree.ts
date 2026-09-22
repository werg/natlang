import { lazyDict, type LazyDict } from './host-tree.js';

export type FileTreeLeaf = { kind: 'text'; text: string; bytes: number } |
  { kind: 'binary'; bytes: number };
export const FILE_TREE_LEAF_TYPE = '{ kind: "text", text: Text, bytes: Num } | { kind: "binary", bytes: Num }';
export type FileWrite = { path: string; text: string };
export const FILE_WRITE_TYPE = '{ path: Text, text: Text }';

/** Turn an in-memory text project into a lazy Dict<FileTreeLeaf>. */
export function textFileTree(files: Record<string, string>, label = 'project files'): LazyDict<FileTreeLeaf> {
  return lazyDict(Object.fromEntries(Object.entries(files).map(([path, text]) =>
    [path.replace(/^[/\\]+/, ''), { kind: 'text', text, bytes: new TextEncoder().encode(text).byteLength }])), label);
}

/** Validate and normalize a portable `{path, text}[]` change plan. */
export function validateFileWrites(value: unknown): FileWrite[] {
  if (!Array.isArray(value)) throw new TypeError('file writes must be a list of {path, text} records');
  const seen = new Set<string>();
  return value.map((raw, index) => {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw) ||
        Object.keys(raw).sort().join(',') !== 'path,text')
      throw new TypeError(`file write ${index} must contain exactly path and text`);
    const { path, text } = raw as Record<string, unknown>;
    if (typeof path !== 'string' || typeof text !== 'string')
      throw new TypeError(`file write ${index} path and text must be strings`);
    if (!path || path.startsWith('/') || path.startsWith('\\') || path.includes('\\') || path.includes('\0') ||
        path.split('/').some(part => !part || part === '.' || part === '..'))
      throw new RangeError(`invalid relative file path: ${JSON.stringify(path)}`);
    const normalized = path.split('/').join('/');
    if (seen.has(normalized)) throw new RangeError(`duplicate file write path: ${normalized}`);
    seen.add(normalized); return { path: normalized, text };
  });
}
