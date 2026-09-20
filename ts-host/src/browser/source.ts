import { loadFunctionSource, type SourceFiles } from '../native/source-core.js';
import type { LambdaNode } from '../native/values.js';

function normalize(path: string): string {
  const parts: string[] = [];
  for (const part of path.replaceAll('\\', '/').split('/')) {
    if (!part || part === '.') continue;
    if (part === '..') parts.pop();
    else parts.push(part);
  }
  return `/${parts.join('/')}`;
}
function extname(path: string): string {
  const name = path.slice(path.lastIndexOf('/') + 1);
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(dot) : '';
}
function dirname(path: string): string { return path.slice(0, path.lastIndexOf('/')) || '/'; }
function basename(path: string, extension = ''): string {
  const name = path.slice(path.lastIndexOf('/') + 1);
  return extension && name.endsWith(extension) ? name.slice(0, -extension.length) : name;
}

/** Load the same .nl/.ts frontmatter and companion graph as the Node file loader. */
export function loadFunctionFiles(root: string, source: Record<string, string>): LambdaNode {
  const files = new Map(Object.entries(source).map(([path, body]) => [normalize(path), body]));
  const ops: SourceFiles = {
    resolve: normalize, join: (...parts) => normalize(parts.join('/')),
    dirname, basename, extname,
    exists: path => files.has(normalize(path)),
    isFile: path => files.has(normalize(path)),
    isDirectory: path => {
      const prefix = `${normalize(path).replace(/\/$/, '')}/`;
      for (const key of files.keys()) if (key.startsWith(prefix)) return true;
      return false;
    },
    read: path => {
      const body = files.get(normalize(path));
      if (body === undefined) throw new Error(`source file ${path} is missing`);
      return body;
    },
    list: path => {
      const prefix = `${normalize(path).replace(/\/$/, '')}/`;
      return [...new Set([...files.keys()].filter(key => key.startsWith(prefix))
        .map(key => key.slice(prefix.length).split('/')[0]!).filter(Boolean))];
    },
  };
  return loadFunctionSource(root, ops);
}
