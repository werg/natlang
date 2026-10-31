/**
 * Acceptance for natlang codebase repair tasks: the folder a directory reducer edited is a natlang package, and the
 * repair is judged by loading it with the actual loader (every named function, `uses` entry, type name and
 * TypeScript child of the package must resolve), plus exact constraints that keep the repair honest: files outside
 * the repaired one stay byte-identical, the repaired file still exists, and declared function signatures that the
 * task fixes stay as specified. A repair that deletes or neuters the broken function therefore fails.
 */
import { createHash } from 'node:crypto';
import { loadVirtualCallables } from '../runtime/virtual-project.js';

export type CodebaseCheckSpec = {
  /** The package must load with the actual natlang loader. */
  load: true;
  /** Files (path -> SHA-256 of their text) that must be unchanged. */
  preserve?: Record<string, string>;
  /** Files that must exist after the repair. */
  require?: string[];
  /** Named functions whose frontmatter must declare exactly these signatures (whitespace-insensitive). */
  signatures?: { path: string; args?: string; returns?: string; uses?: string[] }[];
  /** Text each file must still contain (whitespace-insensitive), e.g. a TypeScript child's exported header. */
  contains?: { path: string; text: string }[];
  /** No file may be added outside these path prefixes (default: none may be added). */
  allow_new_prefixes?: string[];
};

export type CodebaseCheckResult = { ok: boolean; problems: string[] };

const sha = (text: string) => createHash('sha256').update(text).digest('hex');
const squash = (text: string | undefined) => (text ?? '').replace(/\s+/g, '');

function frontmatter(text: string): Record<string, string> {
  const match = /^---\n([\s\S]*?)\n---/.exec(text);
  const out: Record<string, string> = {};
  if (!match) return out;
  let key = '';
  for (const line of match[1]!.split('\n')) {
    const head = /^([A-Za-z_]+):\s?(.*)$/.exec(line);
    if (head) { key = head[1]!; out[key] = head[2]!; } else if (key) out[key] += '\n' + line;
  }
  return out;
}

function usesList(value: string | undefined): string[] {
  if (!value) return [];
  return value.replace(/^\[|\]$/g, '').split(/[,\n]/).map(item => item.replace(/^\s*-\s*/, '').trim()).filter(Boolean).sort();
}

export function checkCodebase(files: Record<string, string>, spec: CodebaseCheckSpec, original: Record<string, string>): CodebaseCheckResult {
  const problems: string[] = [];
  for (const [path, digest] of Object.entries(spec.preserve ?? {}))
    if (files[path] === undefined || sha(files[path]!) !== digest) problems.push(`${path} changed; only the broken function may be repaired`);
  for (const path of spec.require ?? []) if (files[path] === undefined) problems.push(`${path} is missing`);
  const allowed = spec.allow_new_prefixes ?? [];
  for (const path of Object.keys(files))
    if (original[path] === undefined && !allowed.some(prefix => path.startsWith(prefix))) problems.push(`${path} was added`);
  for (const signature of spec.signatures ?? []) {
    const text = files[signature.path];
    if (text === undefined) continue;
    const declared = frontmatter(text);
    if (signature.args !== undefined && squash(declared.args) !== squash(signature.args)) problems.push(`${signature.path}: args differ from the declared contract`);
    if (signature.returns !== undefined && squash(declared.returns) !== squash(signature.returns)) problems.push(`${signature.path}: returns differs from the declared contract`);
    if (signature.uses !== undefined && usesList(declared.uses).join(',') !== [...signature.uses].sort().join(',')) problems.push(`${signature.path}: uses differs from the declared contract`);
  }
  for (const item of spec.contains ?? [])
    if (files[item.path] !== undefined && !squash(files[item.path]).includes(squash(item.text))) problems.push(`${item.path}: its declared export changed`);
  try { loadVirtualCallables(files); }
  catch (error) { problems.push(`the package does not load: ${String((error as Error).message ?? error).slice(0, 400)}`); }
  return { ok: problems.length === 0, problems };
}
