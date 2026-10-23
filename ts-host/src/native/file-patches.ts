import { createHash } from 'node:crypto';

/** Content-addressed patch for ordinary source files; native state updates use named tensor artifacts instead. */
export const FILE_PATCH_SCHEMA = 'natlang.file-patch/1';
export type FilePatch = { readonly schema: typeof FILE_PATCH_SCHEMA; readonly base: string;
  readonly changes: Readonly<Record<string, string | null>> };

export class FilePatchError extends Error {
  constructor(readonly code: string, message: string) { super(`${code}: ${message}`); this.name = 'FilePatchError'; }
}

export const filesDigest = (files: Readonly<Record<string, string>>) => createHash('sha256')
  .update(JSON.stringify(Object.keys(files).sort().map(path => [path, files[path]]))).digest('hex');

/** The patch that turns `base` into `after`: changed and added paths with their text, removed paths as `null`. */
export function diffFiles(after: Readonly<Record<string, string>>, base: Readonly<Record<string, string>>): FilePatch {
  const changes: Record<string, string | null> = {};
  for (const path of new Set([...Object.keys(base), ...Object.keys(after)])) {
    if (!(path in after)) changes[path] = null;
    else if (base[path] !== after[path]) changes[path] = after[path]!;
  }
  return { schema: FILE_PATCH_SCHEMA, base: filesDigest(base),
    changes: Object.fromEntries(Object.entries(changes).sort(([a], [b]) => a < b ? -1 : 1)) };
}

/** `files` with the patch applied; `strict` refuses files other than the patch's base. */
export function applyPatch(files: Readonly<Record<string, string>>, patch: FilePatch,
  options: { strict?: boolean } = {}): Record<string, string> {
  if (patch.schema !== FILE_PATCH_SCHEMA) throw new FilePatchError('patch-schema', 'unsupported file patch schema');
  if (options.strict && filesDigest(files) !== patch.base) throw new FilePatchError('patch-base', 'the files are not the patch\'s base');
  const out: Record<string, string> = { ...files };
  for (const [path, text] of Object.entries(patch.changes)) { if (text === null) delete out[path]; else out[path] = text; }
  return out;
}

/** Patches taken against one base, merged; a path changed differently by two patches is a conflict. */
export function composePatches(patches: readonly FilePatch[]): FilePatch {
  if (!patches.length) throw new FilePatchError('patch-arguments', 'composePatches needs at least one patch');
  const changes: Record<string, string | null> = {};
  for (const patch of patches) {
    if (patch.schema !== FILE_PATCH_SCHEMA) throw new FilePatchError('patch-schema', 'unsupported file patch schema');
    if (patch.base !== patches[0]!.base) throw new FilePatchError('patch-base', 'patches to compose must share their base');
    for (const [path, text] of Object.entries(patch.changes)) {
      if (path in changes && changes[path] !== text) throw new FilePatchError('patch-conflict', `${path} is changed differently by two patches`);
      changes[path] = text;
    }
  }
  return { schema: FILE_PATCH_SCHEMA, base: patches[0]!.base, changes };
}
