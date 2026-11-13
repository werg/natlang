/** What a command (bash or Python) changed in a Folder overlay; no Python runtime, so isolate bundles keep it. */
import { Folder } from './scoped-fs.js';

/** The folder's file contents before a command, for `pathsChangedSince`. */
export const folderContents = (folder: Folder) => new Map(folder.diffSync().changes.map(change => [change.path, change.after]));
/**
 * What a command did to the folder: `changedPaths` are the files it created or rewrote (all exist afterwards), and
 * `removedPaths` the files it removed (present only when there are some). A move reports its new path as changed and
 * its old path as removed.
 */
export function pathsChangedSince(folder: Folder, before: Map<string, Uint8Array | undefined>): { changedPaths: string[]; removedPaths?: string[] } {
  const changedPaths: string[] = [], removedPaths: string[] = [];
  for (const change of folder.diffSync().changes) {
    const prior = before.get(change.path), after = change.after;
    if (before.has(change.path) && prior?.length === after?.length && !prior?.some((byte, i) => byte !== after?.[i])) continue;
    (after === undefined ? removedPaths : changedPaths).push(change.path);
  }
  return { changedPaths, ...(removedPaths.length ? { removedPaths } : {}) };
}
