import type {RewriteResult} from '../../types';

/** Derive the edit report from the folder, rather than asking the editor to copy paths. */
export async function finish(folder:{diff():Promise<{changes:{path:string}[]}>},summary:string,preserves:string[]):Promise<RewriteResult> {
  const diff=await folder.diff();
  return {summary,changed:diff.changes.map(change=>change.path),preserves};
}
