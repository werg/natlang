/** Reuse immutable-source load products only within one synchronous load/build.
 * No parsed record or lexical types escape into a later source revision's cache.
 */
type SourceOwner = { sourceGraphReuse?: boolean };
const sessions = new WeakMap<object, Map<string, unknown>>();
const enabled = (files: SourceOwner) => files.sourceGraphReuse !== false &&
  (globalThis as {process?:{env?:Record<string,string|undefined>}}).process?.env?.NATLANG_SOURCE_GRAPH_REUSE !== '0';
export function withSourceLoadSession<T>(files: SourceOwner, run: () => T): T {
  if (!enabled(files) || sessions.has(files)) return run();
  sessions.set(files, new Map());
  try { return run(); } finally { sessions.delete(files); }
}
export function sourceLoadMemo<T>(files: SourceOwner, key: string, run: () => T): T {
  if (!enabled(files)) return run();
  const memo = sessions.get(files);
  if (!memo) return withSourceLoadSession(files, () => sourceLoadMemo(files, key, run));
  if (memo.has(key)) return memo.get(key) as T;
  const value = run();
  memo.set(key, value);
  return value;
}
