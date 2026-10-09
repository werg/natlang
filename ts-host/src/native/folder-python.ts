/** One warm Pyodide interpreter over per-call mounts into Folder overlays. */
import { folderFS, folderView } from './python-folderfs.js';
import { PYTHON_NATLANG, PYTHON_POLICY } from './python-sources.js';
import { Folder } from './scoped-fs.js';

export type PythonHost = { nl?: (instructions: string, returns: string) => (...args: unknown[]) => Promise<unknown>;
  iterateOn?: (step: unknown, initial: unknown, ...fixed: unknown[]) => unknown };
export type PythonResult = { value: unknown; stdout: string; stderr: string; changedPaths: string[] };

const RUNNER = `
import ast as __ast, builtins as __builtins, inspect as __inspect
from natlang_policy import prepare as __prepare
async def __natlang_run(source, namespace):
    tree = __prepare(source)
    if tree.body and isinstance(tree.body[-1], __ast.Expr):
        last = tree.body[-1]
        tree.body[-1] = __ast.Assign(targets=[__ast.Name(id='__natlang_result__', ctx=__ast.Store())], value=last.value)
        __ast.fix_missing_locations(tree)
    code = __builtins.compile(tree, '<cell>', 'exec', flags=__ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    try:
        answer = __builtins.eval(code, namespace)
        if __inspect.isawaitable(answer): await answer
    except KeyboardInterrupt:
        return {'__natlang_timeout__': True}
    return namespace.pop('__natlang_result__', None)
`;

let singleton: Promise<any> | undefined;
let currentHost: PythonHost | undefined;
let currentRefresh: (() => void) | undefined;
let awaitingChild = false;
let nextCall = 0;
let tail: Promise<void> = Promise.resolve();
function fromPython(value: any): unknown {
  if (value && typeof value.toJs === 'function') return fromPython(value.toJs({ dict_converter: Object.fromEntries }));
  if (typeof value === 'bigint') return Number.isSafeInteger(Number(value)) ? Number(value) : value.toString();
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
    const items = Array.from(value as unknown as ArrayLike<unknown>, fromPython);
    return items.length === 1 ? items[0] : items;
  }
  if (Array.isArray(value)) return value.map(fromPython);
  if (value instanceof Map) return Object.fromEntries([...value].map(([key, item]) => [String(key), fromPython(item)]));
  if (value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype)
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, fromPython(item)]));
  return value;
}
async function interpreter(): Promise<any> {
  singleton ??= (async () => {
    const { loadPyodide } = await import('pyodide');
    const node = typeof process !== 'undefined' && Boolean(process.versions?.node);
    const localAssets = node ? new URL('../../vendor/pyodide/', import.meta.url).href :
      new URL('../pyodide/', import.meta.url).href;
    // Pyodide caches packages under packageCacheDir, which defaults to packageBaseUrl; a file: URL there is
    // path-joined literally and creates a `file:/...` directory under the working directory.
    const py = await loadPyodide(node ? { packageBaseUrl: localAssets,
      packageCacheDir: decodeURIComponent(new URL(localAssets).pathname) } :
      { indexURL: localAssets, packageBaseUrl: localAssets });
    py.FS.mkdir('/calls');
    py.FS.writeFile('/lib/python3.14/site-packages/natlang_policy.py', PYTHON_POLICY);
    py.FS.writeFile('/lib/python3.14/site-packages/natlang.py', PYTHON_NATLANG);
    py.registerJsModule('_natlang_host', {
      nl: (instructions: string, returns: string) => {
        if (!currentHost?.nl) throw new Error('nl is unavailable outside a natlang Python call');
        return async (...args: unknown[]) => {
          awaitingChild = true;
          try { return await currentHost!.nl!(instructions, returns)(...args.map(fromPython)); }
          finally { awaitingChild = false; currentRefresh?.(); }
        };
      },
      iterate_on: (step: unknown, initial: unknown, ...fixed: unknown[]) => {
        if (!currentHost?.iterateOn) throw new Error('iterate_on is unavailable outside a natlang Python call');
        return currentHost.iterateOn(step, fromPython(initial), ...fixed.map(fromPython));
      },
    });
    py.runPython(RUNNER);
    return py;
  })();
  return singleton;
}

/** Run a policy-checked cell. Calls share an interpreter but keep separate folders and Python globals. */
export async function runFolderPython(folder: Folder, source: string, host: PythonHost = {}, timeoutMs = 30_000): Promise<PythonResult> {
  if (awaitingChild) throw new Error('a Python child cannot reenter the shared interpreter while its parent awaits nl');
  const previous = tail;
  let unlock!: () => void;
  tail = new Promise<void>(resolve => { unlock = resolve; });
  await previous;
  let py: any;
  const before = new Map(folder.diffSync().changes.map(change => [change.path, change.after]));
  const mount = `/calls/${++nextCall}`;
  const stdout: string[] = [], stderr: string[] = [];
  let watchdog: { terminate(): Promise<unknown> | unknown } | undefined;
  try {
    py = await interpreter();
    py.setStdout({ batched: (text: string) => stdout.push(text) });
    py.setStderr({ batched: (text: string) => stderr.push(text) });
    currentHost = host;
    await py.loadPackagesFromImports(source, { messageCallback: () => {} });
    py.FS.mkdir(mount);
    const adapter = folderFS(py, folderView(folder));
    currentRefresh = adapter.refresh;
    py.FS.mount(adapter, {}, mount);
    py.FS.chdir(mount);
    const namespace = py.runPython('import natlang_policy; natlang_policy.fresh_namespace()');
    namespace.set('__natlang_source', source);
    namespace.set('__natlang_run', py.globals.get('__natlang_run'));
    if (typeof SharedArrayBuffer !== 'undefined') {
      const buffer = new SharedArrayBuffer(4);
      py.setInterruptBuffer(new Int32Array(buffer));
      const script = 'const flag = new Int32Array(buffer); setTimeout(() => Atomics.store(flag, 0, 2), ms)';
      if (typeof process !== 'undefined' && process.versions?.node) {
        const { Worker } = await import('node:worker_threads');
        watchdog = new Worker(`const { workerData } = require('worker_threads'); const buffer = workerData.buffer, ms = workerData.ms; ${script}`,
          { eval: true, workerData: { buffer, ms: timeoutMs } });
      } else if (typeof Worker !== 'undefined') {
        const workerUrl = URL.createObjectURL(new Blob([`onmessage = ({ data }) => { const { buffer, ms } = data; ${script}; };`],
          { type: 'text/javascript' }));
        const worker = new Worker(workerUrl);
        worker.postMessage({ buffer, ms: timeoutMs });
        watchdog = { terminate: () => { worker.terminate(); URL.revokeObjectURL(workerUrl); } };
      }
    }
    let value: any;
    try {
      value = await py.runPythonAsync('await __natlang_run(__natlang_source, globals())', { globals: namespace });
      const openWriter = py.FS.streams.find((stream: any) => stream?.path?.startsWith(`${mount}/`) && (stream.flags & 3) !== 0);
      if (openWriter) throw new Error(`close the Python file or sqlite connection before leaving the cell: ${openWriter.path}`);
    }
    catch (error) {
      if (String(error).includes('KeyboardInterrupt')) throw new Error(`Python timed out after ${timeoutMs} ms`);
      throw error;
    }
    finally { namespace.destroy(); }
    const converted = fromPython(value);
    if (converted && typeof converted === 'object' && (converted as Record<string, unknown>).__natlang_timeout__)
      throw new Error(`Python timed out after ${timeoutMs} ms`);
    const changedPaths = folder.diffSync().changes.filter(change => {
      const prior = before.get(change.path), after = change.after;
      return !before.has(change.path) || prior?.length !== after?.length || prior?.some((byte, i) => byte !== after?.[i]);
    }).map(change => change.path);
    return { value: converted, stdout: stdout.join('\n'), stderr: stderr.join('\n'), changedPaths };
  } finally {
    currentHost = undefined;
    currentRefresh = undefined;
    if (py) {
      py.FS.chdir('/');
      try { py.FS.unmount(mount); py.FS.rmdir(mount); } catch { /* cleanup after a failed mount */ }
    }
    if (watchdog) await watchdog.terminate();
    unlock();
  }
}
