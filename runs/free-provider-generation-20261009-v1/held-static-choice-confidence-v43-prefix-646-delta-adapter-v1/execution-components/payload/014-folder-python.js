/** One warm Pyodide interpreter over per-call mounts into Folder overlays. */
import { folderFS, folderView } from './python-folderfs.js';
import { PYTHON_NATLANG, PYTHON_POLICY } from './python-sources.js';
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
export const WATCHDOG_SCRIPT = 'const flag = new Int32Array(buffer); setTimeout(() => Atomics.store(flag, 0, 2), ms)';
let startWatchdog = (buffer, ms) => {
    if (typeof Worker === 'undefined')
        return undefined;
    const workerUrl = URL.createObjectURL(new Blob([`onmessage = ({ data }) => { const { buffer, ms } = data; ${WATCHDOG_SCRIPT}; };`], { type: 'text/javascript' }));
    const worker = new Worker(workerUrl);
    worker.postMessage({ buffer, ms });
    return { terminate: () => { worker.terminate(); URL.revokeObjectURL(workerUrl); } };
};
export function setPythonWatchdog(start) { startWatchdog = start; }
let singleton;
let currentHost;
let currentRefresh;
let awaitingChild = false;
let nextCall = 0;
let tail = Promise.resolve();
function fromPython(value) {
    if (value && typeof value.toJs === 'function')
        return fromPython(value.toJs({ dict_converter: Object.fromEntries }));
    if (typeof value === 'bigint')
        return Number.isSafeInteger(Number(value)) ? Number(value) : value.toString();
    if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
        const items = Array.from(value, fromPython);
        return items.length === 1 ? items[0] : items;
    }
    if (Array.isArray(value))
        return value.map(fromPython);
    if (value instanceof Map)
        return Object.fromEntries([...value].map(([key, item]) => [String(key), fromPython(item)]));
    if (value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype)
        return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, fromPython(item)]));
    return value;
}
async function interpreter() {
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
            nl: (instructions, returns) => {
                if (!currentHost?.nl)
                    throw new Error('nl is unavailable outside a natlang Python call');
                return async (...args) => {
                    awaitingChild = true;
                    try {
                        return await currentHost.nl(instructions, returns)(...args.map(fromPython));
                    }
                    finally {
                        awaitingChild = false;
                        currentRefresh?.();
                    }
                };
            },
            iterate_on: (step, initial, ...fixed) => {
                if (!currentHost?.iterateOn)
                    throw new Error('iterate_on is unavailable outside a natlang Python call');
                return currentHost.iterateOn(step, fromPython(initial), ...fixed.map(fromPython));
            },
        });
        py.runPython(RUNNER);
        return py;
    })();
    return singleton;
}
/** Run a policy-checked cell. Calls share an interpreter but keep separate folders and Python globals. */
export async function runFolderPython(folder, source, host = {}, timeoutMs = 30_000) {
    if (awaitingChild)
        throw new Error('a Python child cannot reenter the shared interpreter while its parent awaits nl');
    const previous = tail;
    let unlock;
    tail = new Promise(resolve => { unlock = resolve; });
    await previous;
    let py;
    const before = new Map(folder.diffSync().changes.map(change => [change.path, change.after]));
    const mount = `/calls/${++nextCall}`;
    const stdout = [], stderr = [];
    let watchdog;
    try {
        py = await interpreter();
        py.setStdout({ batched: (text) => stdout.push(text) });
        py.setStderr({ batched: (text) => stderr.push(text) });
        currentHost = host;
        await py.loadPackagesFromImports(source, { messageCallback: () => { } });
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
            watchdog = startWatchdog(buffer, timeoutMs);
        }
        let value;
        try {
            value = await py.runPythonAsync('await __natlang_run(__natlang_source, globals())', { globals: namespace });
            const openWriter = py.FS.streams.find((stream) => stream?.path?.startsWith(`${mount}/`) && (stream.flags & 3) !== 0);
            if (openWriter)
                throw new Error(`close the Python file or sqlite connection before leaving the cell: ${openWriter.path}`);
        }
        catch (error) {
            if (String(error).includes('KeyboardInterrupt'))
                throw new Error(`Python timed out after ${timeoutMs} ms`);
            throw error;
        }
        finally {
            namespace.destroy();
        }
        const converted = fromPython(value);
        if (converted && typeof converted === 'object' && converted.__natlang_timeout__)
            throw new Error(`Python timed out after ${timeoutMs} ms`);
        const changedPaths = folder.diffSync().changes.filter(change => {
            const prior = before.get(change.path), after = change.after;
            return !before.has(change.path) || prior?.length !== after?.length || prior?.some((byte, i) => byte !== after?.[i]);
        }).map(change => change.path);
        return { value: converted, stdout: stdout.join('\n'), stderr: stderr.join('\n'), changedPaths };
    }
    finally {
        currentHost = undefined;
        currentRefresh = undefined;
        if (py) {
            py.FS.chdir('/');
            try {
                py.FS.unmount(mount);
                py.FS.rmdir(mount);
            }
            catch { /* cleanup after a failed mount */ }
        }
        if (watchdog)
            await watchdog.terminate();
        unlock();
    }
}
