/** User programs run off the UI thread. Model turns borrow the shared main-thread model. */
import { simulateInventory } from '../apps/worlds.mjs';
import * as natlang from '../../dist/browser/natlang.js';
import { StudioStore } from './store.mjs';
const { TypeScriptEnvironment, createNatlangRuntime, newPlaygroundProject, runPlaygroundProject } = natlang;
const pending = new Map();
let next = 0;
const modelTurn = request => new Promise((resolve, reject) => { const id = ++next; pending.set(id, { resolve, reject }); self.postMessage({ kind: 'model-turn', id, request }); });
self.onmessage = async ({ data }) => {
    if (data.kind === 'model-result') {
        const item = pending.get(data.id);
        if (!item)
            return;
        pending.delete(data.id);
        data.error ? item.reject(new Error(data.error)) : item.resolve(data.result);
        return;
    }
    if (data.kind !== 'run')
        return;
    let nativeStore;
    try {
        if (data.experiment) {
            self.postMessage({ kind: 'result', result: { value: simulateInventory(data.experiment), trace: [] } });
            return;
        }
        if (data.cell) {
            const environment = new TypeScriptEnvironment({ mode: 'fresh' });
            try {
                const result = await environment.executeAsync({ code: `const deps = self.args.deps;\n${data.cell.source}`, body: true, path: `notebook/${data.cell.id}`, scope: { args: { deps: data.deps } } });
                self.postMessage({ kind: 'result', result: { value: result.result, trace: [] } });
            }
            finally { environment.close(); }
            return;
        }
        const allowed = new Set(data.researchNative ?? []);
        nativeStore = allowed.size ? await StudioStore.open() : null;
        const services = nativeStore ? { research: {
            readNative: async id => {
                if (!allowed.has(id)) throw new Error('Native evidence is outside this manifest');
                const record = await nativeStore.get('native_values', id);
                if (!record) throw new Error('Native evidence is missing');
                return record.value;
            },
        } } : {};
        const { root, files, inputs, seed } = data.request;
        const runtime = createNatlangRuntime({ model: modelTurn, services, seed });
        const run = await runPlaygroundProject(runtime, newPlaygroundProject('child', root, files, inputs), { runtimeNamespace: natlang });
        if (run.outcome.kind !== 'done')
            throw new Error(run.outcome.detail);
        self.postMessage({ kind: 'result', result: { value: run.value, trace: run.trace, invocations: run.invocations } });
    }
    catch (error) {
        self.postMessage({ kind: 'error', error: String(error) });
    }
    finally {
        nativeStore?.close();
    }
};
