/** User programs run off the UI thread. Model turns borrow the shared main-thread model. */
import { simulateInventory } from '../apps/worlds.mjs';
import { BrowserNatlangHost } from '../../dist/browser/natlang.js';
import { StudioStore } from './store.mjs';
const pending = new Map();
let next = 0;
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
    let host, nativeStore;
    try {
        const allowed = new Set(data.researchNative ?? []);
        nativeStore = allowed.size ? await StudioStore.open() : null;
        const researchHost = nativeStore ? { research: {
            readNative: async id => {
                if (!allowed.has(id)) throw new Error('Native evidence is outside this manifest');
                const record = await nativeStore.get('native_values', id);
                if (!record) throw new Error('Native evidence is missing');
                return record.value;
            },
        } } : {};
        host = new BrowserNatlangHost({ host: researchHost });
        if(data.experiment){self.postMessage({kind:'result',result:{value:simulateInventory(data.experiment),trace:[]}});}
        else if (data.cell) {
            const result = await host.environment.executeAsync({ code: data.cell.source, body: true, path: `notebook/${data.cell.id}`, effectful: true, scope: { args: { deps: data.deps }, let: {} } });
            self.postMessage({ kind: 'result', result: { value: result.result, trace: [] } });
        }
        else {
            const modelTurn = request => new Promise((resolve, reject) => { const id = ++next; pending.set(id, { resolve, reject }); self.postMessage({ kind: 'model-turn', id, request }); });
            const result = await host.run({ ...data.request, modelTurn });
            if (result.outcome.kind !== 'done')
                throw new Error(result.outcome.detail);
            self.postMessage({ kind: 'result', result });
        }
    }
    catch (error) {
        self.postMessage({ kind: 'error', error: String(error) });
    }
    finally {
        await host?.close();
        nativeStore?.close();
    }
};
