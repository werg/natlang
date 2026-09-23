/**
 * Studio applications on the browser runtime: an event loop whose reducer and view are items of a
 * natlang program held in memory, a local model loaded on demand, and host services for the program.
 */
import { EventLoop, createNatlangRuntime, loadBrowserLocalModel, loadVirtualCallables } from '../../dist/browser/natlang.js';

const MODEL_ASSETS = { wasmUrl: '/ts-host/dist/browser/wllama.wasm', compatWorkerUrl: '/ts-host/dist/browser/wllama-compat.js',
    compatWasmUrl: '/ts-host/dist/browser/wllama-compat.wasm' };

/** The loaded local interpreter model, shared by every application on the page. */
export class StudioModel {
    constructor() { this.current = null; this.status = null; }
    get loaded() { return !!this.current?.loaded; }
    get id() { return this.status?.id ?? null; }
    /** A model driver; turns fail until a model is loaded. */
    turn = (request, signal) => {
        if (!this.current) throw new Error('Load a local interpreter model first');
        return this.current.turn(request, signal);
    };
    async load(entry, options) {
        await this.current?.close();
        this.current = null; this.status = null;
        const loaded = await loadBrowserLocalModel({ kind: 'url', id: entry.id, url: entry.url, templateUrl: entry.templateUrl },
            { ...options, engine: MODEL_ASSETS });
        this.current = loaded.model; this.status = loaded.status;
        return loaded.status;
    }
}

/**
 * An event loop over `source` ({ files, reducer, view }): `reducer` and `view` name items of the
 * program's root folder. Each step runs as a natlang task; the committed transition carries the
 * reduction's traces.
 */
export function natlangApplication({ source, model, services, seedRoot, initialState, initialRevision = 0, onCommit, onTransition, onFailure }) {
    const program = loadVirtualCallables(source.files);
    const item = path => {
        const fn = program[path.replace(/\.(nl|ts)$/, '')];
        if (typeof fn !== 'function') throw new Error(`The program has no callable ${path}`);
        return fn;
    };
    const reduce = item(source.reducer), view = item(source.view);
    const runtime = createNatlangRuntime({ model, services, seed: { mode: 'derived', root: seedRoot } });
    let traces = [];
    const loop = new EventLoop({ initialState, initialRevision,
        reduce: (state, event) => reduce(state, event), view: state => view(state),
        step: (fn, context) => {
            if (context.stage === 'reduce') traces = [];
            return runtime.run(fn, { signal: context.signal, trace: trace => { if (context.stage === 'reduce') traces.push(trace); } });
        },
        onCommit: commit => onCommit?.({ ...commit, trace: traces.find(trace => trace.parentCallId === null)?.events ?? [], invocations: traces }),
        onTransition, onFailure });
    return loop;
}
