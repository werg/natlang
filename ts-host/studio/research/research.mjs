import { BrowserNatlangClient, BrowserNatlangApplication, BrowserDomRenderer, loadBrowserModelCatalog } from '../../dist/browser/natlang.js';
import { StudioStore } from '../shared/store.mjs';
import { runChild } from '../shared/child-runner.mjs';
import { download } from '../shared/render.mjs';
import { ResearchHost } from './host.mjs';

const $ = id => document.getElementById(id);
const programFiles = ['types.ts', 'reduce.nl', 'view.nl', 'learn.nl', 'revise_schema.nl', 'invent_interaction.nl', 'preserve_intent.nl', 'investigate_beliefs.nl', 'reduce/list.ts', 'reduce/search.ts', 'reduce/read.ts', 'reduce/commit.ts', 'reduce/execute.ts', 'reduce/diff.ts', 'reduce/propose.ts', 'reduce/activate.ts', 'reduce/receipt.ts'];
const store = await StudioStore.open();
let client, app, controllerHead = '', abort, busy = false, renderer, currentInteraction, state, generatedDrafts = {};
const status = (message, error = false) => { $('status').textContent = String(message); $('status').style.color = error ? '#9b442e' : ''; };
const initial = head => ({ revision: 0, head, question: '', notice: 'Ready to investigate.', active_view: '', selected: '', receipts: [] });
const research = new ResearchHost({ store, runSource: async (files, root, inputs) => {
    const result = await runChild({ request: { source: { kind: 'files', root, files: Object.fromEntries(files.map(row => [row.id, row.source])) }, inputs,
        options: { seed: { mode: 'derived', root: Number($('seed').value) } } } },
    { model: client?.model, signal: abort?.signal, onProgress: status });
    const trace_id = crypto.randomUUID();
    await store.put('child_runs', trace_id, { ...result, files, root, inputs });
    return { value: result.value, trace_id };
} });
async function recover(current) {
    const workspace = await research.runtime.workspace.snapshot();
    const linked = new Set(current.receipts);
    const orphaned = (await store.all('effects')).filter(receipt => receipt?.root && workspace.manifests[receipt.manifest] && receipt.status !== 'running' && !linked.has(receipt.id)).map(receipt => receipt.id);
    if (current.head === workspace.head && !orphaned.length) return current;
    const next = { ...current, head: workspace.head, receipts: [...current.receipts, ...orphaned],
        notice: `Recovered workspace changes${orphaned.length ? ` and ${orphaned.length} execution receipts` : ''} after an interrupted reduction.` };
    await store.put('states', 'research', { state: next, revision: next.revision, recovery: true });
    return next;
}
client = new BrowserNatlangClient({ host: { research: research.api() }, mode: 'retained',
    wasmUrl: '/ts-host/dist/browser/wllama.wasm', compatWorkerUrl: '/ts-host/dist/browser/wllama-compat.js', compatWasmUrl: '/ts-host/dist/browser/wllama-compat.wasm' });

async function bootstrap() {
    let head = await research.runtime.workspace.head();
    if (!head) {
        const edits = Object.fromEntries(await Promise.all(programFiles.map(async path => {
            const response = await fetch(`./programs/${path}`);
            if (!response.ok) throw new Error(`Missing research program ${path}`);
            return [path, { kind: path === 'types.ts' ? 'schema' : 'source', content: await response.text() }];
        })));
        const [observations, methods] = await Promise.all([
            fetch('./sample/observations.json').then(r => r.json()),
            fetch('./sample/methods.txt').then(r => r.text()),
        ]);
        edits['evidence/reliability-observations.json'] = { kind: 'evidence', content: observations };
        edits['evidence/reliability-methods.txt'] = { kind: 'evidence', content: methods };
        head = await research.runtime.commit('', edits, { message: 'Initial research program and example evidence' });
    }
    const saved = await store.get('states', 'research');
    generatedDrafts = await store.get('drafts', 'research-interaction') ?? {};
    state = await recover(saved?.state ?? initial(head.id));
    await paint(state);
}
function setBusy(value) {
    busy = value;
    $('question-form').querySelector('[type=submit]').disabled = value;
    $('load-model').disabled = value;
    $('cancel').hidden = !value;
}
async function sourceAt(head) {
    const entries = (await research.runtime.list(head, ['source', 'schema'])).filter(row => /\.(nl|ts)$/.test(row.path));
    const files = Object.fromEntries(await Promise.all(entries.map(async row => [row.path, (await research.runtime.read(head, row.path)).content])));
    for (const path of programFiles) if (!files[path]) throw new Error(`Research controller is missing ${path}`);
    return { files, reducer: 'reduce.nl', view: 'view.nl' };
}
async function mount() {
    if (app) await app.close();
    controllerHead = state.head;
    app = new BrowserNatlangApplication({ client, source: await sourceAt(state.head), initialState: state,
        initialRevision: state.revision, seedRoot: Number($('seed').value),
        onCommit: async commit => {
            await research.verifyCommit(state, commit.state);
            state = commit.state;
            await store.commit('research', { state, revision: state.revision, event: commit.event,
                trace: commit.reducerRun.trace, sourceHead: controllerHead, model: client.modelStatus?.id ?? '', seed: Number($('seed').value) });
        },
        onTransition: transition => { state = transition.state; void paint(state, transition.view, transition.reducerRun?.trace); },
        onFailure: failure => status(`${failure.stage}: ${failure.detail}`, true),
    });
    await app.start();
}
async function paint(current, view, trace) {
    $('heading').textContent = view?.heading || 'Ask something worth investigating.';
    $('summary').textContent = view?.summary || current.notice;
    $('revision').textContent = `Event ${current.revision} · ${current.head.slice(0, 12)}`;
    if (trace) $('trace').textContent = JSON.stringify(trace, null, 2);
    const entries = await research.runtime.list(current.head);
    $('artifact-count').textContent = `${entries.length} artifacts`;
    const root = $('artifacts'); root.replaceChildren();
    const icons = { source: '⌘', schema: '◇', method: '✧', evidence: '▤', claim: '❧', view: '◫', data: '▥', migration: '⇄', assessment: '◌', intent: '◈', change: '↗' };
    for (const row of entries.filter(row => !programFiles.includes(row.path))) {
        const button = document.createElement('button'); button.className = 'artifact';
        const symbol = document.createElement('span'); symbol.className = 'symbol'; symbol.textContent = icons[row.kind] ?? '◦';
        const label = document.createElement('span'), strong = document.createElement('strong'), small = document.createElement('small');
        strong.textContent = row.path; small.textContent = `${row.kind} · ${row.id.slice(0, 12)}`;
        label.append(strong, small); button.append(symbol, label);
        button.onclick = async () => showDetail(row.path, (await research.runtime.read(current.head, row.path)).content);
        root.append(button);
    }
    if (!root.children.length) root.textContent = 'Your methods, evidence, conclusions and views will appear here.';
    const receipts = $('receipts'); receipts.replaceChildren();
    for (const id of current.receipts.slice(-12).reverse()) {
        const receipt = await store.readEffect(id);
        const div = document.createElement('div'); div.className = `receipt ${receipt?.status ?? 'missing'}`;
        div.textContent = `${receipt?.status ?? 'missing'} · ${receipt?.root ?? id} · ${id}`;
        div.onclick = () => showDetail(id, receipt);
        receipts.append(div);
    }
    const savedInputs = new Map([...$('interaction-root').querySelectorAll('input[id]')].map(input => [input.id, input.value]));
    const focusedInput = $('interaction-root').contains(document.activeElement) ? document.activeElement?.id : '';
    if (renderer) renderer.close();
    renderer = null;
    currentInteraction = null;
    const viewPath = view?.active_view || current.active_view;
    if (viewPath) {
        try {
            currentInteraction = await research.runtime.interaction(current.head, viewPath);
            const pinned = currentInteraction;
            renderer = new BrowserDomRenderer($('interaction-root'), event => onGenerated(pinned, event), error => status(error, true));
            renderer.render(pinned.tree);
            for (const input of $('interaction-root').querySelectorAll('input[id]'))
                if (savedInputs.has(input.id)) input.value = savedInputs.get(input.id);
                else if (Object.hasOwn(generatedDrafts, input.id)) input.value = generatedDrafts[input.id];
            if (focusedInput) document.getElementById(focusedInput)?.focus({ preventScroll: true });
        }
        catch (error) { $('interaction-root').textContent = `Generated view could not load: ${error}`; }
    }
    else $('interaction-root').textContent = 'The program can develop an interaction for this investigation.';
}
function showDetail(title, value) {
    $('detail-title').textContent = title;
    $('detail-body').textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    $('detail').showModal();
}
async function onGenerated(pinned, event) {
    const binding = pinned.bindings[event.kind];
    if (!binding) throw new Error('Control has no natlang handler');
    if (pinned.revision !== state.head) throw new Error('This interaction changed; review the current version');
    await submit('generated', async id => {
        const receipt = await research.runtime.execute(pinned.revision, binding.root,
            { value: event.value ?? '', context: JSON.stringify({ question: state.question, head: state.head }) }, `${id}/handler`);
        research.effects.push(receipt.id);
        return JSON.stringify({ control: event.kind, value: event.value ?? '', receipt: receipt.id,
            status: receipt.status, result: receipt.status === 'complete' ? receipt.value : receipt.error });
    });
}
async function submit(kind, value) {
    const run = async () => {
        if (busy) throw new Error('An investigation is already running');
        if (!client.model) throw new Error('Load the interpreter to investigate');
        const recovered = await recover(state);
        if (recovered.head !== state.head || recovered.receipts.length !== state.receipts.length) {
            state = recovered; await paint(state); await mount();
            throw new Error('Workspace or receipts changed; review them and submit again');
        }
        if (controllerHead !== state.head) await mount();
        setBusy(true); abort = new AbortController();
        const id = crypto.randomUUID(), event = { id, kind, value: '' };
        research.begin(event);
        try {
            event.value = typeof value === 'function' ? await value(id) : value;
            status('Natlang is investigating…');
            await app.dispatch(event);
            status(state.notice);
        }
        finally { research.end(); abort = null; setBusy(false); }
    };
    if (navigator.locks) return navigator.locks.request('natlang-research-writer', run);
    return run();
}

const catalog = await loadBrowserModelCatalog();
for (const model of catalog.models) $('model').add(new Option(model.label, model.id));
$('model').value = catalog.defaultId;
$('load-model').onclick = async () => {
    try {
        setBusy(true); status('Loading local interpreter…');
        const model = catalog.models.find(row => row.id === $('model').value);
        await client.loadModel({ kind: 'url', id: model.id, url: model.url, templateUrl: model.templateUrl },
            { contextTokens: model.contextTokens ?? 32768, gpuLayers: $('compute').value === 'gpu' ? 99999 : 0 });
        $('model-state').textContent = model.label;
        await mount(); status('Interpreter ready.');
    }
    catch (error) { status(error, true); }
    finally { setBusy(false); }
};
$('question-form').onsubmit = async event => {
    event.preventDefault(); const text = $('question').value.trim(); if (!text) return;
    try { await submit('question', text); if ($('question').value.trim() === text) $('question').value = ''; }
    catch (error) { status(error, true); }
};
$('cancel').onclick = () => { abort?.abort(); app?.cancel(); status('Stopping the investigation; recorded effects remain inspectable.'); };
$('source').onclick = async () => showDetail('Research program', (await sourceAt(state.head)).files);
$('interaction-root').addEventListener('input', event => {
    if (event.target instanceof HTMLInputElement && event.target.id) {
        generatedDrafts[event.target.id] = event.target.value;
        void store.put('drafts', 'research-interaction', generatedDrafts).catch(error => status(error, true));
    }
});
$('evidence').onclick = () => $('evidence-file').click();
$('evidence-file').onchange = async () => {
    try {
        const file = $('evidence-file').files?.[0]; if (!file) return;
        const raw = await file.text();
        const content = file.name.endsWith('.json') ? JSON.parse(raw) : raw;
        const name = file.name.replace(/[^A-Za-z0-9_.-]/g, '_');
        const path = `evidence/${crypto.randomUUID().slice(0, 8)}-${name}`;
        const commit = await research.runtime.commit(state.head, { [path]: { kind: 'evidence', content } }, { message: `Imported ${file.name}` });
        state = { ...state, head: commit.id, revision: state.revision + 1, notice: `Imported ${file.name}. Ask natlang what it changes.` };
        await store.commit('research', { state, revision: state.revision, event: { kind: 'import', value: path } });
        await paint(state); status(state.notice);
    }
    catch (error) { status(error, true); }
    finally { $('evidence-file').value = ''; }
};
$('export').onclick = async () => download('natlang-research.json', JSON.stringify({ workspace: await research.runtime.workspace.export(), state }, null, 2), 'application/json');
$('import').onclick = () => $('import-file').click();
$('import-file').onchange = async () => {
    try {
        const file = $('import-file').files?.[0]; if (!file) return;
        const bundle = JSON.parse(await file.text());
        const current = await research.runtime.workspace.head();
        const currentFiles = Object.keys(current?.files ?? {});
        if (currentFiles.some(path => !programFiles.includes(path) && !path.startsWith('evidence/reliability-')) || state.revision)
            throw new Error('Import into a fresh research workspace');
        if (bundle.state?.head !== bundle.workspace?.head) throw new Error('Imported state and workspace disagree');
        await research.runtime.workspace.import(bundle.workspace, current?.id ?? '');
        state = bundle.state;
        await store.put('states', 'research', { state, revision: state.revision, imported: true });
        await paint(state);
        status('Workspace imported.');
    }
    catch (error) { status(error, true); }
};
await bootstrap();
