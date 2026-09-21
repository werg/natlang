import { BrowserNatlangClient, BrowserNatlangApplication, BrowserDomRenderer, loadBrowserModelCatalog } from '../../dist/browser/natlang.js';
import { StudioStore } from '../shared/store.mjs';
import { runChild } from '../shared/child-runner.mjs';
import { download } from '../shared/render.mjs';
import { ResearchHost } from './host.mjs';

const $ = id => document.getElementById(id);
const programFiles = ['types.ts', 'reduce.nl', 'view.nl', 'learn.nl', 'revise_schema.nl', 'invent_interaction.nl', 'preserve_intent.nl', 'investigate_beliefs.nl', 'reduce/list.ts', 'reduce/search.ts', 'reduce/read.ts', 'reduce/commit.ts', 'reduce/execute.ts', 'reduce/diff.ts', 'reduce/review_candidate.ts', 'reduce/propose.ts', 'reduce/activate.ts', 'reduce/receipt.ts', 'reduce/branches.ts', 'reduce/belief_graph.ts', 'reduce/affected.ts', 'reduce/audit_migration.ts', 'reduce/native_read.ts', 'reduce/native_search.ts'];
const store = await StudioStore.open();
let client, app, controllerHead = '', abort, busy = false, renderer, currentInteraction, state, generatedDrafts = {};
let selectedMethod = '';
let reviewedCandidate = '';
const status = (message, error = false) => { $('status').textContent = String(message); $('status').style.color = error ? '#9b442e' : ''; };
const initial = head => ({ revision: 0, head, question: '', notice: 'Ready to investigate.', active_view: '', selected: '', receipts: [] });
const research = new ResearchHost({ store, runContext: () => ({
    model: client?.modelStatus?.id ?? '', seed: { mode: 'derived', root: Number($('seed').value) },
    evaluator: 'typescript-host/browser',
}), runSource: async (files, root, inputs, options) => {
    const provenance = options.provenance;
    const result = await runChild({ request: { source: { kind: 'files', root, files: Object.fromEntries(files.map(row => [row.id, row.source])) }, inputs,
        options: { seed: provenance.seed } }, researchNative: options?.native_ids ?? [] },
    { model: client?.model, signal: abort?.signal, onProgress: status });
    const trace_id = crypto.randomUUID();
    await store.put('child_runs', trace_id, { ...result, files, root, inputs, provenance });
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
    const methods = $('methods'); methods.replaceChildren();
    for (const row of entries.filter(row => row.kind === 'source' && !programFiles.includes(row.path) && /\.(nl|ts)$/.test(row.path))) {
        const card = document.createElement('div'); card.className = 'method';
        const text = document.createElement('div'), title = document.createElement('strong'), meta = document.createElement('small');
        title.textContent = row.path; meta.textContent = `${row.path.endsWith('.nl') ? 'natlang' : 'crisp TypeScript'} · ${row.id.slice(0, 12)}`;
        text.append(title, meta);
        const run = document.createElement('button'); run.textContent = 'Run';
        run.onclick = () => { selectedMethod = row.path; $('method-title').textContent = row.path;
            $('method-result').textContent = ''; $('method-status').textContent = ''; $('method-runner').showModal(); };
        card.append(text, run); methods.append(card);
    }
    if (!methods.children.length) methods.textContent = 'Natlang can develop, exercise, and retain a method during an investigation.';
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
    const graph = await research.runtime.beliefGraph(current.head);
    const beliefs = $('beliefs'); beliefs.replaceChildren();
    for (const node of graph.nodes.filter(row => row.kind === 'claim' || row.kind === 'assessment')) {
        const links = graph.links.filter(link => link.from === node.path);
        const card = document.createElement('div'); card.className = 'belief';
        const title = document.createElement('strong'), meta = document.createElement('small');
        title.textContent = node.title;
        meta.textContent = `${node.status || 'unreviewed'} · ${links.filter(link => link.relation === 'supports').length} supporting · ${links.filter(link => link.relation === 'opposes').length} opposing`;
        card.append(title, meta); beliefs.append(card);
    }
    if (graph.missing.length) {
        const warning = document.createElement('div'); warning.className = 'belief';
        warning.textContent = `${graph.missing.length} evidence links point to missing artifacts.`; beliefs.append(warning);
    }
    if (graph.invalid.length) {
        const warning = document.createElement('div'); warning.className = 'belief';
        warning.textContent = `${graph.invalid.length} belief links have an invalid shape. Inspect their artifacts.`; beliefs.append(warning);
    }
    if (!beliefs.children.length) beliefs.textContent = 'Conclusions and their evidence will appear here.';
    const branches = $('branches'); branches.replaceChildren();
    for (const candidate of (await research.runtime.branches()).filter(row => row.kind === 'candidate')) {
        const button = document.createElement('button'); button.className = 'branch';
        const title = document.createElement('strong'), meta = document.createElement('small');
        title.textContent = candidate.message || 'Untitled candidate';
        meta.textContent = `${candidate.changed} changed artifacts · ${candidate.id.slice(0, 12)}`;
        button.append(title, meta);
        button.onclick = async () => {
            const review = await research.runtime.reviewCandidate(candidate.id);
            showDetail(candidate.message || 'Candidate', review); reviewedCandidate = candidate.id;
            $('detail-activate').hidden = !review.can_activate;
        };
        branches.append(button);
    }
    if (!branches.children.length) branches.textContent = 'Proposed alternatives will appear here.';
    const receipts = $('receipts'); receipts.replaceChildren();
    for (const id of current.receipts.slice(-12).reverse()) {
        const receipt = await store.readEffect(id);
        const div = document.createElement('div'); div.className = `receipt ${receipt?.status ?? 'missing'}`;
        div.textContent = `${receipt?.status ?? 'missing'}${receipt?.imported ? ' · imported record' : ''} · ${receipt?.root ?? id} · ${id}`;
        div.onclick = () => showDetail(id, receipt);
        receipts.append(div);
    }
    const savedInputs = new Map([...$('interaction-root').querySelectorAll('input[id],select[id],textarea[id]')].map(input => [input.id, input.value]));
    const focusedInput = $('interaction-root').contains(document.activeElement) ? document.activeElement?.id : '';
    const viewPath = view?.active_view || current.active_view;
    if (viewPath) {
        try {
            const pinned = await research.runtime.interaction(current.head, viewPath);
            const candidateRoot = document.createElement('div');
            const candidate = new BrowserDomRenderer(candidateRoot, event => onGenerated(pinned, event), error => status(error, true));
            candidate.render(pinned.tree);
            for (const input of candidateRoot.querySelectorAll('input[id],select[id],textarea[id]'))
                if (savedInputs.has(input.id)) input.value = savedInputs.get(input.id);
                else if (Object.hasOwn(generatedDrafts, input.id)) input.value = generatedDrafts[input.id];
            renderer?.close();
            $('interaction-root').replaceChildren(...candidateRoot.childNodes);
            renderer = candidate;
            currentInteraction = pinned;
            if (focusedInput) document.getElementById(focusedInput)?.focus({ preventScroll: true });
        }
        catch (error) { status(`Generated view could not load; previous interaction remains available for inspection: ${error}`, true); }
    }
    else { renderer?.close(); renderer = null; currentInteraction = null;
        $('interaction-root').textContent = 'The program can develop an interaction for this investigation.'; }
}
function showDetail(title, value) {
    reviewedCandidate = '';
    $('detail-title').textContent = title;
    $('detail-body').textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    const reference = value && typeof value === 'object' && value.native_id ? value : null;
    $('detail-download').hidden = !reference;
    $('detail-activate').hidden = true;
    $('detail-download').onclick = async () => {
        const record = await store.get('native_values', reference.native_id);
        if (!record) throw new Error('Full native evidence is missing');
        download(reference.name || 'evidence.txt', record.value, record.type || 'text/plain');
    };
    $('detail').showModal();
}
$('detail-activate').onclick = async () => {
    if (!reviewedCandidate || busy) return;
    try {
        const activated = await research.runtime.activate(state.head, reviewedCandidate);
        state = { ...state, head: activated.id, revision: state.revision + 1,
            notice: `Activated reviewed candidate ${activated.id.slice(0, 12)}.` };
        await store.commit('research', { state, revision: state.revision,
            event: { kind: 'activate-candidate', value: reviewedCandidate } });
        $('detail').close(); await paint(state); status(state.notice);
    }
    catch (error) { status(error, true); }
};
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
$('method-execute').onclick = async () => {
    if (!selectedMethod || busy) return;
    let inputs;
    try { inputs = JSON.parse($('method-input').value); }
    catch (error) { $('method-status').textContent = `Invalid JSON: ${error}`; return; }
    try {
        setBusy(true); abort = new AbortController(); $('method-execute').disabled = true;
        $('method-status').textContent = 'Executing pinned source…';
        const id = `manual-${crypto.randomUUID()}`;
        const receipt = await research.runtime.execute(state.head, selectedMethod, inputs, id);
        state = { ...state, revision: state.revision + 1, receipts: [...state.receipts, receipt.id],
            notice: `${selectedMethod} ${receipt.status}; its receipt is preserved.` };
        await store.commit('research', { state, revision: state.revision,
            event: { id, kind: 'method-invocation', value: { root: selectedMethod, inputs } } });
        $('method-result').textContent = JSON.stringify(receipt, null, 2);
        $('method-status').textContent = receipt.status;
        await paint(state);
    }
    catch (error) { $('method-status').textContent = String(error); }
    finally { abort = null; setBusy(false); $('method-execute').disabled = false; }
};
$('source').onclick = async () => showDetail('Research program', (await sourceAt(state.head)).files);
$('interaction-root').addEventListener('input', event => {
    if ((event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement || event.target instanceof HTMLTextAreaElement) && event.target.id) {
        generatedDrafts[event.target.id] = event.target.value;
        void store.put('drafts', 'research-interaction', generatedDrafts).catch(error => status(error, true));
    }
});
$('evidence').onclick = () => $('evidence-file').click();
$('evidence-file').onchange = async () => {
    try {
        const file = $('evidence-file').files?.[0]; if (!file) return;
        const raw = await file.text();
        let content;
        if (raw.length > 256_000) {
            const native_id = `research-${crypto.randomUUID()}`;
            await store.put('native_values', native_id, { value: raw, name: file.name, type: file.type });
            content = { native_id, name: file.name, bytes: file.size, preview: raw.slice(0, 2000) };
        } else content = file.name.endsWith('.json') ? JSON.parse(raw) : raw;
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
$('export').onclick = async () => {
    const workspace = await research.runtime.workspace.export();
    const child_runs = {};
    for (const receipt of Object.values(workspace.receipts))
        if (receipt.trace_id) {
            const run = await store.get('child_runs', receipt.trace_id);
            if (run) child_runs[receipt.trace_id] = run;
        }
    const native_values = {};
    for (const artifact of Object.values(workspace.artifacts)) {
        const id = artifact.content?.native_id;
        if (id) {
            const value = await store.get('native_values', id);
            if (value) native_values[id] = value;
        }
    }
    download('natlang-research.json', JSON.stringify({ format: 1, workspace, state,
        history: await store.history('research'), child_runs, native_values,
        export_profile: { model: client.modelStatus?.id ?? '', seed: Number($('seed').value) } }, null, 2), 'application/json');
};
$('import').onclick = () => $('import-file').click();
$('import-file').onchange = async () => {
    try {
        const file = $('import-file').files?.[0]; if (!file) return;
        const bundle = JSON.parse(await file.text());
        const current = await research.runtime.workspace.head();
        const currentFiles = Object.keys(current?.files ?? {});
        if (currentFiles.some(path => !programFiles.includes(path) && !path.startsWith('evidence/reliability-')) || state.revision)
            throw new Error('Import into a fresh research workspace');
        if (bundle.format !== 1 || bundle.state?.head !== bundle.workspace?.head) throw new Error('Imported state and workspace disagree');
        await research.runtime.workspace.import(bundle.workspace, current?.id ?? '');
        state = bundle.state;
        await store.put('states', 'research', { state, revision: state.revision, imported: true });
        for (const [id, value] of Object.entries(bundle.native_values ?? {})) await store.put('native_values', id, value);
        for (const [id, run] of Object.entries(bundle.child_runs ?? {})) await store.put('child_runs', id, run);
        for (const record of bundle.history ?? []) {
            const { id, ...rest } = record;
            await store.transaction(['history'], tx => tx.objectStore('history').add({ ...rest, imported: true }));
        }
        await paint(state);
        status('Workspace imported.');
    }
    catch (error) { status(error, true); }
};
await bootstrap();
