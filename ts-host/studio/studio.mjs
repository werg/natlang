import { loadBrowserModelCatalog } from '../dist/browser/natlang.js';
import { StudioModel, natlangApplication } from './shared/natlang-app.mjs';
import { apps, appById } from './apps/index.mjs';
import { loadProgram, fixtureTurn } from './shared/program.mjs';
import { runChild } from './shared/child-runner.mjs';
import { StudioStore } from './shared/store.mjs';
import { applyOperation, Companion } from './shared/host.mjs';
import { el, renderPanels, download } from './shared/render.mjs';
import { json, sameValue, valuePreview } from './shared/domain.mjs';
const $ = id => document.getElementById(id), query = new URLSearchParams(location.search);
let lastFailure = null, lastStep=null;
let mode = query.has('fixture') ? 'fixture' : 'preview', spec = null, app = null, currentEvent = null, busy = false, abort = null, operationState = null, opIndex = 0, drafts = {}, source = null, branch = crypto.randomUUID(), appSerial = Promise.resolve();
const store = await StudioStore.open();
const status = (message, error = false) => { $('status').textContent = String(message); document.querySelector('.statusbar').classList.toggle('error', error); };
const fail = error => status(error instanceof Error ? error.message : String(error), true);
const companion = new Companion({ progress: message => status(message) });
const host = { studio: { async apply(state, event, decision) {
            if (abort?.signal.aborted)
                throw new Error('Run cancelled');
            if (!spec || !operationState)
                throw new Error('No active application');
            if (!sameValue(state, operationState))
                throw new Error('Operation must continue from the latest acknowledged state');
            const id = `${event.id}-${++opIndex}`;
            status(`${spec.title}: ${decision.action} · operation ${opIndex}`);
            const before = structuredClone(state), identity = { id, app: spec.id, event, decision, branch, sourceVersion: spec.version };
            await store.effect({ ...identity, status: 'pending', before });
            const services = { service: (operation, payload) => companion.run(operation, payload, id, abort.signal), runSource, cell: runCell, readValue, inspectTrace, trial:async experiment=>(await runChild({experiment},{signal:abort.signal,onProgress:status})).value, signal: abort.signal };
            const result = await applyOperation(spec, state, decision, services);
            operationState = structuredClone(result.state); lastStep=result;
            await store.effect({ ...identity, status: result.ok ? 'complete' : 'failed', before, state: result.state, detail: result.detail });
            return result;
        } } };
const studioModel = new StudioModel();
async function runSource(files, root, inputs) {
    const result = await runChild({ request: { root, files: Object.fromEntries(files.map(row => [row.id, row.source])), inputs, seed: { mode: 'derived', root: Number($('seed').value) } } }, { model: studioModel.loaded ? studioModel.turn : null, signal: abort?.signal, onProgress: status });
    const id = crypto.randomUUID();
    await store.put('child_runs', id, { ...result, source: files, root, inputs, model: studioModel.id ?? mode, seed: Number($('seed').value) });
    return { value: result.value, trace_id: id, trace_count: result.trace.length };
}
async function inspectTrace(id, index) { const run = await store.get('child_runs', id); if (!run || !Number.isSafeInteger(index) || index < 0 || index >= run.trace.length)
    throw new Error('Unknown child run or trace cursor'); return json(run.trace[index]); }
async function evaluateCell(cell, deps) {
    if (cell.engine === 'sqlite') {
        const result = await companion.run('notebook.query', { source: cell.source, ...(Object.keys(deps).length ? { tables: deps } : {}) }, `${currentEvent.id}-${opIndex}-sql`, abort.signal);
        return result.rows;
    }
    if (cell.engine === 'typescript')
        return (await runChild({ cell, deps }, { signal: abort.signal, onProgress: status })).value;
    return (await runSource([{ id: 'cell.nl', source: cell.source }], 'cell.nl', { deps })).value;
}
async function readValue(id){const record=await store.get('native_values',id);if(!record)throw new Error('Cell result is missing from host storage; rerun its source cell');return record.value;}
async function runCell(cell,deps){
 const value=await evaluateCell(cell,deps),id=crypto.randomUUID();
 await store.put('native_values',id,{value,cell:cell.id,source:cell.source,engine:cell.engine,event:currentEvent.id});
 return {id,preview:json(valuePreview(value))};
}

const catalog = await loadBrowserModelCatalog();
for (const model of catalog.models)
    $('model').add(new Option(model.label, model.id));
$('model').value = catalog.defaultId;
const selectedModel = () => catalog.models.find(row => row.id === $('model').value);
$('context').value = selectedModel()?.contextTokens ?? 32768;
$('model').onchange = () => { $('context').value = selectedModel()?.contextTokens ?? 32768; };
const categories = ['Create', 'Explore', 'Organize', 'Play', 'Develop', 'Observe'];
function navigation(filter = '') { const nav = $('navigation'); nav.replaceChildren(); const overview = el('a', 'nav-link' + (!spec ? ' active' : '')); overview.href = '#home'; overview.append(el('span', 'nav-icon', '◫'), el('span', '', 'Overview')); nav.append(overview); for (const category of categories) {
    const members = apps.filter(app => app.category === category && (app.title + ' ' + app.subtitle + ' ' + app.id).toLowerCase().includes(filter.toLowerCase()));
    if (!members.length)
        continue;
    nav.append(el('div', 'nav-group', category));
    for (const item of members) {
        const link = el('a', 'nav-link' + (spec?.id === item.id ? ' active' : ''));
        link.href = '#' + item.id;
        link.append(el('span', 'nav-icon', item.icon), el('span', '', item.title));
        nav.append(link);
    }
} }
function home(category = 'All') { const root = $('home'); root.replaceChildren(); const hero = el('div', 'hero'), intro = el('div', 'intro'); intro.append(el('span', 'intro-kicker', 'A COLLECTION OF LIVING APPLICATIONS'), el('h1', '', 'A little language.\nA world of possibilities.'), el('p', '', 'Spaces to make, think, play, and discover. Choose a starting point. Let natlang help you find what comes next.')); const art = el('div', 'hero-art'); for (let i = 0; i < 3; i++)
    art.append(el('div', 'orbit')); art.append(el('span', 'orbit-core', 'n')); hero.append(intro, art); root.append(hero); const tabs = el('div', 'section-tabs'); for (const label of ['All', ...categories]) {
    const b = el('button', category === label ? 'selected' : '', label);
    b.onclick = () => home(label);
    tabs.append(b);
} root.append(tabs); const grid = el('div', 'app-grid'); for (const item of apps.filter(a => category === 'All' || a.category === category)) {
    const card = el('a', 'app-card');
    card.href = '#' + item.id;
    card.style.setProperty('--card-color', item.color);
    card.append(el('span', 'card-project', item.project), el('div', 'card-icon', item.icon), el('h3', '', item.title), el('p', '', item.subtitle), el('span', 'card-arrow', '↗'));
    grid.append(card);
} root.append(grid); const note = el('div', 'activity-note'); note.append(el('span', '', '✧')); const para = el('p'); para.append(el('strong', '', 'The same language, many ways to think. '), document.createTextNode('Every application has an inspectable natlang program, a recoverable state, and a trail of the work it does.')); note.append(para); root.append(note); }
function setBusy(value) { busy = value; document.querySelector('.statusbar').classList.toggle('busy', value); $('cancel').hidden = !value; $('command-form').querySelector('button').disabled = value; $('model-settings').disabled = value; $('refresh').disabled = value; $('history-button').disabled = value || !spec; }
function ctx() { return { app: spec.id, drafts, draft: (key, value) => { drafts[key] = value; store.put('drafts', spec.id, { ...drafts }).catch(fail); }, dispatch, error: fail, assetUrl: asset => companion.assetUrl(asset), downloadValue:async id=>download('cell-result.json',json(await readValue(id)),'application/json'), upload: async (file) => { const result = await companion.upload(file); await dispatch('import', { target: result.asset }); } }; }
function paint(state, view) { $('app-title').textContent = view.heading; $('app-summary').textContent = view.summary; renderPanels($('panels'), spec.panels(state), view, ctx()); $('suggestions').replaceChildren(...view.suggestions.map(text => { const b = el('button', '', text); b.onclick = () => { $('command').value = text; $('command').focus(); }; return b; })); $('revision').textContent = `${spec.project} · event ${app?.revision??0} · state ${state.revision}`; }
function snapshot(state, revision, extra = {}) { return { version: spec.version, branch, state, revision, at: new Date().toISOString(), model: studioModel.id ?? mode, seed: Number($('seed').value), source: source?.files, ...extra }; }
async function mount(record) {
    if (app)
        await app.close();
    app = null;
    const state = record?.state ?? spec.initial();
    drafts = await store.get('drafts', spec.id) ?? {};
    source = await loadProgram(spec);
    operationState = structuredClone(state);
    branch = record?.branch ?? crypto.randomUUID();
    if (record && record.version !== spec.version)
        throw new Error('Saved workspace has a different schema version. Export it before migrating.');
    if (mode === 'preview') {
        paint(state, { heading: spec.title, summary: spec.subtitle, focus: spec.panelIds, suggestions: [] });
        return;
    }
    paint(state,{heading:spec.title,summary:spec.subtitle,focus:spec.panelIds,suggestions:[]});
    app = natlangApplication({ source, services: host, seedRoot: Number($('seed').value), initialState: state, initialRevision: record?.revision ?? 0,
        model: mode === 'fixture' ? fixtureTurn(spec, () => app?.state ?? state, () => currentEvent) : studioModel.turn,
        onCommit: async (commit) => { if (!sameValue(commit.state, operationState))
            throw new Error('Final state must match acknowledged operations'); await store.commit(spec.id, snapshot(commit.state, commit.revision, { event: commit.event, trace: commit.trace, run: commit.invocations.find(call => call.parentCallId === null)?.callId })); },
        onTransition: transition => { paint(transition.state, transition.view); status(transition.state.notice); }, onFailure: failure => { lastFailure = failure; const detail = failure.error instanceof Error ? failure.error.message : String(failure.error); if(failure.stage==='view')paint(app.state,{heading:spec.title,summary:app.state.notice,focus:spec.panelIds,suggestions:[]}); fail(`${failure.stage}: ${detail}${failure.stage==='view'?' · State is available; refresh the view.':''}`); },
    });
    await app.start();
}
async function route() {
    if (busy) {
        status('Finish or stop the current run before switching spaces.');
        history.replaceState(null, '', '#' + (spec?.id ?? 'home'));
        return;
    }
    const id = location.hash.slice(1) || 'home';
    spec = appById.get(id) ?? null;
    setNavigation(false);
    navigation($('search').value);
    $('home').hidden = !!spec;
    $('application').hidden = !spec;
    $('history-button').disabled = !spec;
    $('crumb').textContent = spec?.title ?? 'Overview';
    if (!spec) {
        if (app)
            await app.close();
        app = null;
        home();
        return;
    }
    $('eyebrow').textContent = `${spec.category} / ${spec.project} / ${spec.subtitle}`;
    setBusy(true);
    try {
        await mount(await store.get('states', spec.id));
        status(mode === 'preview' ? 'Load an interpreter, or explore explicit controls in settings.' : `${spec.title} is ready.`);
    }
    finally {
        setBusy(false);
    }
}
async function dispatch(kind, data = {}, formId) {
    const submittedDrafts={...drafts};
    if (busy)
        throw new Error('An interaction is running. Your draft is preserved; wait or stop it first.');
    if (!spec)
        throw new Error('Select an application');
    return navigator.locks.request('natlang-studio-' + spec.id, { ifAvailable: true }, async (lock) => {
        if (!lock)
            throw new Error('This application is running in another tab. Your draft is preserved.');
        if (app) {
            const latest = await store.get('states', spec.id);
            if (latest && latest.revision !== app.revision) {
                await mount(latest);
                throw new Error('A newer revision arrived from another tab. Review it and submit your draft again.');
            }
        }
        return dispatchLocked(kind, data, formId, submittedDrafts);
    });
}
async function dispatchLocked(kind, data = {}, formId, submittedDrafts={}) {
    if (busy)
        throw new Error('An interaction is running. Your draft is preserved; wait or stop it first.');
    if (mode === 'preview') {
        $('settings').showModal();
        return false;
    }
    if (mode === 'fixture' && (kind === 'command' || (spec.id === 'notebook' && kind === 'run') || (spec.id === 'ide' && kind === 'evaluate') || (spec.id === 'tests' && kind === 'run') || (spec.id === 'experiments' && kind === 'compare')))
        throw new Error('This goal needs natlang orchestration. Load a model, or use an individual exact operation in the control fixture.');
    setBusy(true);
    abort = new AbortController();
    opIndex = 0;
    currentEvent = { id: crypto.randomUUID(), kind, value: kind === 'command' ? data.text : JSON.stringify(data) };
    operationState = app.state; lastStep=null;
    try {
        const result = await app.dispatch(currentEvent);
        if (formId && lastStep?.ok!==false) {
            for (const key of Object.keys(drafts))
                if (key.startsWith(formId + '/') && drafts[key]===submittedDrafts[key])
                    delete drafts[key];
            await store.put('drafts', spec.id, drafts);
            if (result)
                paint(result.state, result.view);
        }
        status(app.state.notice,lastStep?.ok===false);
        return lastStep?.ok!==false;
    }
    catch (error) {
        fail(error);
        throw error;
    }
    finally {
        setBusy(false);
        abort = null;
    }
}
async function enqueueRoute() { appSerial = appSerial.then(route, route).catch(fail); await appSerial; }
window.addEventListener('hashchange', enqueueRoute);
$('search').oninput = () => navigation($('search').value);
function setNavigation(open){document.body.classList.toggle('nav-open',open);$('menu').setAttribute('aria-expanded',String(open));$('studio-sidebar').inert=matchMedia('(max-width:800px)').matches&&!open;}
$('menu').onclick=()=>setNavigation(!document.body.classList.contains('nav-open'));
matchMedia('(max-width:800px)').addEventListener('change',()=>setNavigation(false));
document.addEventListener('keydown',event=>{if(event.key==='Escape')setNavigation(false);});
document.querySelector('.shell').addEventListener('click',event=>{if(!event.target.closest('#menu')&&document.body.classList.contains('nav-open'))setNavigation(false);});
setNavigation(false);
$('model-settings').onclick = () => $('settings').showModal();
$('fixture').onclick = async () => { mode = 'fixture'; $('mode').textContent = 'Controls · no model'; $('mode').className = 'badge fixture'; $('settings').close(); await enqueueRoute(); };
$('load-model').onclick = async () => { const model = selectedModel(); $('load-model').disabled = true; $('fixture').disabled = true; $('model-status').textContent = 'Loading model…'; setBusy(true); try {
    await studioModel.load(model, { contextTokens: Number($('context').value), gpuLayers: $('compute').value === 'gpu' ? 99999 : 0 });
    mode = 'model';
    $('mode').textContent = 'Local interpreter';
    $('mode').className = 'badge';
    $('settings').close();
    setBusy(false);
    await enqueueRoute();
}
catch (error) {
    $('model-status').textContent = String(error);
}
finally {
    $('load-model').disabled = false;
    $('fixture').disabled = false;
    setBusy(false);
} };
$('command-form').onsubmit = async (event) => { event.preventDefault(); const text = $('command').value.trim(); if (!text)
    return; try {
    const completed=await dispatch('command', { text });
    if(completed && $('command').value.trim()===text) $('command').value = '';
}
catch (error) {
    fail(error);
} };
$('cancel').onclick = () => { abort?.abort(); app?.cancel(); companion.cancel().catch(fail); status('Stopping; completed external effects remain in the operation journal.'); };
$('refresh').onclick = async () => { if (!app)
    return; setBusy(true); try {
    await app.refresh();
}
catch (error) {
    fail(error);
}
finally {
    setBusy(false);
} };

$('show-source').onclick=()=>{
 const files=Object.entries(source?.files??{});$('source-files').replaceChildren();
 for(const [name,body]of files){const button=el('button','',name.slice(spec.id.length+1));button.onclick=()=>{$('source-text').textContent=body;};$('source-files').append(button);}
 $('source-text').textContent=files[0]?.[1]??'';$('source-dialog').showModal();
};
$('export').onclick = async () => {
 try{const record=await store.get('states',spec.id)??snapshot(spec.initial(),0);const references=[];
 for(const ref of spec.references?.(record.state)??[])references.push({...ref,value:await store.get(ref.store,ref.id)});
 download(`${spec.id}-workspace.json`,json({schema:'natlang-studio-snapshot/1',app:spec.id,...record,references,effects:(await store.all('effects')).filter(r=>r.app===spec.id)}),'application/json');}catch(error){fail(error);}
};
$('history-button').onclick = async () => {
    const root = $('history-content');
    root.replaceChildren();
    const rows = await store.history(spec.id);
    const effects = (await store.all('effects')).filter(r => r.app === spec.id);
    if (!rows.length && !effects.length)
        root.append(el('p', '', 'Your first operation will start the journal.'));
    async function restore(state) { await navigator.locks.request('natlang-studio-' + spec.id, { ifAvailable: true }, async (lock) => { if (!lock)
        throw new Error('This application is running in another tab.'); const latest = await store.get('states', spec.id); const record = snapshot(state, Math.max(app?.revision ?? 0, latest?.revision ?? 0) + 1, { restored: true, branch: crypto.randomUUID() }); await store.commit(spec.id, record); $('history').close(); await mount(record); }); }
    for (const row of rows.toReversed()) {
        const detail = el('details', 'trace-row');
        detail.append(el('summary', '', `Revision ${row.revision} · ${row.event?.kind ?? 'restored'} · ${new Date(row.at).toLocaleTimeString()}`));
        const b = el('button', '', 'Restore state as a branch');
        b.onclick = () => restore(row.state).catch(fail);
        const exportTrace = el('button', '', 'Export trace');
        exportTrace.onclick = () => download('trace.json', json(row), 'application/json');
        detail.append(b, exportTrace, el('pre', '', json(row.state)));
        if (row.trace?.length) {
            const slider = el('input');
            slider.type = 'range';
            slider.min = '0';
            slider.max = String(row.trace.length - 1);
            slider.value = '0';
            slider.setAttribute('aria-label', 'Reduction trace cursor');
            const frame = el('pre', '', json(row.trace[0]));
            slider.oninput = () => { frame.textContent = json(row.trace[Number(slider.value)]); };
            detail.append(slider, frame);
        }
        root.append(detail);
    }
    for (const row of effects.toReversed()) {
        const detail = el('details', 'trace-row');
        detail.append(el('summary', '', `${row.decision.action} · ${row.status}`), el('pre', '', json({ decision: row.decision, detail: row.detail })));
        if (row.state) {
            const b = el('button', '', 'Continue from this operation');
            b.onclick = () => restore(row.state).catch(fail);
            detail.append(b);
        }
        root.append(detail);
    }
    $('history').showModal();
};
if (mode === 'fixture') {
    $('mode').textContent = 'Controls · no model';
    $('mode').className = 'badge fixture';
}
await companion.connect().catch(() => { });
await enqueueRoute();
window.natlangStudio = { model: studioModel, store, companion, get lastFailure() { return lastFailure; }, get app() { return app; }, get spec() { return spec; }, dispatch };
