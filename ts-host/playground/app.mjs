import { BrowserNatlangClient, BROWSER_MODEL_CATALOG, loadBrowserModelCatalog, checkModelStorage,
  probeBrowserGpu, newPlaygroundProject, assertPlaygroundProject, editPlaygroundProject,
  validProjectPath, validatePlaygroundProject, loadFunctionFiles, runPlaygroundProject, traceFrame,
  admitPlaygroundRun } from '../dist/browser/natlang.js';
import { applicationSource, createLivePreview } from './live.mjs';
import { mountInputForm } from './input-form.mjs';
import { storage } from './storage.mjs';
import { examples, exampleCategories } from './examples.mjs';

const $ = id => document.getElementById(id);
const escapeHTML = value => String(value).replace(/[&<>"']/g, ch =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);
const pretty = value => JSON.stringify(value, null, 2) ?? 'null';
const now = () => new Date().toLocaleString();
const uid = () => crypto.randomUUID();
const download = (name, text, type = 'application/json') => {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement('a'); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
const parseJSON = (text, label, emptyValue) => {
  if (!text.trim()) return emptyValue;
  try { return JSON.parse(text); }
  catch (error) { throw new Error(`${label}: ${error.message}`); }
};
const sameJSON = (a, b) => JSON.stringify(a) === JSON.stringify(b);

let projects = [], project = null, selectedFile = null, runs = [], cases = [], selectedRun = null;
let diagnostics = [], model = null, modelSpec = null, abort = null, checkTimer = null;
const client = new BrowserNatlangClient();
let cursor = 0, importMode = 'project', busy = false, editingCase = null, caseFromRun = false, playTimer = null;
let jobToken = null, jobs = [], selectedJobId = null, jobCatalog = null, jobTimer = null;
let modelCatalog = { defaultId: BROWSER_MODEL_CATALOG[0].id, models: [...BROWSER_MODEL_CATALOG] };
let modelChoices = [...modelCatalog.models], modelSelectionExplicit = false;
window.natlangPlayground = { get project() { return project; }, get runs() { return runs; },
  get selectedRun() { return selectedRun; }, get model() { return model; } };

let autoTimer = null, pendingExecution = null, inputForm = null, rawInputsDirty = false;
const live = createLivePreview({ client,
  onBusy: value => {
    busy = value; $('stopButton').disabled = !value;
    for (const id of ['projectSelect', 'newProject', 'deleteProject', 'importProject', 'rootSelect', 'modelButton', 'loadModel']) $(id).disabled = value;
    renderDiagnostics();
  },
  onError: error => message(error.message, true),
  onRun: async record => {
    runs.unshift(record); await storage.put('runs', record);
    selectedRun = record; cursor = Math.max(0, record.trace.length - 1);
    renderRunList(); renderResult(); renderTrace();
  } });

function openLibrary() { renderExamples(); $('examplesDialog').showModal(); $('exampleSearch').focus(); }
async function openExample(template) {
  if (busy) { message('Stop the current run before opening an example.', true); return; }
  const next = newPlaygroundProject(template.name, template.root, template.files, template.inputs, template.expected);
  await storage.put('projects', next); projects.push(next);
  $('examplesDialog').close(); switchProject(next); selectPanel(applicationSource(next) ? 'preview' : 'result');
  message(`Opened ${template.name} · your editable copy is saved locally`);
  if (applicationSource(next) && !template.modelRequired) await runLive();
}
function renderExperience() {
  const template = examples.find(item => item.name === project.name && item.root === project.root);
  const appSource = applicationSource(project);
  const isApp = Boolean(appSource);
  const needsModel = project.root.endsWith('.nl') || Boolean(appSource?.reducer.endsWith('.nl'));
  document.querySelector('.workspace').classList.toggle('live-project', isApp);
  $('experimentTitle').textContent = project.name;
  $('experimentDescription').textContent = template?.description ?? 'Your own space to write, run, and explore a typed program.';
  $('learningText').textContent = template?.guide ?? (project.root.endsWith('.nl') ?
    'Read the instructions, change an input, and load a local model to try it. Expected values are reference answers; inspect what the model actually returns.' :
    'Change an input and run the program. Compare the result with your expectation, then open Trace to follow the execution.');
  $('executionMode').textContent = needsModel && !model?.loaded ? 'Model required' : '';
  $('executionMode').hidden = !needsModel || Boolean(model?.loaded);
  $('previewTab').hidden = !isApp;
  $('autoPreview').disabled = needsModel;
  if (needsModel) $('autoPreview').checked = false;
  $('autoPreview').title = needsModel ? 'Model-generated views run explicitly with Apply & run.' : 'Apply valid source edits automatically, keeping the current state.';
  $('runButton').innerHTML = needsModel && !model?.loaded ? 'Load model to run' :
    isApp ? '▶ Apply &amp; run <span class="shortcut">Ctrl ↵</span>' :
      '▶ Run <span class="shortcut">Ctrl ↵</span>';
}
function requireModel(callback) {
  pendingExecution = callback;
  $('modelProgress').textContent = 'Load a local model to run this program.';
  if (!$('modelDialog').open) $('modelDialog').showModal();
  void refreshModelDiagnostics();
}
async function runLive(preserve = false) {
  try {
    const inputs = currentInputs();
    if (!inputs?.state || typeof inputs.state !== 'object' || Array.isArray(inputs.state)) throw new Error('Live examples need an inputs.state object.');
    if (!sameJSON(project.inputs, inputs)) replaceProject(editPlaygroundProject(project, { inputs }));
    if (check().length) return;
    if (Object.keys(project.files).some(path => /^ui\/(view|reduce)\.nl$/.test(path)) && !model?.loaded) {
      requireModel(() => runLive(preserve)); return;
    }
    selectPanel('preview');
    await live.start(structuredClone(project), preserve);
    message('Interface running · try its controls');
  } catch (error) { message(error.message, true); }
}

function message(text, error = false) {
  $('runStatus').textContent = text;
  $('runStatus').style.color = error ? 'var(--bad)' : '';
}
function saveSoon() {
  $('saveStatus').textContent = 'Saving…';
  const snapshot = structuredClone(project);
  storage.put('projects', snapshot).then(() => {
    if (project.revision === snapshot.revision) $('saveStatus').textContent = 'Saved locally';
  }).catch(error => { $('saveStatus').textContent = `Save failed: ${error.message}`; });
}
function replaceProject(next) {
  const signatureChanged = next.root !== project.root || !sameJSON(next.files, project.files);
  project = next;
  projects = projects.filter(item => item.id !== next.id).concat(next)
    .sort((a, b) => a.name.localeCompare(b.name));
  saveSoon(); renderProjectSelect(); renderFileList(); renderProjectHeader(); renderRootSelect();
  if (signatureChanged) renderInputFields();
  scheduleCheck();
}
function renderInputFields() {
  try {
    const lambda = loadFunctionFiles(project.root, project.files);
    inputForm = mountInputForm($('inputFields'), lambda, project.inputs, () => {
      try {
        const inputs = inputForm.read();
        $('inputError').hidden = true;
        $('inputs').value = pretty(inputs); rawInputsDirty = false;
        if (!sameJSON(project.inputs, inputs)) replaceProject(editPlaygroundProject(project, { inputs }));
      } catch (error) { $('inputError').textContent = error.message; $('inputError').hidden = false; }
    });
    $('inputError').hidden = true;
  } catch {
    inputForm = null;
    $('inputFields').replaceChildren();
    $('rawInputs').open = true;
  }
}
function currentInputs() {
  try {
    const inputs = rawInputsDirty || !inputForm ? parseJSON($('inputs').value, 'Inputs', {}) : inputForm.read();
    inputForm?.validate(inputs);
    if (!inputs || typeof inputs !== 'object' || Array.isArray(inputs)) throw new Error('Inputs must be an object');
    $('inputs').value = pretty(inputs); rawInputsDirty = false;
    $('inputError').hidden = true;
    return inputs;
  } catch (error) {
    $('inputError').textContent = error.message; $('inputError').hidden = false;
    throw error;
  }
}
function renderRootSelect() {
  $('rootSelect').replaceChildren(...Object.keys(project.files).filter(path => path.endsWith('.nl') || path.endsWith('.ts'))
    .sort().map(path => { const option = new Option(path, path); option.selected = path === project.root; return option; }));
}
function renderProjectSelect() {
  $('projectSelect').replaceChildren(...projects.map(item => {
    const option = new Option(item.name, item.id); option.selected = item.id === project?.id; return option;
  }));
}
function renderProjectHeader() {
  $('projectName').textContent = project.name;
  $('revisionInfo').textContent = `rev ${project.revision.slice(0, 8)}`;
  renderExperience();
}
function renderFileList() {
  const filter = $('fileFilter').value.trim().toLowerCase();
  $('fileList').replaceChildren(...Object.keys(project.files).sort().filter(path =>
    path.toLowerCase().includes(filter)).map(path => {
    const button = document.createElement('button'); button.className = `file-item ${path === selectedFile ? 'active' : ''}`;
    button.type = 'button'; button.title = path;
    button.innerHTML = `<span class="file-kind ${path.endsWith('.nl') ? 'nl' : ''}">${path.endsWith('.nl') ? 'λ' : 'TS'}</span><span>${escapeHTML(path)}</span>`;
    button.onclick = () => selectFile(path);
    return button;
  }));
}
function renderEditor() {
  const source = project.files[selectedFile] ?? '';
  $('editor').value = source; $('fileName').textContent = selectedFile ?? 'No file';
  $('fileIcon').textContent = selectedFile?.endsWith('.nl') ? 'λ' : 'TS';
  $('languageInfo').textContent = selectedFile?.endsWith('.nl') ? 'natlang' : 'TypeScript';
  updateHighlight(); updateCursor();
}
function selectFile(path) { selectedFile = path; renderFileList(); renderEditor(); }

function highlightSource(text, isNl) {
  const token = /(\/\/[^\n]*|#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|\b(?:return|function|const|let|if|else|for|each|while|await|type|true|false|null|in)\b|\b(?:Text|Num|Bool|Null|Blob|Lambda|Map|Fold|Iterate|Dict)\b|\b\d+(?:\.\d+)?\b|^---.*$|^\s*[A-Za-z_]+:)/gm;
  let output = '', at = 0, match;
  while ((match = token.exec(text))) {
    output += escapeHTML(text.slice(at, match.index));
    const value = match[0], kind = /^(\/\/|#)/.test(value) ? 'comment' : /^['"`]/.test(value) ? 'string' :
      /^\d/.test(value) ? 'number' : /^(---|\s*\w+:)/.test(value) && isNl ? 'meta' :
        /^(Text|Num|Bool|Null|Blob|Lambda|Map|Fold|Iterate|Dict)$/.test(value) ? 'type' : 'keyword';
    output += `<span class="token ${kind}">${escapeHTML(value)}</span>`;
    at = match.index + value.length;
  }
  return output + escapeHTML(text.slice(at)) + '\n';
}
function updateHighlight() {
  const text = $('editor').value;
  $('highlight').innerHTML = highlightSource(text, selectedFile?.endsWith('.nl'));
  $('lineNumbers').textContent = Array.from({ length: text.split('\n').length }, (_, index) => index + 1).join('\n');
  $('highlight').scrollTop = $('editor').scrollTop;
  $('highlight').scrollLeft = $('editor').scrollLeft;
  $('lineNumbers').scrollTop = $('editor').scrollTop;
}
function updateCursor() {
  const start = $('editor').selectionStart, before = $('editor').value.slice(0, start);
  const lines = before.split('\n');
  $('cursorInfo').textContent = `Ln ${lines.length}, Col ${lines.at(-1).length + 1}`;
}

function renderDiagnostics() {
  $('diagnosticCount').textContent = diagnostics.length ? `${diagnostics.length} diagnostic${diagnostics.length === 1 ? '' : 's'}` : 'No diagnostics';
  $('diagnosticsList').replaceChildren(...diagnostics.slice(0, 8).map(item => {
    const line = document.createElement('div'); line.className = 'diagnostic';
    const button = document.createElement('button'); button.textContent = item.file;
    button.onclick = () => { const path = Object.keys(project.files).find(file => item.file.includes(file)); if (path) selectFile(path); };
    line.append(button, document.createTextNode(`  ${item.message}`)); return line;
  }));
  $('runButton').disabled = busy || diagnostics.some(item => item.severity === 'error');
}
function check() {
  try { diagnostics = validatePlaygroundProject(project); }
  catch (error) { diagnostics = [{ file: project.root, severity: 'error', message: error.message }]; }
  renderDiagnostics();
  if (diagnostics.length) message(`${diagnostics.length} source diagnostic${diagnostics.length === 1 ? '' : 's'}`, true);
  else if (!busy) message('Source checked · ready');
  return diagnostics;
}
function scheduleCheck() {
  clearTimeout(checkTimer);
  const revision = project.revision;
  checkTimer = setTimeout(() => { if (project.revision === revision) check(); }, 350);
}

function renderRunList() {
  const visible = runs.filter(run => run.projectId === project.id).slice(0, 80);
  $('runList').replaceChildren(...visible.map(run => {
    const button = document.createElement('button'); button.type = 'button';
    button.className = `run-item ${run.id === selectedRun?.id ? 'active' : ''}`;
    button.innerHTML = `<span class="run-indicator ${escapeHTML(run.outcome.kind)}"></span><span>${escapeHTML(new Date(run.startedAt).toLocaleTimeString())} · ${escapeHTML(run.outcome.kind)} · ${run.durationMs} ms</span>`;
    button.onclick = () => selectRun(run); return button;
  }));
  if (!visible.length) { const empty = document.createElement('p'); empty.className = 'run-history-empty';
    empty.textContent = 'Your runs will appear here.'; $('runList').append(empty); }
}
function selectRun(run) {
  if (playTimer) { clearInterval(playTimer); playTimer = null; $('tracePlay').textContent = '▶'; }
  selectedRun = run; cursor = Math.max(0, run.trace.length - 1);
  renderRunList(); renderResult(); renderTrace(); selectPanel('result');
}
function renderResult() {
  $('resultEmpty').hidden = Boolean(selectedRun); $('resultBody').hidden = !selectedRun;
  if (!selectedRun) return;
  const run = selectedRun, outcome = run.outcome.kind;
  $('outcomeBanner').className = `outcome-banner ${outcome === 'done' ? '' : outcome}`;
  $('outcomeBanner').textContent = `${outcome.toUpperCase()} · ${run.outcome.detail || 'Execution finished'}`;
  const labels = [`${run.durationMs} ms`, `${run.trace.length} events`, `rev ${run.revision.slice(0, 8)}`];
  if (run.correct !== undefined) labels.push(run.correct ? 'Expected value matched' : 'Expected value differed');
  if (run.revision !== project.revision) labels.push('Source has changed since this run');
  $('resultMeta').replaceChildren(...labels.map(label => { const span = document.createElement('span'); span.textContent = label; return span; }));
  $('resultValue').textContent = pretty(run.value); $('resultEmitted').textContent = pretty(run.emitted);
  const prior = $('compareRun').value;
  const comparable = runs.filter(item => item.projectId === run.projectId && item.id !== run.id);
  $('compareRun').replaceChildren(new Option('Choose another run', ''), ...comparable.map(item =>
    new Option(`${new Date(item.startedAt).toLocaleTimeString()} · ${item.model?.id ?? 'crisp'} · ${item.outcome.kind}`, item.id)));
  if (comparable.some(item => item.id === prior)) $('compareRun').value = prior;
  renderComparison();
}
function runMetrics(run) {
  const turns = run.model?.turns ?? [];
  const sum = field => turns.reduce((total, turn) => total + (turn[field] ?? 0), 0);
  return { model: run.model?.id ?? 'crisp only', outcome: run.outcome.kind,
    correct: run.correct ?? null, durationMs: run.durationMs, turns: turns.length,
    promptTokens: sum('promptTokens'), cachedTokens: sum('cachedTokens'),
    completionTokens: sum('completionTokens'), retries: sum('retries') };
}
function renderComparison() {
  const other = runs.find(item => item.id === $('compareRun').value);
  const box = $('compareSummary'); box.hidden = !selectedRun || !other;
  if (!selectedRun || !other) return;
  const sameRevision = selectedRun.revision === other.revision;
  const sameInputs = sameJSON(selectedRun.inputs, other.inputs);
  const sameOutput = selectedRun.outcome.kind === other.outcome.kind && sameJSON(selectedRun.value, other.value);
  box.textContent = pretty({ sameRevision, sameInputs, sameOutput,
    selected: runMetrics(selectedRun), compared: runMetrics(other) });
}
function summary(event) {
  if (event.kind === 'action') return `${event.name ?? 'action'} → ${event.outcome ?? ''}`;
  if (event.kind === 'invocation') return `${event.phase ?? ''} ${event.path || '$root'}`;
  if (event.kind === 'node') return `${event.transition ?? ''} ${event.path ?? ''}`;
  return `${event.phase ?? event.surface ?? ''}`;
}
function renderTrace() {
  const events = selectedRun?.trace ?? [];
  $('traceEmpty').hidden = events.length > 0; $('traceBody').hidden = events.length === 0;
  if (!events.length) return;
  cursor = Math.max(0, Math.min(cursor, events.length - 1));
  const frame = traceFrame(events, cursor);
  $('traceSlider').max = String(events.length - 1); $('traceSlider').value = String(cursor);
  $('tracePosition').textContent = `Event ${cursor + 1} / ${events.length}`;
  $('tracePrev').disabled = cursor === 0; $('traceNext').disabled = cursor === events.length - 1;
  $('traceDetail').textContent = pretty(frame.event);
  $('traceState').textContent = pretty(frame.state?.value ?? null);
  $('traceContext').textContent = pretty({ activeCalls: frame.activeCalls,
    actionsSoFar: frame.actions.length, effectsSoFar: frame.effects });
  const from = Math.max(0, cursor - 40), to = Math.min(events.length, cursor + 41);
  $('traceEvents').replaceChildren(...events.slice(from, to).map((event, offset) => {
    const index = from + offset, button = document.createElement('button');
    button.className = `trace-event ${index === cursor ? 'active' : ''}`;
    button.textContent = `${String(index).padStart(4, '0')}  ${String(event.kind).padEnd(15)} ${summary(event)}`;
    button.onclick = () => { cursor = index; renderTrace(); }; return button;
  }));
  $('traceEvents').querySelector('.active')?.scrollIntoView({ block: 'nearest' });
}
function selectPanel(name) {
  if (['cases', 'jobs'].includes(name)) setAdvanced(true);
  for (const panel of ['result', 'trace', 'cases', 'jobs', 'preview']) {
    $(`${panel}Panel`).hidden = panel !== name;
    const tab = document.querySelector(`[data-panel="${panel}"]`);
    tab.classList.toggle('selected', panel === name); tab.setAttribute('aria-selected', String(panel === name)); tab.tabIndex = panel === name ? 0 : -1;
  }
  if (name === 'cases') renderCases();
  if (name === 'jobs') void refreshJobs();
}

async function jobApi(path, method = 'GET', payload) {
  const response = await fetch(`/api/playground/${path}`, {
    method, headers: method === 'GET' ? {} : { 'Content-Type': 'application/json',
      'X-Natlang-Token': jobToken }, body: payload === undefined ? undefined : JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error ?? `Local job service returned ${response.status}`);
  }
  return response.headers.get('content-type')?.includes('application/json') ? response.json() : response.text();
}
function updateJobForm() {
  const kind = $('jobKind').value;
  $('jobDataset').parentElement.hidden = kind === 'gguf';
  $('jobCheckpoint').parentElement.hidden = !['train', 'gguf'].includes(kind);
  $('jobSteps').parentElement.hidden = kind !== 'train';
  $('jobSeed').parentElement.hidden = kind !== 'teacher';
  $('jobModelId').parentElement.hidden = kind !== 'teacher';
  $('jobServer').parentElement.hidden = !['teacher', 'export', 'evaluate'].includes(kind);
  $('jobModelLabel').parentElement.hidden = kind !== 'evaluate';
  $('jobQuant').parentElement.hidden = kind !== 'gguf';
}
async function refreshJobs() {
  if (!jobToken) return;
  try {
    jobs = await jobApi('jobs'); renderJobs();
    jobCatalog = await jobApi('catalog');
    modelCatalog = await loadBrowserModelCatalog();
    renderJobCatalog();
    if (selectedJobId) await renderJobLog();
  } catch (error) { $('jobsAvailability').textContent = `Job service error: ${error.message}`; }
}
function renderModelChoices(selectedId = modelCatalog.defaultId) {
  $('modelSelect').replaceChildren(...(modelChoices.length ?
    modelChoices.map((spec, index) => new Option(spec.label, String(index))) :
    [new Option('No checkpoint installed — choose a GGUF file', '')]));
  $('modelSelect').disabled = !modelChoices.length;
  if (!modelChoices.length) $('modelDialog').querySelector('.model-options').open = true;
  const index = modelChoices.findIndex(spec => spec.id === selectedId);
  $('modelSelect').selectedIndex = index >= 0 ? index : 0;
}
function renderJobCatalog() {
  if (!jobCatalog) return;
  const selectedModel = modelSelectionExplicit ? modelChoices[Number($('modelSelect').value)]?.id : modelCatalog.defaultId;
  const selectedData = $('jobDataset').value, selectedCheckpoint = $('jobCheckpoint').value;
  $('jobDataset').replaceChildren(...jobCatalog.datasets.map(item => new Option(
    `${item.path} · ${(item.bytes / 1_000_000).toFixed(1)} MB`, item.path)));
  $('jobCheckpoint').replaceChildren(...jobCatalog.checkpoints.map(item => new Option(item.name, item.merged)));
  if (selectedData) $('jobDataset').value = selectedData;
  if (selectedCheckpoint) $('jobCheckpoint').value = selectedCheckpoint;
  const known = new Set(modelCatalog.models.map(spec => spec.file));
  modelChoices = [...modelCatalog.models, ...jobCatalog.models.filter(item => !known.has(item.path.split('/').at(-1)))
    .map(item => ({ id: `local:${item.path}`, label: item.path.split('/').at(-1),
      trainingRun: 'local workbench', file: item.path.split('/').at(-1),
      templateUrl: null, quant: 'local', bytes: item.bytes, sha256: '',
      contextTokens: 4096, url: `/${item.path}` }))];
  renderModelChoices(selectedModel);
  $('evaluationList').replaceChildren(...(jobCatalog.evaluations ?? []).map(item => {
    const card = document.createElement('div'); card.className = 'case-card';
    const count = item.summary?.all?.n ?? 0, exact = item.summary?.all?.exact ?? 0;
    card.textContent = `${item.model ?? 'unspecified'} · ${exact}/${count} exact · manifest ${(item.manifest ?? '').slice(0, 10)} · ${item.path}`;
    return card;
  }));
  if (!jobCatalog.evaluations?.length) $('evaluationList').textContent = 'No completed evaluations yet.';
}
function renderJobs() {
  $('jobList').replaceChildren(...jobs.map(item => {
    const card = document.createElement('div'); card.className = 'case-card';
    card.innerHTML = `<header><span>${escapeHTML(item.name)}</span><span class="case-split">${escapeHTML(item.state)}</span></header><p>${escapeHTML(item.kind)} · ${escapeHTML(new Date(item.startedAt).toLocaleString())}</p>`;
    const inspect = document.createElement('button'); inspect.className = 'quiet'; inspect.textContent = 'Log';
    inspect.onclick = () => { selectedJobId = item.id; void renderJobLog(); };
    card.append(inspect);
    if (item.state === 'running') { const stop = document.createElement('button'); stop.className = 'danger';
      stop.textContent = 'Stop'; stop.onclick = async () => {
        try { await jobApi(`jobs/${item.id}/stop`, 'POST'); await refreshJobs(); }
        catch (error) { message(error.message, true); } }; card.append(stop); }
    return card;
  }));
  if (!jobs.length) $('jobList').textContent = 'No local workbench jobs yet.';
}
async function renderJobLog() {
  if (!selectedJobId) return;
  try { $('jobLog').textContent = await jobApi(`jobs/${selectedJobId}/log`); }
  catch (error) { $('jobLog').textContent = error.message; }
}
async function initJobs() {
  try {
    const session = await jobApi('session'); jobToken = session.token;
    $('jobsBody').hidden = false;
    $('jobsAvailability').textContent = 'Local pipeline service ready';
    updateJobForm(); await refreshJobs();
    jobTimer = setInterval(() => { if (jobs.some(item => item.state === 'running')) void refreshJobs(); }, 3000);
  } catch {
    $('jobsAvailability').textContent = 'Local pipeline jobs are available when this playground is served with npm run playground.';
  }
}
function renderCases() {
  const visible = cases.filter(item => item.projectId === project.id).sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  $('caseList').replaceChildren(...visible.map(item => {
    const card = document.createElement('div'); card.className = 'case-card';
    card.innerHTML = `<header><span>${escapeHTML(item.name)}</span><span class="case-split">${escapeHTML(item.split)} · ${escapeHTML(item.reviewStatus ?? 'draft')}</span></header><p>${escapeHTML(item.expected.kind)} · rev ${escapeHTML(item.revision.slice(0, 8))} · ${escapeHTML(new Date(item.createdAt).toLocaleDateString())}</p>`;
    const open = document.createElement('button'); open.className = 'quiet'; open.textContent = 'Open source';
    open.onclick = async () => { const next = newPlaygroundProject(`${item.name} (fork)`, item.source.root,
      item.source.files, item.inputs, item.expected.value);
      await storage.put('projects', next); projects.push(next); switchProject(next); };
    const edit = document.createElement('button'); edit.className = 'quiet'; edit.textContent = 'Edit';
    edit.onclick = () => openCaseDialog(item);
    const execute = document.createElement('button'); execute.className = 'quiet'; execute.textContent = 'Run case';
    execute.onclick = () => void runCase(item);
    const accept = document.createElement('button'); accept.className = 'quiet'; accept.textContent = 'Accept';
    accept.onclick = async () => { try { const admission = verifyCase(item);
      item.admission = admission; item.reviewStatus = 'accepted'; await storage.put('cases', item);
      renderCases(); message('Case passed exact trace admission and was accepted'); }
    catch (error) { message(`Cannot accept case: ${error.message}`, true); } };
    const reject = document.createElement('button'); reject.className = 'quiet'; reject.textContent = 'Reject';
    reject.onclick = async () => { item.reviewStatus = 'rejected'; await storage.put('cases', item);
      renderCases(); message('Case rejected'); };
    const remove = document.createElement('button'); remove.className = 'quiet'; remove.textContent = 'Delete';
    remove.onclick = async () => { if (!confirm(`Delete case “${item.name}”?`)) return;
      await storage.delete('cases', item.id); cases = cases.filter(found => found.id !== item.id); renderCases(); };
    card.append(open, execute, edit, accept, reject, remove); return card;
  }));
  if (!visible.length) { const p = document.createElement('p'); p.className = 'cases-intro';
    p.textContent = 'No saved cases yet. Run a program, then use “Save as case” to capture its source and trace.'; $('caseList').append(p); }
}

function renderExamples() {
  const term = $('exampleSearch').value.trim().toLowerCase();
  const category = $('exampleCategory').value;
  const visible = examples.filter(item => (category === 'All' || item.category === category) &&
    (!term || [item.name, item.description, item.level, ...item.concepts].join(' ').toLowerCase().includes(term)));
  $('exampleCount').textContent = `${visible.length} of ${examples.length} examples`;
  $('exampleList').replaceChildren(...visible.map(template => {
    const button = document.createElement('button'); button.type = 'button';
    button.className = 'example-card';
    button.innerHTML = `<strong>${escapeHTML(template.name)}</strong><span class="example-meta">${escapeHTML(template.category)} · ${escapeHTML(template.level)}</span><span>${escapeHTML(template.description)}</span><span class="example-concepts">${template.concepts.map(escapeHTML).join(' · ')}</span>`;
    button.onclick = () => void openExample(template).catch(error => message(error.message, true));
    return button;
  }));
  if (!visible.length) $('exampleList').textContent = 'No examples match these filters.';
}

function verifyCase(item) {
  if (!item.trace?.length) throw new Error('No recorded trace; run the scenario first');
  return admitPlaygroundRun({ schema: 'natlang.playground.run/1', trace: item.trace }, {
    outcome: item.expected.kind, value: item.expected.value,
    requiredActions: item.requiredActions ?? [], effects: item.effects ?? [],
    constrainedCalls: item.constrainedCalls ?? [],
  });
}

async function startJob() {
  if (!jobToken) { message('Local job service is unavailable', true); return; }
  const config = { kind: $('jobKind').value, name: $('jobName').value.trim(),
    dataset: $('jobDataset').value, checkpoint: $('jobCheckpoint').value,
    model: $('jobKind').value === 'train' ? $('jobCheckpoint').value : undefined,
    steps: Number($('jobSteps').value), rootSeed: Number($('jobSeed').value),
    modelId: $('jobModelId').value.trim(), modelLabel: $('jobModelLabel').value.trim(),
    server: $('jobServer').value.trim(), quant: $('jobQuant').value };
  try { const job = await jobApi('jobs', 'POST', config);
    selectedJobId = job.id; message(`${job.kind} job started`); await refreshJobs(); }
  catch (error) { message(`Cannot start job: ${error.message}`, true); }
}
async function sendCases() {
  if (!jobToken) { message('Start the local playground server to send cases to the pipeline', true); return; }
  const accepted = cases.filter(item => item.projectId === project.id && item.reviewStatus === 'accepted' && item.admission);
  if (!accepted.length) { message('Accept at least one exact-admitted case first', true); return; }
  try {
    for (const item of accepted) item.admission = verifyCase(item);
    const name = $('jobName').value.trim();
    const manifest = await jobApi('cases', 'POST', { name, cases: accepted });
    jobCatalog = await jobApi('catalog'); renderJobCatalog();
    $('jobKind').value = 'cases_ir'; $('jobDataset').value = manifest.path;
    updateJobForm(); selectPanel('jobs');
    message(`${manifest.count} accepted cases frozen in ${manifest.path}`);
  } catch (error) { message(`Cannot send cases: ${error.message}`, true); }
}

function promptText(title, help, value = '') {
  return new Promise(resolve => {
    const dialog = $('textDialog'); $('textDialogTitle').textContent = title;
    $('textDialogHelp').textContent = help; $('textDialogInput').value = value;
    $('textDialogError').textContent = '';
    const finish = () => { dialog.removeEventListener('close', finish); resolve(dialog.returnValue === 'confirm' ? $('textDialogInput').value.trim() : null); };
    dialog.addEventListener('close', finish); dialog.showModal(); $('textDialogInput').focus();
  });
}
function switchProject(next) {
  clearTimeout(autoTimer); void live.close();
  if (playTimer) { clearInterval(playTimer); playTimer = null; $('tracePlay').textContent = '▶'; }
  try { localStorage.setItem('natlang-project', next.id); } catch {}
  project = next; selectedFile = next.root;
  selectedRun = runs.find(run => run.projectId === next.id) ?? null;
  renderProjectSelect(); renderProjectHeader(); renderFileList(); renderRootSelect(); renderEditor();
  $('inputs').value = pretty(next.inputs); $('expected').value = next.expected === undefined ? '' : pretty(next.expected);
  rawInputsDirty = false; $('rawInputs').open = false; renderInputFields();
  renderRunList(); renderResult(); renderTrace(); check();
  selectPanel(applicationSource(next) ? 'preview' : 'result');
}

async function run() {
  if (busy) return;
  clearTimeout(autoTimer);
  let inputs;
  try { inputs = currentInputs(); }
  catch (error) { $('inputError').textContent = error.message; $('inputError').hidden = false; message(error.message, true); return; }
  const appSource = applicationSource(project);
  if ((project.root.endsWith('.nl') || appSource?.reducer.endsWith('.nl')) && !model?.loaded) {
    requireModel(() => run()); return;
  }
  if (appSource) return runLive();
  try {
    const expected = parseJSON($('expected').value, 'Expected value', undefined);
    if (!sameJSON(project.inputs, inputs) || !sameJSON(project.expected, expected))
      replaceProject(editPlaygroundProject(project, { inputs, expected }));
    if (check().length) return;
    const record = await executeProject(project);
    runs.unshift(record); await storage.put('runs', record); selectRun(record);
    message(`${record.outcome.kind} in ${record.durationMs} ms${record.correct === false ? ' · expected value differed' : ''}`,
      record.outcome.kind !== 'done' || record.correct === false);
  } catch (error) { message(error.message, true); }
  finally { busy = false; abort = null; $('stopButton').disabled = true; renderDiagnostics(); }
}

async function executeProject(snapshot) {
    if (snapshot.root.endsWith('.nl') && !model?.loaded) throw new Error('Load a local model to run natural instructions');
    busy = true; abort = new AbortController();
    $('runButton').disabled = true; $('stopButton').disabled = false;
    message(`Running revision ${snapshot.revision.slice(0, 8)}…`);
    return runPlaygroundProject(client, snapshot, { signal: abort.signal,
      runOptions: { seed: { mode: 'compatibility' } } });
}

async function runCase(item) {
  if (busy) return;
  try {
    const snapshot = { ...project, id: item.projectId, name: item.name, root: item.source.root,
      files: structuredClone(item.source.files), inputs: structuredClone(item.inputs),
      expected: structuredClone(item.expected.value), revision: item.revision };
    if (snapshot.root.endsWith('.nl') && !model?.loaded) { requireModel(() => runCase(item)); return; }
    const record = await executeProject(snapshot);
    runs.unshift(record); await storage.put('runs', record);
    Object.assign(item, { runId: record.id, trace: record.trace, observed: {
      outcome: record.outcome, value: record.value, emitted: record.emitted },
      model: record.model ?? null, reviewStatus: 'draft', admission: null });
    await storage.put('cases', item); selectRun(record); renderCases();
    message(`Case run recorded: ${record.outcome.kind} · review and accept if correct`, record.outcome.kind !== 'done');
  } catch (error) { message(`Case run failed: ${error.message}`, true); }
  finally { busy = false; abort = null; $('stopButton').disabled = true; renderDiagnostics(); }
}

async function refreshModelDiagnostics() {
  const spec = modelChoices[Number($('modelSelect').value)] ?? modelChoices[0];
  const file = $('modelFile').files[0];
  if (!spec && !file) {
    $('modelSummary').textContent = model?.loaded ? `${modelSpec?.label ?? 'Local model'} is loaded.` :
      'No checkpoint is installed in models/. Choose a local GGUF file below.';
    $('modelDiagnostics').textContent = '';
    return;
  }
  const choice = file ? { ...spec, label: file.name, bytes: file.size } : spec;
  const [gpu, disk] = await Promise.all([probeBrowserGpu(), checkModelStorage(choice)]);
  $('modelSummary').textContent = `${Math.round(choice.bytes / 1_000_000)} MB ${file ? 'local file' : 'download'} · ${gpu.usable ? 'GPU acceleration available' : 'CPU execution available'}. ${disk.available === null ? 'Storage availability could not be measured.' : `${Math.round(disk.available / 1_000_000)} MB of browser storage available.`}`;
  $('modelDiagnostics').textContent = pretty({ gpu, model: choice.label,
    downloadMB: file ? 0 : Math.round(choice.bytes / 1_000_000),
    availableMB: disk.available === null ? null : Math.round(disk.available / 1_000_000),
    recommendedFreeMB: Math.round(disk.recommendedFree / 1_000_000), loaded: model?.diagnostics ?? null });
}
async function loadModel() {
  const button = $('loadModel'); button.disabled = true;
  const spec = modelChoices[Number($('modelSelect').value)] ?? modelChoices[0];
  const file = $('modelFile').files[0], contextTokens = Number($('modelContext').value);
  const rawLayers = $('modelGpuLayers').value;
  const gpuLayers = rawLayers === '' ? undefined : Number(rawLayers);
  try {
    if (!file && !spec) throw new Error('Choose a GGUF file to load.');
    if (!Number.isInteger(contextTokens) || contextTokens < 512) throw new Error('Context must be at least 512 tokens');
    if (gpuLayers !== undefined && (!Number.isInteger(gpuLayers) || gpuLayers < 0)) throw new Error('GPU layers must be nonnegative');
    const options = { contextTokens,
      ...(gpuLayers === undefined ? {} : { gpuLayers }),
      onProgress: ({ loaded, total }) => { $('modelProgress').textContent = total ?
        `Downloading ${Math.round(100 * loaded / total)}%` : 'Loading model…'; } };
    const status = await client.loadModel(file ? { kind: 'files', files: [file], id: `file:${file.name}`,
      templateUrl: spec?.templateUrl } : { kind: 'url', url: spec.url, id: spec.id,
      templateUrl: spec.templateUrl }, options);
    model = client.model;
    modelSpec = file ? { id: `file:${file.name}`, label: file.name } : spec;
    $('modelButton').classList.add('loaded'); $('modelLabel').textContent = modelSpec.label;
    $('modelProgress').textContent = `Ready · ${model.diagnostics.gpuSelectionReason}` +
      (status.gpuFallbackReason ? ` (GPU load failed: ${status.gpuFallbackReason})` : '');
    $('runtimeStatus').textContent = model.diagnostics.gpuSelectionReason;
    await refreshModelDiagnostics();
    renderExperience();
    const resume = pendingExecution; pendingExecution = null;
    if ($('modelDialog').open) $('modelDialog').close();
    if (resume) queueMicrotask(() => void resume());
  } catch (error) { model = client.model;
    if (!model?.loaded) { modelSpec = null; $('modelButton').classList.remove('loaded');
      $('modelLabel').textContent = 'Set up a model'; }
    $('modelProgress').textContent = `Load failed: ${error.message}`; }
  finally { button.disabled = false; }
}

async function captureCase() {
  if (!selectedRun) return;
  openCaseDialog(null, selectedRun);
}
function openCaseDialog(item = null, run = null) {
  editingCase = item;
  caseFromRun = Boolean(run);
  $('caseDialogTitle').textContent = item ? 'Edit case' : run ? 'Save run as case' : 'New scenario';
  $('caseName').value = item?.name ?? (run ? `${run.projectName} · ${new Date(run.startedAt).toLocaleDateString()}` : `${project.name} scenario`);
  $('caseOutcome').value = item?.expected.kind ?? (['done', 'error', 'blocked'].includes(run?.outcome.kind) ? run.outcome.kind : 'done');
  $('caseExpected').value = item ? pretty(item.expected.value) : run ?
    pretty(run.expected === undefined ? run.value : run.expected) : project.expected === undefined ? '' : pretty(project.expected);
  $('caseActions').value = pretty(item?.requiredActions ?? []);
  $('caseEffects').value = pretty(item?.effects ?? []);
  $('caseCalls').value = pretty(item?.constrainedCalls ?? []);
  $('caseSplit').value = item?.split ?? 'train';
  $('caseNotes').value = item?.notes ?? '';
  $('caseError').textContent = '';
  $('caseDialog').showModal();
}
function saveCase(event) {
  event.preventDefault();
  try {
    const expected = parseJSON($('caseExpected').value, 'Expected value', null);
    const requiredActions = parseJSON($('caseActions').value, 'Required actions', []);
    const effects = parseJSON($('caseEffects').value, 'Effects', []);
    const constrainedCalls = parseJSON($('caseCalls').value, 'Constrained calls', []);
    if (![requiredActions, effects, constrainedCalls].every(Array.isArray))
      throw new Error('Actions, effects, and constrained calls must be JSON arrays');
    const run = editingCase || !caseFromRun ? null : selectedRun;
    const item = editingCase ? { ...editingCase } : { schema: 'natlang.playground.case/1', id: uid(),
      projectId: run?.projectId ?? project.id, groupId: run?.projectId ?? project.id,
      revision: run?.revision ?? project.revision, runId: run?.id ?? null,
      createdAt: new Date().toISOString(), source: structuredClone(run?.source ?? {
        root: project.root, files: project.files }), inputs: structuredClone(run?.inputs ?? project.inputs),
      observed: run ? { outcome: run.outcome, value: run.value, emitted: run.emitted } : null,
      model: run?.model ?? null, trace: run?.trace ?? [] };
    Object.assign(item, { name: $('caseName').value.trim(), split: $('caseSplit').value,
      expected: { kind: $('caseOutcome').value, value: expected },
      requiredActions, effects, constrainedCalls, notes: $('caseNotes').value,
      reviewStatus: 'draft', admission: null });
    storage.put('cases', item).then(() => { cases = cases.filter(found => found.id !== item.id).concat(item);
      renderCases(); message('Case saved as draft'); })
      .catch(error => message(`Case save failed: ${error.message}`, true));
    $('caseDialog').close('confirm');
  } catch (error) { $('caseError').textContent = error.message; }
}

async function readImport(file) {
  const text = await file.text();
  if (importMode === 'project') {
    const candidate = JSON.parse(text); assertPlaygroundProject(candidate);
    const imported = { ...candidate, id: uid(), revision: uid(), updatedAt: new Date().toISOString() };
    await storage.put('projects', imported); projects.push(imported); switchProject(imported);
    message(`Imported ${imported.name}`);
  } else {
    const records = text.split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
    for (const record of records) {
      if (record.schema !== 'natlang.playground.case/1' || !record.source?.files || !record.expected)
        throw new Error('Unsupported case JSONL record');
      record.id = uid(); await storage.put('cases', record); cases.push(record);
    }
    renderCases(); message(`Imported ${records.length} case${records.length === 1 ? '' : 's'}`);
  }
}

function setAdvanced(open) {
  $('advancedButton').setAttribute('aria-expanded', String(open));
  for (const name of ['cases', 'jobs']) document.querySelector(`[data-panel="${name}"]`).hidden = !open;
}
function bind() {
  $('libraryNav').onclick = openLibrary;
  const setExplorer = open => {
    $('projectExplorer').hidden = !open;
    $('explorerRail').hidden = open;
    document.querySelector('.workspace').classList.toggle('explorer-open', open);
    $('toggleExplorer').setAttribute('aria-expanded', String(open));
  };
  $('toggleExplorer').onclick = () => setExplorer(false);
  $('reopenExplorer').onclick = () => setExplorer(true);
  $('advancedButton').onclick = () => {
    const open = $('advancedButton').getAttribute('aria-expanded') !== 'true'; setAdvanced(open);
    if (!open && (!$('casesPanel').hidden || !$('jobsPanel').hidden)) selectPanel('result');
  };
  $('editLiveSource').onclick = () => { $('editor').focus(); $('editor').scrollIntoView({ block: 'center' }); };
  $('inspectTrace').onclick = () => selectPanel('trace');
  $('restartPreview').onclick = () => void runLive();
  document.querySelector('.inspector-tabs').addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    const tabs = [...document.querySelectorAll('[data-panel]')].filter(tab => !tab.hidden);
    const index = tabs.indexOf(document.activeElement);
    if (index < 0) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    selectPanel(tabs[next].dataset.panel); tabs[next].focus();
  });
  for (const tab of document.querySelectorAll('[data-panel]')) {
    tab.id ||= `${tab.dataset.panel}Tab`;
    tab.setAttribute('aria-controls', `${tab.dataset.panel}Panel`);
    const panel = $(`${tab.dataset.panel}Panel`); panel.setAttribute('role', 'tabpanel'); panel.setAttribute('aria-labelledby', tab.id);
  }
  $('projectSelect').onchange = () => { const next = projects.find(item => item.id === $('projectSelect').value); if (next) switchProject(next); };
  $('newProject').onclick = async () => {
    const name = await promptText('New project', 'Create a browser-local natlang workspace.', 'Untitled project');
    if (!name) return;
    const template = examples[0]; const next = newPlaygroundProject(name, template.root, template.files, template.inputs, template.expected);
    await storage.put('projects', next); projects.push(next); switchProject(next);
  };
  $('renameProject').onclick = async () => { const name = await promptText('Rename project', 'Give this workspace a clear name.', project.name);
    if (name) { replaceProject(editPlaygroundProject(project, { name })); renderProjectSelect(); } };
  $('deleteProject').onclick = async () => {
    if (!confirm(`Delete project “${project.name}”, its runs, and its cases from this browser? Export anything you need first.`)) return;
    const former = project; await storage.delete('projects', former.id);
    for (const item of runs.filter(item => item.projectId === former.id)) await storage.delete('runs', item.id);
    for (const item of cases.filter(item => item.projectId === former.id)) await storage.delete('cases', item.id);
    runs = runs.filter(item => item.projectId !== former.id);
    cases = cases.filter(item => item.projectId !== former.id);
    projects = projects.filter(item => item.id !== former.id);
    if (!projects.length) { const template = examples[0]; const next = newPlaygroundProject(template.name, template.root, template.files, template.inputs, template.expected);
      await storage.put('projects', next); projects.push(next); }
    switchProject(projects[0]);
  };
  $('fileFilter').oninput = renderFileList;
  $('rootSelect').onchange = () => { replaceProject(editPlaygroundProject(project, { root: $('rootSelect').value }));
    selectFile(project.root); message(`Entry function set to ${project.root}`); };
  $('addFile').onclick = async () => { const path = await promptText('Add file', 'Use a relative .nl or .ts path, such as tasks/answer.nl.', 'tasks/new.nl');
    if (!path) return;
    if (!validProjectPath(path) || project.files[path]) { message('Choose a unique relative .nl or .ts path', true); return; }
    const source = path.endsWith('.nl') ? '---\nreturns: Text\n---\nWrite a short answer to return.\n' :
      '/*---\nreturns: Text\nengine: typescript-host\n---*/\nreturn "Hello";\n';
    replaceProject(editPlaygroundProject(project, { files: { ...project.files, [path]: source } })); selectFile(path);
  };
  $('renameFile').onclick = async () => { const path = await promptText('Rename file', 'The root path follows the renamed file.', selectedFile);
    if (!path || path === selectedFile) return;
    if (!validProjectPath(path) || project.files[path]) { message('Choose a unique relative .nl or .ts path', true); return; }
    const files = { ...project.files, [path]: project.files[selectedFile] }; delete files[selectedFile];
    replaceProject(editPlaygroundProject(project, { files, root: project.root === selectedFile ? path : project.root })); selectFile(path);
  };
  $('deleteFile').onclick = () => {
    if (selectedFile === project.root) { message('Choose another root before deleting this file', true); return; }
    if (!confirm(`Delete ${selectedFile}?`)) return;
    const files = { ...project.files }; delete files[selectedFile];
    replaceProject(editPlaygroundProject(project, { files })); selectFile(project.root);
  };
  $('editor').oninput = () => {
    replaceProject(editPlaygroundProject(project, { files: { ...project.files, [selectedFile]: $('editor').value } }));
    updateHighlight(); updateCursor();
    if (applicationSource(project)) {
      live.stale(); clearTimeout(autoTimer);
      if ($('autoPreview').checked && !$('autoPreview').disabled) autoTimer = setTimeout(() => { if (!busy) void runLive(true); }, 700);
    }
  };
  $('editor').onscroll = updateHighlight;
  for (const event of ['click', 'keyup', 'select']) $('editor').addEventListener(event, updateCursor);
  $('editor').onkeydown = event => {
    if (event.key !== 'Tab') return; event.preventDefault();
    const editor = $('editor'), start = editor.selectionStart, end = editor.selectionEnd;
    editor.setRangeText('  ', start, end, 'end'); editor.dispatchEvent(new Event('input'));
  };
  $('inputs').oninput = () => { rawInputsDirty = true; };
  $('inputs').onchange = () => { try {
    const inputs = parseJSON($('inputs').value, 'Inputs', {});
    inputForm?.validate(inputs);
    if (!sameJSON(project.inputs, inputs)) replaceProject(editPlaygroundProject(project, { inputs }));
    rawInputsDirty = false; renderInputFields(); $('inputError').hidden = true;
  } catch (error) { $('inputError').textContent = error.message; $('inputError').hidden = false; message(error.message, true); } };
  $('expected').onchange = () => { try { const expected = parseJSON($('expected').value, 'Expected value', undefined);
    if (!sameJSON(project.expected, expected)) replaceProject(editPlaygroundProject(project, { expected }));
  } catch (error) { message(error.message, true); } };
  $('checkButton').onclick = check; $('runButton').onclick = run;
  $('stopButton').onclick = () => { abort?.abort(); live.cancel(); message('Stopping…'); };
  document.addEventListener('keydown', event => {
    if (document.querySelector('dialog[open]')) return;
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); void run(); }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault();
      storage.put('projects', project).then(() => $('saveStatus').textContent = 'Saved locally'); }
  });
  $('clearRuns').onclick = async () => { if (!confirm('Delete this project’s local run history?')) return;
    for (const item of runs.filter(item => item.projectId === project.id)) await storage.delete('runs', item.id);
    runs = runs.filter(item => item.projectId !== project.id); selectedRun = null;
    renderRunList(); renderResult(); renderTrace(); };
  for (const tab of document.querySelectorAll('[data-panel]')) tab.onclick = () => selectPanel(tab.dataset.panel);
  $('tracePrev').onclick = () => { cursor--; renderTrace(); };
  $('traceNext').onclick = () => { cursor++; renderTrace(); };
  $('traceSlider').oninput = () => { cursor = Number($('traceSlider').value); renderTrace(); };
  $('tracePlay').onclick = () => {
    if (playTimer) { clearInterval(playTimer); playTimer = null; $('tracePlay').textContent = '▶'; return; }
    if (!selectedRun?.trace.length) return;
    if (cursor >= selectedRun.trace.length - 1) cursor = 0;
    $('tracePlay').textContent = 'Ⅱ';
    playTimer = setInterval(() => { cursor++; renderTrace();
      if (cursor >= selectedRun.trace.length - 1) { clearInterval(playTimer); playTimer = null; $('tracePlay').textContent = '▶'; }
    }, 300);
  };
  $('downloadRun').onclick = () => selectedRun && download(`natlang-run-${selectedRun.id}.json`, pretty(selectedRun));
  $('compareRun').onchange = renderComparison;
  $('forkRun').onclick = async () => { if (!selectedRun) return;
    const run = selectedRun, next = newPlaygroundProject(`${run.projectName} (fork)`, run.source.root,
      run.source.files, run.inputs, run.expected);
    await storage.put('projects', next); projects.push(next); switchProject(next); message('Forked source from recorded revision'); };
  $('captureCase').onclick = captureCase;
  $('newCase').onclick = () => openCaseDialog();
  $('sendCases').onclick = sendCases;
  $('jobKind').onchange = updateJobForm;
  $('startJob').onclick = startJob;
  $('refreshJobs').onclick = async () => { try { jobCatalog = await jobApi('catalog'); renderJobCatalog(); await refreshJobs(); }
    catch (error) { message(error.message, true); } };
  $('caseDialog').querySelector('form').onsubmit = saveCase;
  $('exportCases').onclick = () => {
    const records = cases.filter(item => item.projectId === project.id);
    if (!records.length) { message('No cases to export', true); return; }
    download(`${project.name.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-cases.jsonl`,
      records.map(item => JSON.stringify(item)).join('\n') + '\n', 'application/x-ndjson');
  };
  $('importProject').onclick = () => { importMode = 'project'; $('importInput').click(); };
  $('importCases').onclick = () => { importMode = 'cases'; $('importInput').click(); };
  $('importInput').onchange = async () => { const file = $('importInput').files[0]; if (!file) return;
    try { await readImport(file); } catch (error) { message(`Import failed: ${error.message}`, true); }
    $('importInput').value = ''; };
  $('exportProject').onclick = () => download(`${project.name.toLowerCase().replace(/[^a-z0-9]+/g, '-')}.json`, pretty(project));
  $('modelButton').onclick = async () => { pendingExecution = null; $('modelDialog').showModal(); await refreshModelDiagnostics(); };
  $('modelDialog').addEventListener('close', () => { pendingExecution = null; });
  $('modelSelect').onchange = () => { modelSelectionExplicit = true; void refreshModelDiagnostics(); };
  $('modelFile').onchange = () => void refreshModelDiagnostics();
  $('loadModel').onclick = loadModel;
  $('closeExamples').onclick = () => $('examplesDialog').close();
  $('exampleSearch').oninput = renderExamples;
  $('exampleCategory').onchange = renderExamples;
  $('exampleCategory').replaceChildren(...exampleCategories.map(category => new Option(
    `${category} (${category === 'All' ? examples.length : examples.filter(item => item.category === category).length})`, category)));
  renderExamples();
}

async function start() {
  try {
    [projects, runs, cases] = await Promise.all(['projects', 'runs', 'cases'].map(name => storage.list(name)));
    projects = projects.filter(item => { try { assertPlaygroundProject(item); return true; } catch { return false; } });
    runs.sort((a, b) => b.startedAt.localeCompare(a.startedAt));
    if (!projects.length) { const template = examples[0]; const starter = newPlaygroundProject(template.name,
      template.root, template.files, template.inputs, template.expected);
      await storage.put('projects', starter); projects.push(starter); }
    try { modelCatalog = await loadBrowserModelCatalog(); modelChoices = [...modelCatalog.models]; }
    catch (error) { message(`Model catalog: ${error.message}`, true); }
    renderModelChoices();
    bind();
    let savedProject;
    try { savedProject = localStorage.getItem('natlang-project'); } catch {}
    switchProject(projects.find(item => item.id === savedProject) ?? projects[0]);
    const gpu = await probeBrowserGpu();
    $('runtimeStatus').textContent = gpu.usable ? 'WebGPU ready' : `CPU available · ${gpu.reason}`;
    void initJobs();
  } catch (error) {
    $('runtimeStatus').textContent = 'Startup failed'; message(`Cannot start playground: ${error.message}`, true);
  }
}
void start();
