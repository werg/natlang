import { BrowserDomRenderer, EventLoop, projectEntry } from '../dist/browser/natlang.js';

export function applicationSource(project) {
  const view = project.root;
  if (!/^ui\/view\.(ts|nl)$/.test(view)) return null;
  const reducer = ['ui/reduce.ts', 'ui/reduce.nl'].find(path => project.files[path]);
  return reducer ? { view, reducer, files: project.files } : null;
}

/**
 * A live projection of real application transitions. `ui/reduce` and `ui/view` are project entries
 * taking named inputs ({ state, event } and { state }). Scrubbing never dispatches events.
 */
export function createLivePreview({ runtime, runtimeNamespace, onBusy, onError, onRun }) {
  const $ = id => document.getElementById(id);
  let app = null, renderer = null, history = [], at = -1, generation = 0, locked = false;
  const status = text => { $('previewStatus').textContent = text; };
  const lock = value => {
    locked = value; onBusy(value);
    $('liveCanvas').inert = value || at < history.length - 1;
    $('restartPreview').disabled = value;
  };
  function show(index) {
    at = Math.max(0, Math.min(index, history.length - 1));
    const item = history[at];
    if (!item) return;
    renderer.render(item.view);
    $('liveCanvas').inert = locked || at < history.length - 1;
    $('liveTimeline').max = String(history.length - 1);
    $('liveTimeline').value = String(at);
    $('liveView').textContent = JSON.stringify(item.view, null, 2);
    $('liveState').textContent = JSON.stringify(item.state, null, 2);
    $('liveTrace').textContent = JSON.stringify({ event: item.event, invocations: item.traces ?? [] }, null, 2);
    $('livePosition').textContent = `Step ${item.revision} · ${item.event?.kind ?? 'render'}${at < history.length - 1 ? ' · recorded snapshot (controls paused)' : ' · live'} · ${history.length} snapshots`;
    $('liveHistory').hidden = false;
    $('liveLatest').hidden = at === history.length - 1;
  }
  async function close() {
    generation++;
    const former = app; app = null;
    renderer?.close(); renderer = null;
    history = []; at = -1;
    $('liveHistory').hidden = true;
    $('liveView').textContent = '';
    status('Run this example to bring the interface to life.');
    const placeholder = document.createElement('div'); placeholder.className = 'canvas-placeholder';
    placeholder.textContent = 'Your program makes the interface. Its controls send events back.';
    $('liveCanvas').replaceChildren(placeholder);
    await former?.close();
  }
  async function start(project, preserve = false) {
    const source = applicationSource(project);
    if (!source) return;
    if (locked) throw new Error('Wait for the current execution to finish.');
    const previousState = preserve && app ? app.state : null;
    const previousRevision = preserve && app ? app.revision : 0;
    const previousHistory = preserve ? history : [];
    lock(true);
    try {
      await close();
      const current = generation;
      history = previousHistory;
      renderer = new BrowserDomRenderer($('liveCanvas'), async event => {
        if (locked || at < history.length - 1 || !app) return;
        lock(true); status(`Handling ${event.kind}…`);
        try {
          await app.dispatch(event);
          if (current === generation) status('Live · click a control or change the source.');
        } catch (error) { if (current === generation) { status(error.message); onError(error); } }
        finally { if (current === generation) lock(false); }
      }, onError);
      const reduce = projectEntry(source.files, source.reducer, runtimeNamespace);
      const view = projectEntry(source.files, source.view, runtimeNamespace);
      const stepRuntime = runtime();
      let traces = [];
      app = new EventLoop({ initialState: previousState ?? structuredClone(project.inputs.state), initialRevision: previousRevision,
        reduce: (state, event) => reduce({ state, event }), view: state => view({ state }),
        step: async (fn, context) => {
          if (context.stage === 'reduce') traces = [];
          const startedAt = new Date().toISOString(), started = performance.now(), calls = [];
          let value, outcome = { kind: 'done', detail: '' };
          try { value = await stepRuntime.run(fn, { signal: context.signal, trace: trace => { calls.push(trace); traces.push(trace); } }); }
          catch (error) { outcome = { kind: 'failed', detail: error instanceof Error ? error.message : String(error) }; throw error; }
          finally {
            if (context.stage === 'view' && current === generation) await onRun({ schema: 'natlang.playground.run/2', id: crypto.randomUUID(),
              projectId: project.id, projectName: project.name, revision: project.revision,
              source: { root: source.view, files: structuredClone(source.files) }, inputs: { state: app?.state ?? null }, startedAt,
              durationMs: Math.round(performance.now() - started), outcome, value: value ?? null,
              trace: calls.find(trace => trace.parentCallId === null)?.events ?? [], invocations: calls });
          }
          return value;
        },
        onTransition: transition => {
          if (current !== generation) return;
          history.push({ ...transition, traces });
          if (history.length > 80) history.shift();
          show(history.length - 1);
        } });
      status(source.view.endsWith('.nl') ? 'Generating a view with your local model…' : 'Building the live interface…');
      await app.start();
      status('Live · click a control or change the source.');
    } catch (error) { status(`Cannot render: ${error.message}`); throw error; }
    finally { lock(false); }
  }
  $('liveTimeline').oninput = () => show(Number($('liveTimeline').value));
  $('liveLatest').onclick = () => show(history.length - 1);
  return { start, close, cancel: () => app?.cancel(),
    get active() { return Boolean(app); },
    stale: () => status('Source changed · Apply & run to update the interface.') };
}
