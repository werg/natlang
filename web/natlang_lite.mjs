// Lightweight browser/JS embedding for the declared portable fixture subset.
// Native host access is explicit. This engine is trusted JavaScript, not a sandbox.
export const TRACE_VERSION = 'reduction-trace/1';

export async function deriveSeed(root, path, attempt, purpose, ordinal = 0) {
  const payload = { attempt, ordinal, path, purpose, root, version: 'sha256-json-v1' };
  const bytes = new TextEncoder().encode(JSON.stringify(payload));
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  let number = 0n;
  for (let i = 0; i < 8; i++) number = (number << 8n) | BigInt(hash[i]);
  return Number(number % (2n ** 31n));
}

export function checkedGraph(entries, root) {
  const snapshot = structuredClone(entries);
  if (!Object.hasOwn(snapshot, root)) throw new Error(`missing root ${root}`);
  const seen = new Set(), stack = new Set();
  function visit(name) {
    if (!Object.hasOwn(snapshot, name)) throw new Error(`missing definition ${name}`);
    if (stack.has(name)) throw new Error(`cyclic definition ${name}`);
    if (seen.has(name)) return;
    stack.add(name);
    for (const target of Object.values(snapshot[name].uses || {})) visit(target);
    if (snapshot[name].kind === 'map') visit(snapshot[name].fn);
    if (snapshot[name].kind === 'fold') visit(snapshot[name].step);
    stack.delete(name); seen.add(name);
  }
  for (const name of Object.keys(snapshot)) visit(name);
  return { root, definitions: snapshot };
}

function portable(value) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value) || (Number.isInteger(value) && !Number.isSafeInteger(value)))
      throw new Error('nonportable number');
    return value;
  }
  if (Array.isArray(value)) return value.map(portable);
  if (typeof value === 'object' && Object.prototype.toString.call(value) === '[object Object]') {
    const out = {};
    for (const [k, v] of Object.entries(value)) out[k] = portable(v);
    return out;
  }
  throw new Error('native value cannot cross typed boundary');
}

function fits(value, type) {
  if (type.endsWith('[]')) return Array.isArray(value) && value.every(item => fits(item, type.slice(0, -2)));
  if (type === 'Text') return typeof value === 'string';
  if (type === 'Bool') return typeof value === 'boolean';
  if (type === 'Num') return typeof value === 'number' && Number.isFinite(value);
  if (type === 'Null') return value === null;
  if (type.startsWith('{')) {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
    const fields = type.slice(1, -1).split(',').map(x => x.trim()).filter(Boolean);
    return fields.every(field => {
      const [name, child] = field.split(':').map(x => x.trim());
      return Object.hasOwn(value, name) && fits(value[name], child);
    });
  }
  throw new Error(`unsupported portable type ${type}`);
}

export function readTrace(events) {
  if (!Array.isArray(events) || events[0]?.kind !== 'manifest') throw new Error('missing trace manifest');
  events.forEach((e, i) => { if (e.version !== TRACE_VERSION || e.seq !== i) throw new Error('invalid trace sequence'); });
  const states = events.filter(e => e.kind === 'state');
  let reconstructed = structuredClone(states[0]?.value);
  for (const event of events.filter(e => e.kind === 'reduction')) {
    for (const change of event.changes) {
      if (!change.path.length) { reconstructed = structuredClone(change.after); continue; }
      let target = reconstructed;
      for (const key of change.path.slice(0, -1)) target = target[key];
      const key = change.path.at(-1);
      if (change.after_present) target[key] = structuredClone(change.after);
      else delete target[key];
    }
  }
  const canonical = value => JSON.stringify(value, (_, item) =>
    item && !Array.isArray(item) && typeof item === 'object'
      ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => a.localeCompare(b))) : item);
  if (canonical(reconstructed) !== canonical(states.at(-1)?.value))
    throw new Error('reduction changes do not reconstruct final state');
  return { manifest: events[0], final: states.at(-1)?.value,
           reconstructed,
           outcome: states.at(-1)?.outcome,
           actions: events.filter(e => e.kind === 'action'),
           coverage: { mode: 'recorded-observations-only', native_effects_replayable: false } };
}

export async function runGraph(graph, inputs, { decisions = {}, host = {}, runId = 'browser-run',
                                                   seed = null, maxEpisodes = 256, driver = null } = {}) {
  const events = [];
  const emit = (kind, data = {}) => events.push({ version: TRACE_VERSION, seq: events.length, kind, ...data });
  emit('manifest', { run_id: runId, semantic_version: 'portable-js/1', engine_bindings: ['browser-js'],
                     seed_policy: seed === null ? 'backend' : { mode: 'derived', root: seed },
                     capture: 'reduction-subset' });
  emit('state', { phase: 'initial', value: { root: graph.root, inputs: portable(inputs) } });
  let episodes = 0;
  async function invoke(name, args, path) {
    const fn = graph.definitions[name];
    if (!fn) throw new Error(`unknown function ${name}`);
    if (fn.kind === 'map') {
      const items = args[fn.over];
      if (!Array.isArray(items)) throw new Error('Map input must be a finite list');
      const result = [];
      for (let i = 0; i < items.length; i++) result.push(await invoke(fn.fn,
        { [fn.itemName || 'item']: items[i], ...fn.bindings }, `${path}/${i}`));
      if (!fits(result, fn.returns)) throw new Error(`type mismatch at ${path}`);
      emit('node', { path, transition: 'done', node_type: 'MapNode' });
      return result;
    }
    if (fn.kind === 'fold') {
      const items = args[fn.over];
      if (!Array.isArray(items)) throw new Error('Fold input must be a finite list');
      let accumulator = structuredClone(fn.init);
      for (let i = 0; i < items.length; i++) accumulator = await invoke(fn.step,
        { acc: accumulator, item: items[i] }, `${path}/step/${i}`);
      if (!fits(accumulator, fn.returns)) throw new Error(`type mismatch at ${path}`);
      emit('node', { path, transition: 'done', node_type: 'FoldNode' });
      return accumulator;
    }
    if (fn.kind === 'code' || Object.hasOwn(fn, 'code')) {
      if (fn.engine && fn.engine !== 'browser-js') throw new Error(`unsupported engine ${fn.engine}`);
      const code = fn.code;
      const value = new Function('args', 'host', `'use strict';\n${code}`)(structuredClone(args), host);
      const result = portable(value === undefined ? null : value);
      if (!fits(result, fn.returns)) throw new Error(`type mismatch at ${path}`);
      emit('eval', { path, phase: 'completed', engine: 'browser-js', value: result });
      emit('node', { path, transition: 'done', node_type: 'Lambda' });
      return result;
    }
    if (++episodes > maxEpisodes) throw new Error('episode budget exhausted');
    const callId = `${path || '$root'}@1`;
    emit('invocation', { phase: 'start', call_id: callId, path, attempt: 1, parent_path: null });
    const steps = driver ? await driver({ name, args: structuredClone(args), path,
                                          callId, seed: seed === null ? null :
                                            await deriveSeed(seed, path, 1, 'model-turn', 0) })
                         : (decisions[path] || decisions[name]);
    if (!Array.isArray(steps)) throw new Error(`missing recorded decisions at ${path}`);
    let result;
    for (const [index, action] of steps.entries()) {
      if (action.name === 'write') {
        if (action.arguments.path !== 'return') throw new Error('portable subset writes return only');
        result = portable(action.arguments.value);
        if (!fits(result, fn.returns)) throw new Error(`type mismatch at ${path}`);
      } else if (action.name === 'call') {
        const target = fn.uses?.[action.arguments.function];
        if (!target) throw new Error('callee is absent from lexical source graph');
        const childArgs = {};
        for (const [key, value] of Object.entries(action.arguments.inputs || {})) childArgs[key] =
          value.startsWith('args/') ? args[value.slice(5)] : value;
        result = await invoke(target, childArgs, `${path}/call/${index}`);
      } else if (action.name === 'report_blocker') {
        emit('action', { call_id: callId, surface: 'tools-v2', name: action.name,
                         arguments: action.arguments, outcome: 'blocked' });
        emit('node', { path, transition: 'quiesced', detail: action.arguments.missing });
        emit('invocation', { phase: 'end', call_id: callId });
        return { $blocked: action.arguments.missing };
      } else throw new Error(`unsupported action ${action.name}`);
      emit('action', { call_id: callId, surface: 'tools-v2', name: action.name,
                       arguments: action.arguments, outcome: 'ok' });
    }
    if (result === undefined) throw new Error(`return missing at ${path}`);
    emit('invocation', { phase: 'end', call_id: callId });
    emit('node', { path, transition: 'done', node_type: 'Lambda' });
    return result;
  }
  const value = await invoke(graph.root, inputs, '');
  const outcome = value?.$blocked ? 'quiesced' : 'done';
  emit('reduction', { phase: 'final', changes: [{ path: [], before_present: true,
    after_present: true, before: events[1].value, after: outcome === 'done' ? value : null,
    type: outcome === 'done' ? typeof value : 'Null' }] });
  emit('state', { phase: 'final', value: outcome === 'done' ? value : null, outcome });
  return { outcome, value: outcome === 'done' ? value : null, events };
}
