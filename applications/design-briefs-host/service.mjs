/** Exact invocation custody. Domain lifecycles and approval decisions live in each native program. */
import { createHash } from 'node:crypto';
import { ApplicationError, describeApplication, validateArguments, validateResult, validateCheckpoint, jsonCopy } from './runtime.mjs';

const error = code => { throw new ApplicationError(code); };
const canonical = value => {
  if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
  if (value && typeof value === 'object') return '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + canonical(value[key])).join(',') + '}';
  return JSON.stringify(value);
};
const hash = value => createHash('sha256').update(canonical(value)).digest('hex');
function atPath(value, path, optional = false) {
  for (const key of path) {
    if (!value || typeof value !== 'object' || !Object.hasOwn(value, key)) {
      if (optional) return { present: false };
      error('declared_result_path_missing');
    }
    value = value[key];
  }
  return optional ? { present: true, value } : value;
}
function putPath(value, path, state) {
  let cursor = value;
  for (const key of path.slice(0, -1)) {
    if (!Object.hasOwn(cursor, key) || cursor[key] === null) cursor[key] = {};
    if (typeof cursor[key] !== 'object' || Array.isArray(cursor[key])) error('invalid_arguments');
    cursor = cursor[key];
  }
  cursor[path.at(-1)] = state;
}
function freeze(value) {
  if (value && typeof value === 'object') {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}

export function createApplicationService({ catalog, store, execute, capabilities, effectAdapters = {} }) {
  if (!(catalog instanceof Map) || !store || typeof execute !== 'function') error('invalid_configuration');
  const inflight = new Map();
  const canceled = () => new ApplicationError('invocation_cancelled');
  function join(running, signal) {
    if (signal?.aborted) return Promise.reject(canceled());
    return new Promise((resolve, reject) => {
      const waiter = { resolve, reject, signal, onAbort: null };
      const cleanup = () => signal?.removeEventListener('abort', waiter.onAbort);
      waiter.onAbort = () => {
        if (!running.waiters.delete(waiter)) return;
        cleanup(); reject(canceled());
        if (running.waiters.size === 0) {
          running.controller.abort();
          if (inflight.get(running.key) === running) inflight.delete(running.key);
        }
      };
      running.waiters.add(waiter);
      signal?.addEventListener('abort', waiter.onAbort, { once: true });
      // Close the small race between the initial check and listener registration.
      if (signal?.aborted) waiter.onAbort();
    });
  }
  function finish(running, outcome, value) {
    if (inflight.get(running.key) === running) inflight.delete(running.key);
    for (const waiter of running.waiters) {
      running.waiters.delete(waiter);
      waiter.signal?.removeEventListener('abort', waiter.onAbort);
      if (outcome === 'fulfilled') waiter.resolve(value);
      else waiter.reject(value);
    }
  }
  function application(id) { const app = catalog.get(id); if (!app) error('application_not_found'); return app; }
  function ifaceFor(app, name) { const iface = app.interfaces.get(name); if (!iface) error('interface_not_found'); return iface; }
  function authorize(p, appId, operation) {
    if (!p || typeof p.id !== 'string' || typeof p.tenantId !== 'string' || !Array.isArray(p.roles)) error('unauthorized');
    const rule = capabilities?.[appId];
    if (!Array.isArray(rule?.[operation]) || !p.roles.some(role => rule[operation].includes(role)) ||
        (rule.scopes?.[operation] ?? []).some(scope => !p.scopes?.includes(scope))) error('forbidden');
  }
  const keyFor = (p, appId, caseId) => ({ tenantId: p.tenantId, appId, caseId });
  function initialSnapshot(iface, inputs) {
    const value = jsonCopy(inputs);
    validateArguments(iface, value);
    return { initial: { interface: iface.name, inputs: value } };
  }
  return {
    list(p) {
      return [...catalog.values()].filter(app => { try { authorize(p, app.id, 'read'); return true; } catch { return false; } }).map(describeApplication);
    },
    describe(p, appId) { authorize(p, appId, 'read'); return describeApplication(application(appId)); },
    create(p, appId, caseId, { interface: name, inputs, idempotencyKey }) {
      const iface = ifaceFor(application(appId), name); authorize(p, appId, 'write');
      return store.createCase(p, keyFor(p, appId, caseId), initialSnapshot(iface, inputs), { idempotencyKey });
    },
    replace(p, appId, caseId, { interface: name, inputs, expectedVersion, idempotencyKey }) {
      const iface = ifaceFor(application(appId), name); authorize(p, appId, 'write');
      return store.replaceCase(p, keyFor(p, appId, caseId), expectedVersion, initialSnapshot(iface, inputs), { idempotencyKey });
    },
    read(p, appId, caseId) { application(appId); return store.readCase(p, keyFor(p, appId, caseId)); },
    cases(p, appId, { after, limit } = {}) {
      application(appId);
      return store.listCases(p, { tenantId: p.tenantId, appId, after, limit });
    },
    async invoke(p, appId, caseId, { interface: name, inputs, expectedVersion, idempotencyKey }, { signal } = {}) {
      if (signal?.aborted) error('invocation_cancelled');
      const app = application(appId), iface = ifaceFor(app, name);
      authorize(p, appId, 'invoke:' + name);
      if (typeof idempotencyKey !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,100}$/.test(idempotencyKey)) error('invalid_idempotency_key');
      if (!Number.isSafeInteger(expectedVersion) || expectedVersion < 1) error('invalid_version');
      // Take an immutable request copy before the first await; callers and adapters cannot mutate its binding.
      const supplied = inputs === undefined ? undefined : jsonCopy(inputs);
      if (supplied !== undefined && (!supplied || typeof supplied !== 'object' || Array.isArray(supplied)))
        error('invalid_arguments');
      const key = keyFor(p, appId, caseId), requestHash = hash({ actor: p.id, name, inputs: supplied ?? null, expectedVersion });
      const invocationId = 'call:' + hash({ tenant: p.tenantId, appId, caseId, idempotencyKey }).slice(0,48);
      try {
        const old = store.readInvocation(p, key, invocationId);
        if (old.requestHash !== requestHash || old.requester !== p.id) error('idempotency_conflict');
        if (signal?.aborted) error('invocation_cancelled');
        return old;
      } catch (e) { if (e.code !== 'not_found') throw e; }
      const runningKey = hash({ key, idempotencyKey });
      const running = inflight.get(runningKey);
      if (running) {
        if (running.requestHash !== requestHash) error('idempotency_conflict');
        return await join(running, signal);
      }
      const entry = { key: runningKey, requestHash, controller: new AbortController(), waiters: new Set() };
      inflight.set(runningKey, entry);
      const response = join(entry, signal);
      const task = (async () => {
        const executionSignal = entry.controller.signal;
        if (executionSignal.aborted) error('invocation_cancelled');
        const snapshot = store.readCase(p, key); if (!snapshot) error('not_found');
        if (snapshot.version !== expectedVersion) error('stale_version');
        const saved = snapshot.initial?.interface === name ? snapshot.initial.inputs : undefined;
        if (supplied === undefined && saved === undefined) error('arguments_required');
        const args = structuredClone(supplied ?? saved);
        if (!args || typeof args !== 'object' || Array.isArray(args)) error('invalid_arguments');
        const recordId = iface.stateKey ?? iface.name;
        const prior = snapshot.domainRecords.find(row => row.recordType === 'NativeCheckpoint' && row.recordId === recordId);
        if (iface.stateParameter && prior) {
          const path = [iface.stateParameter, ...(iface.stateInputPath ?? [])];
          const competing = supplied && atPath(supplied, path, true);
          if (competing?.present && hash(competing.value) !== hash(prior.value.state)) error('state_override');
          putPath(args, path, structuredClone(prior.value.state));
        }
        validateArguments(iface, args);
        const returned = await execute({ app, iface, inputs: jsonCopy(args), signal: executionSignal });
        if (executionSignal.aborted) error('invocation_cancelled');
        const result = jsonCopy(returned);
        validateResult(iface, result);
        const carriesState = iface.stateParameter !== undefined || iface.stateResultPath !== undefined;
        const state = carriesState ? atPath(result, iface.stateResultPath ?? []) : null;
        if (carriesState) {
          // Every operation sharing this key must be able to consume the actual saved value.
          for (const other of app.interfaces.values())
            if (other.stateParameter && (other.stateKey ?? other.name) === recordId) validateCheckpoint(other, state);
        }
        const adapter = effectAdapters[appId]?.[name];
        if (iface.effectsPath !== undefined) {
          const intentions = atPath(result, iface.effectsPath);
          if (!Array.isArray(intentions)) error('invalid_effects_path');
          if (intentions.length && !adapter) error('effect_adapter_required');
        }
        const packet = adapter ? await adapter(freeze({ result: jsonCopy(result), inputs: jsonCopy(args),
          principal: jsonCopy(p), caseId, sourceHash: app.sourceHash })) : [];
        if (!Array.isArray(packet)) error('invalid_effect_adapter');
        const outbox = jsonCopy(packet);
        if (executionSignal.aborted) error('invocation_cancelled');
        const invocation = { interface: name, nativePath: iface.path, inputs: args, result, effectAdapter: !!adapter };
        const writes = carriesState ? [{ recordType: 'NativeCheckpoint', recordId,
          expectedVersion: prior?.version ?? null,
          value: { state, interface: name, sourceHash: app.sourceHash, interfaceHash: app.interfaceHash } }] : [];
        return store.commitInvocation(p, key, { invocationId, operation: name, requestHash,
          snapshotVersion: snapshot.version, snapshotHash: snapshot.snapshotHash,
          sourceHash: app.sourceHash, interfaceHash: app.interfaceHash, invocation, writes, outbox });
      })();
      // Attach both handlers immediately so a task that outlives its canceled waiters
      // cannot produce an unhandled rejection.
      task.then(value => finish(entry, 'fulfilled', value), cause => finish(entry, 'rejected', cause));
      return await response;
    },
    invocation(p, appId, caseId, id) { return store.readInvocation(p, keyFor(p, appId, caseId), id); },
    events(p, appId, caseId, page) { return store.listEvents(p, keyFor(p, appId, caseId), page); },
    outbox(p, id) { return store.readOutbox(p, id); },
    reconcile(p, id, binding) { return store.reconcileOutbox(p, id, binding); },
  };
}
