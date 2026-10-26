/** Exact invocation/checkpoint custody. Each program keeps its own parameters, result and state machine. */
import { createHash } from 'node:crypto';
import { ApplicationError, describeApplication, validateArguments, validateResult } from './runtime.mjs';
import { formatType } from '../../ts-host/dist/native/types.js';

const error = (code) => { throw new ApplicationError(code); };
const canonical = value => {
  if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
  if (value && typeof value === 'object') return '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + canonical(value[key])).join(',') + '}';
  const text = JSON.stringify(value);
  if (text === undefined || (typeof value === 'number' && !Number.isFinite(value))) error('invalid_json');
  return text;
};
const hash = value => createHash('sha256').update(canonical(value)).digest('hex');
function atPath(value, path) {
  for (const key of path) {
    if (!value || typeof value !== 'object' || !Object.hasOwn(value, key) || ['__proto__','constructor','prototype'].includes(key))
      error('declared_result_path_missing');
    value = value[key];
  }
  return value;
}

export function createApplicationService({ catalog, store, execute, capabilities, effectAdapters = {} }) {
  if (!(catalog instanceof Map) || !store || typeof execute !== 'function') error('invalid_configuration');
  const inflight = new Map();
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
    validateArguments(iface, inputs);
    return { caseRecords: [{ interface: iface.name, inputs }], evidencePages: [], policyCandidates: [], searchedPartitions: [],
      capturedAt: new Date().toISOString() };
  }
  return {
    list(p) {
      return [...catalog.values()].filter(app => { try { authorize(p, app.id, 'read'); return true; } catch { return false; } }).map(describeApplication);
    },
    describe(p, appId) { authorize(p, appId, 'read'); return describeApplication(application(appId)); },
    create(p, appId, caseId, { interface: name, inputs, idempotencyKey }) {
      const app = application(appId), iface = ifaceFor(app, name);
      authorize(p, appId, 'write');
      return store.createCase(p, keyFor(p, appId, caseId), initialSnapshot(iface, inputs), { idempotencyKey });
    },
    replace(p, appId, caseId, { interface: name, inputs, expectedVersion, idempotencyKey }) {
      const iface = ifaceFor(application(appId), name); authorize(p, appId, 'write');
      return store.replaceCase(p, keyFor(p, appId, caseId), expectedVersion, initialSnapshot(iface, inputs), { idempotencyKey });
    },
    read(p, appId, caseId) { application(appId); return store.readCase(p, keyFor(p, appId, caseId)); },
    cases(p, appId, page = {}) { application(appId); return store.listCases(p, { tenantId: p.tenantId, appId, ...page }); },
    async invoke(p, appId, caseId, { interface: name, inputs, expectedVersion, idempotencyKey }, { signal } = {}) {
      const app = application(appId), iface = ifaceFor(app, name); authorize(p, appId, 'propose');
      if (typeof idempotencyKey !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,100}$/.test(idempotencyKey)) error('invalid_idempotency_key');
      if (!Number.isSafeInteger(expectedVersion) || expectedVersion < 1) error('invalid_version');
      const key = keyFor(p, appId, caseId), requestHash = hash({ actor: p.id, name, inputs: inputs ?? null, expectedVersion });
      const proposalId = 'call:' + hash({ tenant: p.tenantId, appId, caseId, idempotencyKey }).slice(0,48);
      // A durable replay is returned before inference, including after process restart.
      try {
        const old = store.readProposal(p, key, proposalId);
        if (old.proposal.requestHash !== requestHash || old.requester !== p.id) error('idempotency_conflict');
        return old;
      } catch (e) { if (e.code !== 'not_found') throw e; }
      const runningKey = `${p.tenantId}/${appId}/${caseId}/${idempotencyKey}`;
      const running = inflight.get(runningKey);
      if (running) { if (running.requestHash !== requestHash) error('idempotency_conflict'); return running.promise; }
      const promise = (async () => {
        const snapshot = store.readCase(p, key); if (!snapshot) error('not_found');
        if (snapshot.version !== expectedVersion) error('stale_version');
        const saved = snapshot.caseRecords.find(row => row.interface === name);
        const args = structuredClone(inputs ?? saved?.inputs ?? {});
        const recordId = iface.stateKey ?? iface.name;
        const prior = snapshot.domainRecords.find(row => row.recordType === 'NativeCheckpoint' && row.recordId === recordId);
        if (iface.stateParameter && prior) {
          // Saved source inputs describe the initial invocation. Resume always replaces that old
          // state; only a new caller-supplied competing state is an attempted override.
          if (inputs && Object.hasOwn(inputs, iface.stateParameter) && hash(inputs[iface.stateParameter]) !== hash(prior.value.state)) error('state_override');
          args[iface.stateParameter] = structuredClone(prior.value.state);
        }
        validateArguments(iface, args);
        const result = await execute({ app, iface, inputs: args, signal });
        if (signal?.aborted) error('invocation_cancelled');
        validateResult(iface, result);
        const state = atPath(result, iface.stateResultPath ?? []);
        const adapter = effectAdapters[appId]?.[name];
        // Only an explicitly installed application adapter can turn domain intentions into an outbox packet.
        // Without an adapter, the result is persisted for review, and no delivery is claimed.
        const outbox = adapter ? await adapter({ result, inputs: args, principal: p, caseId, sourceHash: app.sourceHash }) : [];
        if (!Array.isArray(outbox)) error('invalid_effect_adapter');
        const value = { id: recordId, tenantId: p.tenantId, version: (prior?.version ?? 0) + 1,
          interface: name, sourceHash: app.sourceHash, interfaceHash: app.interfaceHash, state, result, inputs: args };
        const proposal = { requestHash, interface: name, nativePath: iface.path, sourceHash: app.sourceHash, interfaceHash: app.interfaceHash,
          inputs: args, result, effectAdapter: !!adapter, resultType: formatResult(iface) };
        try {
          return store.saveProposal(p, key, { proposalId, snapshotVersion: snapshot.version, snapshotHash: snapshot.snapshotHash,
            sourceHash: app.sourceHash, proposal, writes: [{ recordType: 'NativeCheckpoint', recordId,
              expectedVersion: prior?.version ?? null, value }], outbox }, { idempotencyKey });
        } catch (e) {
          if (!['already_exists','idempotency_conflict','stale_version'].includes(e.code)) throw e;
          // Another process may have completed this exact read-only inference while this one was running.
          let existing; try { existing = store.readProposal(p, key, proposalId); } catch { throw e; }
          if (existing.proposal.requestHash !== requestHash || existing.requester !== p.id) error('idempotency_conflict');
          return existing;
        }
      })();
      inflight.set(runningKey, { requestHash, promise });
      try { return await promise; } finally { inflight.delete(runningKey); }
    },
    proposal(p, appId, caseId, id) { return store.readProposal(p, keyFor(p, appId, caseId), id); },
    approve(p, appId, caseId, id, { idempotencyKey, ...binding }) {
      return store.approveProposal(p, keyFor(p, appId, caseId), id, binding, { idempotencyKey });
    },
    commit(p, appId, caseId, id, { idempotencyKey, ...binding }) {
      return store.commitProposal(p, keyFor(p, appId, caseId), id, binding, { idempotencyKey });
    },
    reject(p, appId, caseId, id, { idempotencyKey, ...binding }) {
      return store.rejectProposal(p, keyFor(p, appId, caseId), id, binding, { idempotencyKey });
    },
    events(p, appId, caseId, page) { return store.listEvents(p, keyFor(p, appId, caseId), page); },
    outbox(p, id) { return store.readOutbox(p, id); },
    reconcile(p, id, binding) { return store.reconcileOutbox(p, id, binding); },
  };
}
function formatResult(iface) { return formatType(iface.node.type.returns); }
