import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import referralAdapter from './adapters/076-referral-packet-transfer.mjs';
import documentaryAdapter from './adapters/092-documentary-delivery.mjs';
import liveEventAdapter from './adapters/097-live-event-effects.mjs';
import worldAdapter from './adapters/098-persistent-world-actions.mjs';
import { openDesignBriefStore } from './store.mjs';

const operator = { id: 'adapter-operator', tenantId: 'adapter-tenant', roles: ['operator'] };
const dispatcher = { id: 'adapter-worker', tenantId: 'adapter-tenant', roles: ['dispatcher'] };
const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const adapters = [
  {
    appId: '076', operation: 'manageReferralPacket', adapter: referralAdapter,
    effects: [
      { kind: 'send-referral-packet', effectId: 'transfer-1', packetId: 'packet-1', packetRevision: 'r3', patientId: 'patient-1', episodeScope: 'episode-1', recipientServiceId: 'service-1', documentIds: ['doc-1'], authorizationRef: 'authorization-1' },
      { kind: 'read-recipient-status', effectId: 'read-1', packetId: 'packet-1', packetRevision: 'r3', patientId: 'patient-1', episodeScope: 'episode-1', recipientServiceId: 'service-1', documentIds: ['doc-1'], authorizationRef: 'authorization-1' },
    ],
    topics: ['076.referral.packet.send', '076.referral.status.read'],
    change(effect) { effect.packetRevision = 'r4'; },
  },
  {
    appId: '092', operation: 'prepareDocumentaryDelivery', adapter: documentaryAdapter,
    effects: [
      { kind: 'deliver-documentary-package', assemblyId: 'assembly-1', revision: 'r4', digest: 'sha256:bundle-4', deliverables: [{ kind: 'captions', revision: 'r4', assemblyDigest: 'sha256:bundle-4' }], clearanceRefs: ['clearance-1'], idempotencyKey: 'delivery-1' },
      { kind: 'read-documentary-delivery-status', assemblyId: 'assembly-1', revision: 'r4', digest: 'sha256:bundle-4', deliverables: [], clearanceRefs: ['clearance-1'], idempotencyKey: 'status-1' },
    ],
    topics: ['092.documentary.package.deliver', '092.documentary.delivery.status.read'],
    change(effect) { effect.digest = 'sha256:bundle-changed'; },
  },
  {
    appId: '097', operation: 'prepareLiveEventEffects', adapter: liveEventAdapter,
    effects: [
      { kind: 'reserve-event-resources', eventId: 'event-1', revision: 'r8', venueId: 'venue-1', producerId: 'producer-1', resourceIds: ['stage-1'], roleUpdates: [], idempotencyKey: 'reserve-1' },
      { kind: 'request-role-acknowledgements', eventId: 'event-1', revision: 'r8', venueId: 'venue-1', producerId: 'producer-1', resourceIds: [], roleUpdates: [{ role: 'safety', revision: 'r8', status: 'pending', receivedAt: 0 }], idempotencyKey: 'ack-1' },
      { kind: 'publish-event-revision', eventId: 'event-1', revision: 'r8', venueId: 'venue-1', producerId: 'producer-1', resourceIds: [], roleUpdates: [], idempotencyKey: 'publish-1' },
    ],
    topics: ['097.live-event.resources.reserve', '097.live-event.roles.acknowledge', '097.live-event.revision.publish'],
    change(effect) { effect.revision = 'r9'; },
  },
  {
    appId: '098', operation: 'prepareWorldActionEffects', adapter: worldAdapter,
    effects: [
      { kind: 'apply-player-actions', worldId: 'world-1', priorRevision: 12, ruleVersion: 'rules-2', clientActionId: 'action-1', actions: [{ id: 'legal-1', actorId: 'actor-1', kind: 'move', targetId: 'gate-1', location: 'north', resourceDelta: 'none', requires: [], clientActionId: 'action-1' }], idempotencyKey: 'world-action-1' },
      { kind: 'read-action-status', worldId: 'world-1', priorRevision: 12, ruleVersion: 'rules-2', clientActionId: 'action-1', actions: [], idempotencyKey: 'world-status-1' },
    ],
    topics: ['098.persistent-world.actions.apply', '098.persistent-world.action.status.read'],
    change(effect) { effect.priorRevision = 13; },
  },
];

function openStore(t) {
  const directory = mkdtempSync(join(process.cwd(), '.test-effect-adapters-'));
  const filename = join(directory, 'state.sqlite');
  const capabilities = Object.fromEntries(adapters.map(({ appId, operation }) => [appId, {
    read: ['operator', 'dispatcher'], write: ['operator'], [`invoke:${operation}`]: ['operator'], dispatch: ['dispatcher'],
  }]));
  const store = openDesignBriefStore({ filename, capabilities });
  t.after(() => { try { store.close(); } catch {} rmSync(directory, { recursive: true, force: true }); });
  return store;
}

function commitInput({ store, appId, operation, caseId, invocationId, result, outbox, version, snapshotHash }) {
  return store.commitInvocation(operator, { tenantId: operator.tenantId, appId, caseId }, {
    invocationId, operation, requestHash: hash({ invocationId, result }), snapshotVersion: version, snapshotHash,
    sourceHash: hash({ appId, source: 'native-source' }), interfaceHash: hash({ appId, operation }),
    invocation: { interface: operation, inputs: {}, result }, writes: [], outbox,
  });
}

test('application-specific adapters serialize each checked effect kind into durable receiver packets', t => {
  const store = openStore(t);
  for (const { appId, operation, adapter, effects, topics, change } of adapters) {
    const caseId = `case-${appId}`;
    const key = { tenantId: operator.tenantId, appId, caseId };
    store.createCase(operator, key, { initial: { interface: operation, inputs: {} } }, { idempotencyKey: `create-${appId}` });

    const result = { effects };
    const packet = adapter({ result, inputs: {}, principal: operator, caseId, sourceHash: hash(appId) });
    assert.deepEqual(packet.map(row => row.topic), topics);
    assert.equal(new Set(packet.map(row => row.key)).size, effects.length);
    for (const [index, row] of packet.entries()) {
      assert.match(row.key, /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/);
      assert.deepEqual(row.payload, effects[index], 'the original checked intent is the exact receiver payload');
    }
    assert.deepEqual(adapter({ result, inputs: {}, principal: operator, caseId, sourceHash: hash(appId) }), packet,
      'reconstructing the same checked result yields byte-stable packet identities');

    const before = store.readCase(operator, key);
    const committed = commitInput({ store, appId, operation, caseId, invocationId: `call-${appId}-1`, result,
      outbox: packet, version: before.version, snapshotHash: before.snapshotHash });
    const replay = commitInput({ store, appId, operation, caseId, invocationId: `call-${appId}-1`, result,
      outbox: packet, version: before.version, snapshotHash: before.snapshotHash });
    assert.deepEqual(replay, committed, 'the store replays the durable invocation and does not duplicate outbox rows');
    assert.equal(committed.outbox.length, effects.length);

    const altered = structuredClone(effects[0]);
    change(altered);
    const changedPacket = adapter({ result: { effects: [altered] }, inputs: {}, principal: operator, caseId, sourceHash: hash(appId) });
    assert.equal(changedPacket[0].key, packet[0].key, 'domain identity and operation kind, not payload bytes, own the key');
    const current = store.readCase(operator, key);
    assert.throws(() => commitInput({ store, appId, operation, caseId, invocationId: `call-${appId}-2`,
      result: { effects: [altered] }, outbox: changedPacket, version: current.version, snapshotHash: current.snapshotHash }),
    error => error.code === 'effect_identity_conflict');
    assert.equal(store.readCase(operator, key).version, current.version, 'a changed payload cannot partially commit');
    for (let id = committed.outbox[0].id; id <= committed.outbox.at(-1).id; id++) {
      const row = store.readOutbox(dispatcher, id);
      assert.equal(row.status, 'pending');
    }
  }
});

test('closed-kind adapters reject unknown effects instead of routing a fallback topic', () => {
  for (const { adapter } of adapters) {
    assert.throws(() => adapter({ result: { effects: [{ kind: 'unrecognized-operation', idempotencyKey: 'unknown', effectId: 'unknown' }] } }),
      error => error.code === 'invalid_effect_adapter');
    assert.throws(() => adapter({ result: { effects: null } }), error => error.code === 'invalid_effect_adapter');
  }
});
