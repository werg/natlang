import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { openDesignBriefStore } from './store.mjs';

const capabilities = { brief: {
  read: ['reader','operator'], write: ['operator'], 'invoke:post': ['operator'],
  'invoke:release': ['releaser'], dispatch: ['dispatcher'], reconcile: ['reconciler'],
} };
const op = { id: 'op-1', tenantId: 'tenant-1', roles: ['operator'] };
const dispatcher = { id: 'worker', tenantId: 'tenant-1', roles: ['dispatcher'] };
const reconciler = { id: 'reconciler', tenantId: 'tenant-1', roles: ['reconciler'] };
const reader = { id: 'reader', tenantId: 'tenant-1', roles: ['reader'] };
const key = { tenantId: 'tenant-1', appId: 'brief', caseId: 'case-1' };
const snapshot = () => ({ initial: { interface: 'post', inputs: { passage: 'original', edition: 4 } } });
const hash = value => createHash('sha256').update(value).digest('hex');
const code = expected => error => error.code === expected;
const receipt = entryKey => ({ status: 'delivered', idempotencyKey: entryKey, receiptId: 'receiver-' + entryKey });
const evidence = {kind:'operator_assertion',reference:'receiver-audit',assertion:'Checked the receiver journal for this exact key.'};

function environment(t, clock) {
  const root = mkdtempSync(join(process.cwd(), '.test-design-brief-'));
  const filename = join(root, 'state.sqlite'), stores = [];
  const open = () => {
    const store = openDesignBriefStore({filename,capabilities,...(clock ? {clock} : {})});
    stores.push(store); return store;
  };
  t.after(() => { for (const store of stores) try { store.close(); } catch {} rmSync(root,{recursive:true,force:true}); });
  return {store:open(),open};
}
function invocation(snap, overrides = {}) {
  return {invocationId:'inv-1',operation:'post',requestHash:hash('request-1'),snapshotVersion:snap.version,
    snapshotHash:snap.snapshotHash,sourceHash:hash('source'),interfaceHash:hash('interface'),
    invocation:{result:{edition:5,acknowledged:false}},
    writes:[{recordType:'NativeCheckpoint',recordId:'edition',expectedVersion:null,value:{state:{edition:5}}}],
    outbox:[{key:'notify-1',topic:'edition.ready',payload:{edition:5}}],...overrides};
}
function create(store, caseKey = key) {
  return store.createCase(op,caseKey,snapshot(),{idempotencyKey:'create'});
}
function commit(store, overrides) {
  const snap = store.readCase(reader,key) ?? create(store);
  return store.commitInvocation(op,key,invocation(snap,overrides));
}

test('one atomic native invocation persists result, checkpoint, audit and outbox across restart', t => {
  const e=environment(t), first=commit(e.store);
  assert.equal(first.version,2);
  const reopened=e.open(), snap=reopened.readCase(reader,key);
  assert.deepEqual(snap.initial,snapshot().initial);
  assert.equal(Object.hasOwn(snap,'policyCandidates'),false);
  assert.deepEqual(snap.domainRecords[0].value,{state:{edition:5}});
  assert.equal(snap.domainRecords[0].version,1);
  assert.deepEqual(reopened.readInvocation(reader,key,first.id),first);
  assert.equal(reopened.readOutbox(dispatcher,1).status,'pending');
  assert.deepEqual(reopened.listEvents(reader,key).items.map(e=>e.operation),['case.created','invocation.committed']);
});

test('different domain operations have independent capabilities; there is no shared approval workflow', t => {
  const {store}=environment(t), snap=create(store);
  assert.throws(()=>store.commitInvocation(op,key,invocation(snap,{operation:'release'})),code('forbidden'));
  assert.throws(()=>store.commitInvocation(reader,key,invocation(snap)),code('forbidden'));
  assert.equal(store.readCase(reader,key).version,1);
  assert.equal('approveProposal' in store,false);
});

test('concurrent connections replay one request and reject changed request or actor identities', t => {
  const e=environment(t), snap=create(e.store), other=e.open(), input=invocation(snap);
  const first=e.store.commitInvocation(op,key,input);
  assert.deepEqual(other.commitInvocation(op,key,{...input,invocation:{result:'another model completion'}}),first);
  assert.throws(()=>other.commitInvocation(op,key,{...input,requestHash:hash('changed')}),code('idempotency_conflict'));
  assert.throws(()=>other.commitInvocation({...op,id:'another'},key,input),code('idempotency_conflict'));
  assert.equal(other.readCase(reader,key).version,2);
  assert.equal(other.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'worker'}).length,1);
});

test('stale snapshots, altered snapshot hashes and competing checkpoint versions cannot partly commit', t => {
  const {store}=environment(t), before=create(store);
  assert.throws(()=>store.commitInvocation(op,key,invocation(before,{snapshotHash:hash('different')})),code('stale_version'));
  const first=store.commitInvocation(op,key,invocation(before));
  assert.throws(()=>store.commitInvocation(op,key,invocation(before,{invocationId:'inv-2',requestHash:hash('request-2')})),code('stale_version'));
  const now=store.readCase(reader,key);
  assert.throws(()=>store.commitInvocation(op,key,invocation(now,{invocationId:'inv-2',requestHash:hash('request-2')})),code('record_version_mismatch'));
  assert.equal(store.readCase(reader,key).version,first.version);
  assert.throws(()=>store.readInvocation(reader,key,'inv-2'),code('not_found'));
});

test('effect identities are durable: exact repeated intentions never resend, conflicting content rolls back', t => {
  const {store}=environment(t);commit(store);
  const [item]=store.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'worker'});
  store.ackOutbox(dispatcher,item.id,'worker',receipt(item.key));
  const now=store.readCase(reader,key);
  const next=invocation(now,{invocationId:'inv-2',requestHash:hash('request-2'),
    writes:[{recordType:'NativeCheckpoint',recordId:'edition',expectedVersion:1,value:{state:{edition:6}}}]});
  assert.throws(()=>store.commitInvocation(op,key,{...next,outbox:[{key:'notify-1',topic:'edition.ready',payload:{edition:6}}]}),code('effect_identity_conflict'));
  assert.equal(store.readCase(reader,key).version,2);
  assert.equal(store.readCase(reader,key).domainRecords[0].value.state.edition,5);
  const second=store.commitInvocation(op,key,next);
  assert.equal(second.outbox[0].status,'delivered');
  assert.deepEqual(store.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'next'}),[]);
});

test('source refresh keeps typed checkpoints and invalidates in-flight snapshots mechanically', t => {
  const {store}=environment(t);commit(store);
  const old=store.readCase(reader,key);
  const changed=store.replaceCase(op,key,old.version,{initial:{interface:'post',inputs:{passage:'revised',edition:6}}},{idempotencyKey:'refresh'});
  assert.equal(changed.initial.inputs.passage,'revised');
  assert.deepEqual(changed.domainRecords,old.domainRecords);
  assert.throws(()=>store.commitInvocation(op,key,invocation(old,{invocationId:'slow',requestHash:hash('slow')})),code('stale_version'));
});

test('cross-tenant isolation and actor-bound create idempotency survive a second connection', t => {
  const e=environment(t), created=create(e.store), other=e.open();
  assert.deepEqual(create(other),created);
  assert.throws(()=>other.createCase({...op,id:'other'},key,snapshot(),{idempotencyKey:'create'}),code('idempotency_conflict'));
  assert.throws(()=>other.readCase({...reader,tenantId:'tenant-2'},key),code('not_found'));
  assert.throws(()=>other.createCase(reader,key,snapshot(),{idempotencyKey:'reader'}),code('forbidden'));
  assert.throws(()=>other.readInvocation({...reader,tenantId:'tenant-2'},key,'inv-1'),code('not_found'));
});

test('expired or uncertain delivery stays unresolved until an authorized evidence-bound reconciliation', t => {
  let time=1000;const e=environment(t,()=>time);commit(e.store);
  const [item]=e.store.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'worker',leaseMs:100});
  assert.throws(()=>e.store.ackOutbox(dispatcher,item.id,'worker',receipt('another-key')),code('receipt_mismatch'));
  time=1200;const reopened=e.open();
  assert.deepEqual(reopened.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'next'}),[]);
  assert.equal(reopened.readOutbox(dispatcher,item.id).status,'unknown');
  assert.throws(()=>reopened.ackOutbox(dispatcher,item.id,'worker',receipt(item.key)),code('lease_lost'));
  assert.throws(()=>reopened.reconcileOutbox(dispatcher,item.id,{outcome:'not_sent',evidence}),code('forbidden'));
  assert.throws(()=>reopened.reconcileOutbox(reconciler,item.id,{outcome:'delivered',receipt:receipt('wrong'),evidence}),code('receipt_mismatch'));
  reopened.reconcileOutbox(reconciler,item.id,{outcome:'delivered',receipt:receipt(item.key),evidence});
  assert.equal(reopened.readOutbox(dispatcher,item.id).status,'delivered');
  assert.ok(reopened.listEvents(reader,key).items.some(e=>e.operation==='outbox.reconciled'&&e.actor===reconciler.id));
});

test('not-sent evidence requeues the same packet without changing its identity or losing attempt count', t => {
  const {store}=environment(t);commit(store);
  const [item]=store.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'worker'});
  store.failOutbox(dispatcher,item.id,'worker',{code:'timeout',ambiguous:true});
  assert.deepEqual(store.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'next'}),[]);
  assert.throws(()=>store.reconcileOutbox(reconciler,item.id,{outcome:'not_sent'}),code('evidence_required'));
  store.reconcileOutbox(reconciler,item.id,{outcome:'not_sent',evidence});
  const [retry]=store.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'next'});
  assert.equal(retry.id,item.id);assert.equal(retry.key,item.key);assert.equal(retry.attempt,2);
});

test('pagination is scoped and arbitrary native JSON is stored without a common domain record schema', t => {
  const {store}=environment(t);create(store,{...key,caseId:'case-a'});create(store,{...key,caseId:'case-b'});
  const first=store.listCases(reader,{tenantId:key.tenantId,appId:key.appId,limit:1});
  assert.equal(first.next,'case-a');
  assert.equal(store.listCases(reader,{tenantId:key.tenantId,appId:key.appId,after:first.next,limit:1}).items[0].caseId,'case-b');
  const snap=create(store);
  store.commitInvocation(op,key,invocation(snap,{writes:[{recordType:'NativeCheckpoint',recordId:'poem',expectedVersion:null,value:['line one','line two']}],outbox:[]}));
  assert.deepEqual(store.readCase(reader,key).domainRecords[0].value,['line one','line two']);
});
