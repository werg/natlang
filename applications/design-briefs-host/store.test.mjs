import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { openDesignBriefStore, StoreError } from './store.mjs';

const capabilities = { brief: {
  read: ['reader','operator','reviewer'], write: ['operator'], propose: ['operator'], approve: ['reviewer'],
  commit: ['operator'], reject: ['reviewer','operator'], dispatch: ['dispatcher'],
} };
const op = { id: 'op-1', tenantId: 'tenant-1', roles: ['operator'] };
const reviewer = { id: 'reviewer-1', tenantId: 'tenant-1', roles: ['reviewer'] };
const dispatcher = { id: 'worker-principal', tenantId: 'tenant-1', roles: ['dispatcher'] };
const reader = { id: 'reader-1', tenantId: 'tenant-1', roles: ['reader'] };
const key = { tenantId: 'tenant-1', appId: 'brief', caseId: 'case-1' };
const snapshot = () => ({ caseRecords: [{ customer: { id: 'c-1' } }], capturedAt: '2026-10-26T00:00:00Z', previousEvidencePages: [{ id: 'previous-1' }], evidencePages: [{ id: 'page-1' }], policyCandidates: [], searchedPartitions: [] });
const hash = value => createHash('sha256').update(value).digest('hex');
const tempDb = () => {
  const dir = mkdtempSync(join(process.cwd(), '.test-design-brief-'));
  return { filename: join(dir, 'store.sqlite'), cleanup: () => rmSync(dir, { recursive: true, force: true }) };
};
const open = (filename, clock) => openDesignBriefStore({ filename, capabilities, ...(clock ? { clock } : {}) });
function proposalFrom(caseSnapshot, { id = 'proposal-1', writes = [{ recordType: 'decision', recordId: 'd-1', value: { id: 'd-1', tenantId: key.tenantId, version: 1, result: 'hold' }, expectedVersion: null }], outbox = [{ key: 'notify-1', topic: 'review.ready', payload: { caseId: key.caseId } }] } = {}) {
  return { proposalId: id, snapshotVersion: caseSnapshot.version, snapshotHash: caseSnapshot.snapshotHash, sourceHash: hash('source'),
    proposal: { disposition: 'review', explanation: 'Evidence retained.' }, writes, outbox };
}
function make(store, opts) {
  const c = store.createCase(op, key, snapshot(), { idempotencyKey: opts?.createKey ?? 'create-1' });
  const p = store.saveProposal(op, key, proposalFrom(c, opts), { idempotencyKey: opts?.proposalKey ?? 'save-1' });
  return { c, p };
}
function approve(store, p, { idempotencyKey = 'approve-1' } = {}) {
  return store.approveProposal(reviewer, key, p.id, { proposalHash: p.hash, snapshotVersion: p.snapshotVersion }, { idempotencyKey });
}
function commit(store, p, a, { idempotencyKey = 'commit-1' } = {}) {
  return store.commitProposal(op, key, p.id, { proposalHash: p.hash, approvalId: a.id, approvalHash: a.hash,
    snapshotVersion: p.snapshotVersion }, { idempotencyKey });
}
const code = (error, expected) => error instanceof StoreError && error.code === expected;

test('durable case/proposal/approval/commit survive close and reopen with typed inputs separate from materialized domain records', () => {
  const t = tempDb();
  try {
    let store = open(t.filename); const { c, p } = make(store); const a = approve(store,p); const result = commit(store,p,a);
    assert.equal(result.status, 'committed'); assert.equal(result.version, 2);
    store.close();
    store = open(t.filename);
    const restored = store.readCase(reader,key);
    assert.equal(restored.caseRecords[0].customer.id, 'c-1');
    assert.equal(restored.caseId,key.caseId);assert.equal(restored.capturedAt,'2026-10-26T00:00:00Z');assert.deepEqual(restored.previousEvidencePages,[{id:'previous-1'}]);
    assert.equal(restored.domainRecords[0].recordType, 'decision');
    assert.deepEqual(restored.domainRecords[0].value,{id:'d-1',tenantId:key.tenantId,version:1,result:'hold'});
    assert.equal(restored.domainRecords[0].version,1);assert.equal(restored.domainRecords[0].value.version,1);
    assert.deepEqual(store.listCases(reader,{tenantId:key.tenantId,appId:key.appId,limit:1}).next,null);
    assert.equal(store.readProposal(reader,key,p.id).status,'committed');
    assert.equal(store.readOutbox(dispatcher,1).status,'pending');
    assert.deepEqual(store.listEvents(reader,key).items.map(e=>e.operation),['case.created','proposal.saved','proposal.approved','proposal.committed']);
    store.close();
  } finally { t.cleanup(); }
});

test('approval is a separate authorized exact binding; requester cannot approve and tampering blocks commit', () => {
  const t=tempDb();try{const s=open(t.filename);const {p}=make(s);
    assert.throws(()=>s.approveProposal(op,key,p.id,{proposalHash:p.hash,snapshotVersion:p.snapshotVersion},{idempotencyKey:'self'}),e=>code(e,'forbidden'));
    const selfAllowed={id:op.id,tenantId:op.tenantId,roles:['reviewer']};
    assert.throws(()=>s.approveProposal(selfAllowed,key,p.id,{proposalHash:p.hash,snapshotVersion:p.snapshotVersion},{idempotencyKey:'self2'}),e=>code(e,'separation_of_duties'));
    const a=approve(s,p);assert.throws(()=>s.commitProposal(op,key,p.id,{proposalHash:p.hash,approvalId:a.id,approvalHash:hash('tampered'),snapshotVersion:p.snapshotVersion},{idempotencyKey:'bad'}),e=>code(e,'binding_mismatch'));
    assert.equal(s.readProposal(reader,key,p.id).status,'approved');s.close();}finally{t.cleanup();}
});

test('cross-tenant and role isolation fail closed without revealing whether another tenant has a case', () => {
  const t=tempDb();try{const s=open(t.filename);make(s);
    assert.throws(()=>s.readCase({...reader,tenantId:'tenant-2'},{...key,tenantId:'tenant-1'}),e=>code(e,'not_found'));
    assert.throws(()=>s.createCase(reader,{...key,caseId:'case-2'},snapshot(),{idempotencyKey:'x'}),e=>code(e,'forbidden')); 
    assert.throws(()=>s.readProposal(reader,{...key,tenantId:'tenant-2'},'proposal-1'),e=>code(e,'not_found'));
    s.close();}finally{t.cleanup();}
});

test('concurrent proposals bind versions, commit makes sibling proposals stale, record versions are optimistic', () => {
  const t=tempDb();try{const s=open(t.filename);const c=s.createCase(op,key,snapshot(),{idempotencyKey:'create'});
    const a=s.saveProposal(op,key,proposalFrom(c,{id:'p-a'}),{idempotencyKey:'save-a'});
    const b=s.saveProposal(op,key,proposalFrom(c,{id:'p-b'}),{idempotencyKey:'save-b'});
    const aa=approve(s,a,{idempotencyKey:'approve-a'});commit(s,a,aa);
    assert.equal(s.readProposal(reader,key,'p-b').status,'stale');
    assert.throws(()=>approve(s,b,{idempotencyKey:'approve-b'}),e=>code(e,'stale_version'));
    const after=s.readCase(reader,key);
    assert.throws(()=>s.saveProposal(op,key,{...proposalFrom(after,{id:'p-c',writes:[{recordType:'decision',recordId:'d-1',value:{id:'d-1',tenantId:key.tenantId,version:1,result:'new'},expectedVersion:null}]}),sourceHash:hash('src')},{idempotencyKey:'save-c'}),e=>code(e,'record_version_mismatch'));
    s.close();}finally{t.cleanup();}
});

test('durable idempotency returns exact replay and rejects key reuse with changed payload', () => {
  const t=tempDb();try{const s=open(t.filename);const a=s.createCase(op,key,snapshot(),{idempotencyKey:'same'});assert.deepEqual(s.createCase(op,key,snapshot(),{idempotencyKey:'same'}),a);
    assert.throws(()=>s.createCase(op,key,{...snapshot(),evidencePages:[{id:'other'}]},{idempotencyKey:'same'}),e=>code(e,'idempotency_conflict'));s.close();}finally{t.cleanup();}
});

test('expired outbox lease after a lost acknowledgment becomes unknown and is reconciled explicitly after restart', () => {
  const t=tempDb();try{let now=1000;let s=open(t.filename,()=>now);const {p}=make(s);const a=approve(s,p);commit(s,p,a);const [claim]=s.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'worker-1',leaseMs:100,limit:1});
    now=1200;s.close();s=open(t.filename,()=>now);
    assert.equal(s.readOutbox(dispatcher,claim.id).status,'leased');
    assert.deepEqual(s.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'worker-2',leaseMs:100,limit:1}),[]);
    assert.equal(s.readOutbox(dispatcher,claim.id).status,'unknown');
    assert.deepEqual(s.reconcileOutbox(dispatcher,claim.id,{outcome:'delivered',receipt:{remoteId:'receipt-1'},evidence:{kind:'remote_receipt',reference:'receipt-1'}}),{id:claim.id,status:'delivered',evidence:{kind:'remote_receipt',reference:'receipt-1'}});
    assert.equal(s.readOutbox(dispatcher,claim.id).receipt.remoteId,'receipt-1');
    assert.deepEqual(s.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'worker-3',leaseMs:100,limit:1}),[]);s.close();
  }finally{t.cleanup();}
});

test('failed and unknown sends are not automatically retried; only explicit not-sent reconciliation requeues', () => {
  const t=tempDb();try{const s=open(t.filename);const {p}=make(s);const a=approve(s,p);commit(s,p,a);const [claim]=s.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'worker-1',leaseMs:1000,limit:1});
    assert.deepEqual(s.failOutbox(dispatcher,claim.id,'worker-1',{code:'timeout',ambiguous:true}),{id:claim.id,status:'unknown',code:'timeout'});
    assert.deepEqual(s.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'worker-2',leaseMs:1000,limit:1}),[]);
    assert.deepEqual(s.reconcileOutbox(dispatcher,claim.id,{outcome:'not_sent',evidence:{kind:'not_sent_confirmation',reference:'operator-check-22'}}),{id:claim.id,status:'pending',evidence:{kind:'not_sent_confirmation',reference:'operator-check-22'}});
    assert.equal(s.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'worker-2',leaseMs:1000,limit:1})[0].attempt,2);s.close();
  }finally{t.cleanup();}
});


test('case pagination returns stable case ids and domain writes require exact tenant/id/record-version data', () => {
  const t=tempDb();try{const s=open(t.filename);s.createCase(op,{...key,caseId:'case-a'},snapshot(),{idempotencyKey:'a'});s.createCase(op,{...key,caseId:'case-b'},snapshot(),{idempotencyKey:'b'});
    const first=s.listCases(reader,{tenantId:key.tenantId,appId:key.appId,limit:1});assert.equal(first.items[0].caseId,'case-a');assert.equal(first.next,'case-a');
    assert.equal(s.listCases(reader,{tenantId:key.tenantId,appId:key.appId,after:first.next,limit:1}).items[0].caseId,'case-b');
    const current=s.readCase(reader,{...key,caseId:'case-a'});
    const bad=[{value:{id:'different',tenantId:key.tenantId,version:1},expectedVersion:null},{value:{id:'d',tenantId:'other-tenant',version:1},expectedVersion:null},{value:{id:'d',tenantId:key.tenantId,version:7},expectedVersion:null}];
    for(let i=0;i<bad.length;i++) assert.throws(()=>s.saveProposal(op,{...key,caseId:'case-a'},{...proposalFrom(current,{id:`bad-${i}`,writes:[{recordType:'decision',recordId:'d',...bad[i]}]}),sourceHash:hash('src')},{idempotencyKey:`bad-${i}`}),e=>code(e,'invalid_write_packet'));
    s.close();}finally{t.cleanup();}
});

test('idempotency is actor-bound and replay lookup works across two open SQLite connections', () => {
  const t=tempDb();try{const a=open(t.filename),b=open(t.filename);const created=a.createCase(op,key,snapshot(),{idempotencyKey:'shared'});
    assert.deepEqual(b.createCase(op,key,snapshot(),{idempotencyKey:'shared'}),created);
    const approvalKey={...key,caseId:'actor-case'};const c=a.createCase(op,approvalKey,snapshot(),{idempotencyKey:'actor-case-create'});
    const p=a.saveProposal(op,approvalKey,proposalFrom(c,{id:'actor-proposal'}),{idempotencyKey:'p1'});
    const first=a.approveProposal(reviewer,approvalKey,p.id,{proposalHash:p.hash,snapshotVersion:p.snapshotVersion},{idempotencyKey:'shared-approval'});
    const secondReviewer={id:'reviewer-2',tenantId:key.tenantId,roles:['reviewer']};
    assert.throws(()=>b.approveProposal(secondReviewer,approvalKey,p.id,{proposalHash:p.hash,snapshotVersion:p.snapshotVersion},{idempotencyKey:'shared-approval'}),e=>code(e,'idempotency_conflict'));
    assert.equal(first.approver,reviewer.id);a.close();b.close();}finally{t.cleanup();}
});

test('expired lease and failed delivery reconciliation are audited, and requeue requires durable evidence', () => {
  const t=tempDb();try{let now=3000;const s=open(t.filename,()=>now);const {p}=make(s);const a=approve(s,p);commit(s,p,a);
    const [claim]=s.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'worker-x',leaseMs:100,limit:1});
    assert.deepEqual(s.failOutbox(dispatcher,claim.id,'worker-x',{code:'remote_rejected'}),{id:claim.id,status:'failed',code:'remote_rejected'});
    assert.throws(()=>s.reconcileOutbox(dispatcher,claim.id,{outcome:'not_sent'}),e=>code(e,'evidence_required'));
    const result=s.reconcileOutbox(dispatcher,claim.id,{outcome:'not_sent',evidence:{kind:'remote_negative_receipt',reference:'receipt-negative-3'}});
    assert.equal(result.status,'pending');assert.deepEqual(s.readOutbox(dispatcher,claim.id).reconciliation,{kind:'remote_negative_receipt',reference:'receipt-negative-3'});
    const [next]=s.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'worker-y',leaseMs:100,limit:1});now=3200;
    assert.deepEqual(s.claimOutbox(dispatcher,{tenantId:key.tenantId,appId:key.appId,workerId:'worker-z',leaseMs:100,limit:1}),[]);
    assert.equal(s.readOutbox(dispatcher,next.id).status,'unknown');
    assert.ok(s.listEvents(reader,key).items.some(e=>e.operation==='outbox.claimed'));
    assert.ok(s.listEvents(reader,key).items.some(e=>e.operation==='outbox.unknown'));
    assert.ok(s.listEvents(reader,key).items.some(e=>e.operation==='outbox.reconciled'));
    s.close();}finally{t.cleanup();}
});
