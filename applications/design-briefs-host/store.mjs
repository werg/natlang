/** Durable, exact store shared by the native design-brief applications. No policy or model code lives here. */
import { createHash } from 'node:crypto';
import { closeSync, openSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';

const ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const MAX_JSON_BYTES = 2_000_000;
const DEFAULT_PAGE = 50;
const MAX_PAGE = 200;
const MAX_LEASE_MS = 15 * 60_000;

export class StoreError extends Error {
  constructor(code, message = code) { super(message); this.name = 'StoreError'; this.code = code; }
}
const fail = (code) => { throw new StoreError(code); };
function assertId(value, code = 'invalid_input') { if (typeof value !== 'string' || !ID.test(value)) fail(code); return value; }
function canonical(value) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'number') { if (!Number.isFinite(value)) fail('invalid_input'); return JSON.stringify(value); }
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype) fail('invalid_input');
  return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${canonical(value[k])}`).join(',')}}`;
}
function json(value) {
  const text = canonical(value);
  if (Buffer.byteLength(text) > MAX_JSON_BYTES) fail('payload_too_large');
  return text;
}
const digest = value => createHash('sha256').update(typeof value === 'string' ? value : json(value)).digest('hex');
function parse(text) { try { return JSON.parse(text); } catch { fail('storage_corrupt'); } }
function keyOf(key) {
  if (!key || typeof key !== 'object') fail('invalid_input');
  return { tenantId: assertId(key.tenantId), appId: assertId(key.appId), caseId: assertId(key.caseId) };
}
function principalOf(p) {
  if (!p || typeof p !== 'object') fail('unauthorized');
  assertId(p.id, 'unauthorized'); assertId(p.tenantId, 'unauthorized');
  if (!Array.isArray(p.roles) || p.roles.length > 64 || p.roles.some(x => typeof x !== 'string' || !ID.test(x))) fail('unauthorized');
  if (p.scopes != null && (!Array.isArray(p.scopes) || p.scopes.some(x => typeof x !== 'string' || !ID.test(x)))) fail('unauthorized');
  return { id: p.id, tenantId: p.tenantId, roles: [...new Set(p.roles)], scopes: p.scopes ? [...new Set(p.scopes)] : null };
}
function pageArgs({ after = 0, limit = DEFAULT_PAGE } = {}) {
  if (!Number.isSafeInteger(after) || after < 0 || !Number.isSafeInteger(limit) || limit < 1 || limit > MAX_PAGE) fail('invalid_pagination');
  return { after, limit };
}
function snapshotInput(s) {
  if (!s || typeof s !== 'object' || Array.isArray(s)) fail('invalid_snapshot');
  const out = {};
  const arrays = ['caseRecords', 'evidencePages', 'policyCandidates', 'searchedPartitions'];
  for (const name of arrays) {
    const value = s[name] ?? [];
    if (!Array.isArray(value)) fail('invalid_snapshot');
    out[name] = value;
  }
  if (s.capturedAt !== undefined) out.capturedAt = s.capturedAt;
  if (s.previousEvidencePages !== undefined) {
    if (!Array.isArray(s.previousEvidencePages)) fail('invalid_snapshot');
    out.previousEvidencePages = s.previousEvidencePages;
  }
  json(out);
  return out;
}
function writePacket(input) {
  if (!Array.isArray(input) || input.length > 500) fail('invalid_write_packet');
  const seen = new Set();
  return input.map(row => {
    if (!row || typeof row !== 'object') fail('invalid_write_packet');
    const recordType = assertId(row.recordType), recordId = assertId(row.recordId);
    const id = `${recordType}\0${recordId}`;
    if (seen.has(id)) fail('invalid_write_packet'); seen.add(id);
    if (!(row.expectedVersion === null || (Number.isSafeInteger(row.expectedVersion) && row.expectedVersion >= 1))) fail('invalid_write_packet');
    json(row.value);
    return { recordType, recordId, value: row.value, expectedVersion: row.expectedVersion };
  });
}
function outboxInput(rows) {
  if (rows === undefined || rows === null) return [];
  if (!Array.isArray(rows) || rows.length > 100) fail('invalid_outbox');
  return rows.map(row => {
    if (!row || typeof row !== 'object') fail('invalid_outbox');
    const key = assertId(row.key), topic = assertId(row.topic);
    json(row.payload);
    return { key, topic, payload: row.payload };
  });
}

export function openDesignBriefStore({ filename, capabilities, clock = () => Date.now() } = {}) {
  if (typeof filename !== 'string' || !filename || !capabilities || typeof capabilities !== 'object' || typeof clock !== 'function')
    fail('invalid_configuration');
  let db;
  try {
    if (filename !== ':memory:') closeSync(openSync(filename, 'a', 0o600));
    db = new DatabaseSync(filename); db.exec('PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA busy_timeout=5000;');
  }
  catch { try { db?.close(); } catch {} throw new StoreError('storage_unavailable'); }
  try { db.exec(`
    CREATE TABLE IF NOT EXISTS cases (
      tenant_id TEXT NOT NULL, app_id TEXT NOT NULL, case_id TEXT NOT NULL,
      version INTEGER NOT NULL CHECK(version > 0), snapshot_json TEXT NOT NULL, snapshot_hash TEXT NOT NULL,
      created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
      PRIMARY KEY(tenant_id, app_id, case_id)
    ) STRICT;
    CREATE TABLE IF NOT EXISTS records (
      tenant_id TEXT NOT NULL, app_id TEXT NOT NULL, case_id TEXT NOT NULL,
      record_type TEXT NOT NULL, record_id TEXT NOT NULL, value_json TEXT NOT NULL, version INTEGER NOT NULL,
      PRIMARY KEY(tenant_id, app_id, case_id, record_type, record_id),
      FOREIGN KEY(tenant_id, app_id, case_id) REFERENCES cases(tenant_id, app_id, case_id) ON DELETE CASCADE
    ) STRICT;
    CREATE TABLE IF NOT EXISTS proposals (
      tenant_id TEXT NOT NULL, app_id TEXT NOT NULL, case_id TEXT NOT NULL, proposal_id TEXT NOT NULL,
      snapshot_version INTEGER NOT NULL, snapshot_hash TEXT NOT NULL, source_hash TEXT NOT NULL,
      proposal_json TEXT NOT NULL, writes_json TEXT NOT NULL, outbox_json TEXT NOT NULL, proposal_hash TEXT NOT NULL, requester TEXT NOT NULL,
      status TEXT NOT NULL CHECK(status IN ('proposed','approved','committed','stale','rejected')),
      created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
      PRIMARY KEY(tenant_id, app_id, case_id, proposal_id),
      FOREIGN KEY(tenant_id, app_id, case_id) REFERENCES cases(tenant_id, app_id, case_id)
    ) STRICT;
    CREATE TABLE IF NOT EXISTS approvals (
      tenant_id TEXT NOT NULL, app_id TEXT NOT NULL, case_id TEXT NOT NULL, approval_id TEXT NOT NULL,
      proposal_id TEXT NOT NULL, proposal_hash TEXT NOT NULL, snapshot_version INTEGER NOT NULL,
      approver TEXT NOT NULL, approval_hash TEXT NOT NULL, created_at INTEGER NOT NULL,
      PRIMARY KEY(tenant_id, app_id, case_id, approval_id),
      UNIQUE(tenant_id, app_id, case_id, proposal_id),
      FOREIGN KEY(tenant_id, app_id, case_id, proposal_id) REFERENCES proposals(tenant_id, app_id, case_id, proposal_id)
    ) STRICT;
    CREATE TABLE IF NOT EXISTS events (
      seq INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, app_id TEXT NOT NULL, case_id TEXT NOT NULL,
      actor TEXT NOT NULL, operation TEXT NOT NULL, payload_hash TEXT NOT NULL,
      before_version INTEGER, after_version INTEGER, lifecycle TEXT, detail_json TEXT NOT NULL, created_at INTEGER NOT NULL
    ) STRICT;
    CREATE INDEX IF NOT EXISTS events_scope ON events(tenant_id, app_id, case_id, seq);
    CREATE TABLE IF NOT EXISTS idempotency (
      tenant_id TEXT NOT NULL, app_id TEXT NOT NULL, case_id TEXT NOT NULL, operation TEXT NOT NULL, idem_key TEXT NOT NULL,
      payload_hash TEXT NOT NULL, result_json TEXT NOT NULL, created_at INTEGER NOT NULL,
      PRIMARY KEY(tenant_id, app_id, case_id, operation, idem_key)
    ) STRICT;
    CREATE TABLE IF NOT EXISTS outbox (
      id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, app_id TEXT NOT NULL, case_id TEXT NOT NULL,
      entry_key TEXT NOT NULL, topic TEXT NOT NULL, payload_json TEXT NOT NULL,
      status TEXT NOT NULL CHECK(status IN ('pending','leased','delivered','failed','unknown')),
      worker_id TEXT, lease_until INTEGER, attempts INTEGER NOT NULL DEFAULT 0, receipt_json TEXT, failure_code TEXT, reconciliation_json TEXT,
      created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
      UNIQUE(tenant_id, app_id, case_id, entry_key),
      FOREIGN KEY(tenant_id, app_id, case_id) REFERENCES cases(tenant_id, app_id, case_id)
    ) STRICT;
  `); } catch { try { db.close(); } catch {} throw new StoreError('storage_unavailable'); }
  const now = () => { const t = clock(); if (!Number.isSafeInteger(t) || t < 0) fail('invalid_clock'); return t; };
  const transact = fn => {
    db.exec('BEGIN IMMEDIATE');
    try { const value = fn(); db.exec('COMMIT'); return value; }
    catch (e) { try { db.exec('ROLLBACK'); } catch {} throw e instanceof StoreError ? e : new StoreError('storage_failure'); }
  };
  function authorize(principalArg, key, operation) {
    const p = principalOf(principalArg);
    if (p.tenantId !== key.tenantId) fail('not_found');
    const app = capabilities[key.appId];
    const roles = app?.[operation];
    if (!Array.isArray(roles) || !p.roles.some(r => roles.includes(r))) fail('forbidden');
    const requiredScopes = app?.scopes?.[operation] ?? [];
    if (!Array.isArray(requiredScopes) || requiredScopes.some(scope => !p.scopes?.includes(scope))) fail('forbidden');
    return p;
  }
  function idem(p, key, operation, idempotencyKey, payload) {
    assertId(idempotencyKey, 'invalid_idempotency_key');
    const payloadHash = digest({ principalId: p.id, payload });
    const old = db.prepare(`SELECT payload_hash,result_json FROM idempotency WHERE tenant_id=? AND app_id=? AND case_id=? AND operation=? AND idem_key=?`)
      .get(key.tenantId, key.appId, key.caseId, operation, idempotencyKey);
    if (old) { if (old.payload_hash !== payloadHash) fail('idempotency_conflict'); return { replay: parse(old.result_json) }; }
    return { payloadHash };
  }
  function saveIdem(p, key, operation, idempotencyKey, payloadHash, result, time) {
    db.prepare(`INSERT INTO idempotency VALUES (?,?,?,?,?,?,?,?)`).run(key.tenantId,key.appId,key.caseId,operation,idempotencyKey,payloadHash,json(result),time);
  }
  function event(p,key,operation,payloadHash,{beforeVersion=null,afterVersion=null,lifecycle=null,detail={}}={}) {
    db.prepare(`INSERT INTO events(tenant_id,app_id,case_id,actor,operation,payload_hash,before_version,after_version,lifecycle,detail_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)`)
      .run(key.tenantId,key.appId,key.caseId,p.id,operation,payloadHash,beforeVersion,afterVersion,lifecycle,json(detail),now());
  }
  function current(key) {
    const row = db.prepare('SELECT version,snapshot_json,snapshot_hash,created_at,updated_at FROM cases WHERE tenant_id=? AND app_id=? AND case_id=?')
      .get(key.tenantId,key.appId,key.caseId);
    if (!row) return null;
    const records = db.prepare('SELECT record_type,record_id,value_json,version FROM records WHERE tenant_id=? AND app_id=? AND case_id=? ORDER BY record_type,record_id')
      .all(key.tenantId,key.appId,key.caseId);
    const domainRecords = records.map(r => ({ recordType: r.record_type, recordId: r.record_id, value: parse(r.value_json), version: r.version }));
    return { ...parse(row.snapshot_json), tenantId: key.tenantId, appId: key.appId, caseId: key.caseId, version: row.version, snapshotHash: row.snapshot_hash, domainRecords, createdAt: row.created_at, updatedAt: row.updated_at };
  }
  function proposalRow(row) {
    if (!row) fail('not_found');
    return { id: row.proposal_id, snapshotVersion: row.snapshot_version, snapshotHash: row.snapshot_hash,
      sourceHash: row.source_hash, proposal: parse(row.proposal_json), writes: parse(row.writes_json), outbox: parse(row.outbox_json), hash: row.proposal_hash, requester: row.requester,
      status: row.status, createdAt: row.created_at, updatedAt: row.updated_at };
  }

  return {
    createCase(principalArg, keyArg, initialSnapshot, { idempotencyKey } = {}) {
      const key = keyOf(keyArg), p = authorize(principalArg,key,'write');
      const snap = snapshotInput(initialSnapshot), payload = { key, snap };
      return transact(() => {
        const idemCheck=idem(p,key,'createCase',idempotencyKey,payload);if(idemCheck.replay)return idemCheck.replay;
        const exists = db.prepare('SELECT 1 FROM cases WHERE tenant_id=? AND app_id=? AND case_id=?').get(key.tenantId,key.appId,key.caseId);
        if (exists) fail('already_exists');
        const time=now(), version=1, snapshotHash=digest({version,snapshot:snap,domainRecords:[]});
        db.prepare('INSERT INTO cases VALUES(?,?,?,?,?,?,?,?)').run(key.tenantId,key.appId,key.caseId,version,json(snap),snapshotHash,time,time);
        const result=current(key); event(p,key,'case.created',idemCheck.payloadHash,{afterVersion:version,detail:{snapshotHash}});
        saveIdem(p,key,'createCase',idempotencyKey,idemCheck.payloadHash,result,time); return result;
      });
    },
    readCase(principalArg,keyArg) {
      const key=keyOf(keyArg); authorize(principalArg,key,'read'); return current(key);
    },
    listCases(principalArg,{tenantId,appId,after='',limit=DEFAULT_PAGE}={}) {
      const k={tenantId:assertId(tenantId),appId:assertId(appId),caseId:'list'}; const p=authorize(principalArg,k,'read');
      if(after!=='' )assertId(after,'invalid_pagination'); const pg=pageArgs({limit});
      const rows=db.prepare('SELECT case_id,version FROM cases WHERE tenant_id=? AND app_id=? AND case_id>? ORDER BY case_id LIMIT ?').all(k.tenantId,k.appId,after,pg.limit+1);
      const items=rows.slice(0,pg.limit).map(row=>current({...k,caseId:row.case_id}));
      return {items,next:rows.length>pg.limit?items.at(-1)?.caseId:null};
    },
    replaceCase(principalArg,keyArg,expectedVersion,snapshot,{idempotencyKey}={}) {
      const key=keyOf(keyArg),p=authorize(principalArg,key,'write');
      if(!Number.isSafeInteger(expectedVersion)||expectedVersion<1)fail('invalid_version');
      const snap=snapshotInput(snapshot),payload={key,expectedVersion,snap};
      return transact(()=>{
        const ic=idem(p,key,'replaceCase',idempotencyKey,payload);if(ic.replay)return ic.replay;
        const old=current(key);if(!old)fail('not_found');if(old.version!==expectedVersion)fail('stale_version');
        const version=old.version+1,time=now(),domainRecords=db.prepare('SELECT record_type,record_id,value_json,version FROM records WHERE tenant_id=? AND app_id=? AND case_id=? ORDER BY record_type,record_id').all(key.tenantId,key.appId,key.caseId).map(r=>({recordType:r.record_type,recordId:r.record_id,value:parse(r.value_json),version:r.version})),snapshotHash=digest({version,snapshot:snap,domainRecords});
        db.prepare('UPDATE cases SET version=?,snapshot_json=?,snapshot_hash=?,updated_at=? WHERE tenant_id=? AND app_id=? AND case_id=? AND version=?')
          .run(version,json(snap),snapshotHash,time,key.tenantId,key.appId,key.caseId,expectedVersion);
        const staleIds=db.prepare("SELECT proposal_id FROM proposals WHERE tenant_id=? AND app_id=? AND case_id=? AND status IN ('proposed','approved')").all(key.tenantId,key.appId,key.caseId).map(r=>r.proposal_id);
        db.prepare("UPDATE proposals SET status='stale',updated_at=? WHERE tenant_id=? AND app_id=? AND case_id=? AND status IN ('proposed','approved')")
          .run(time,key.tenantId,key.appId,key.caseId);
        for(const proposalId of staleIds) event(p,key,'proposal.stale',digest({proposalId,version}),{beforeVersion:old.version,afterVersion:version,lifecycle:'stale',detail:{proposalId}});
        const result=current(key);event(p,key,'case.replaced',ic.payloadHash,{beforeVersion:old.version,afterVersion:version,detail:{snapshotHash}});
        saveIdem(p,key,'replaceCase',idempotencyKey,ic.payloadHash,result,time);return result;
      });
    },
    saveProposal(principalArg,keyArg,input,{idempotencyKey}={}) {
      const key=keyOf(keyArg),p=authorize(principalArg,key,'propose');
      if(!input||typeof input!=='object')fail('invalid_proposal');
      const proposalId=assertId(input.proposalId),snapshotVersion=input.snapshotVersion;
      if(!Number.isSafeInteger(snapshotVersion)||snapshotVersion<1)fail('invalid_version');
      const snapshotHash=assertHash(input.snapshotHash),sourceHash=assertHash(input.sourceHash),proposal=input.proposal;
      json(proposal);
      const writes=writePacket(input.writes),outbox=outboxInput(input.outbox);
      for(const w of writes) if(!w.value || typeof w.value!=='object' || Array.isArray(w.value) || w.value.id!==w.recordId || w.value.tenantId!==key.tenantId || w.value.version!==(w.expectedVersion??0)+1) fail('invalid_write_packet');
      const body={proposalId,snapshotVersion,snapshotHash,sourceHash,proposal,writes,outbox},proposalHash=digest(body),payload={key,...body};
      return transact(()=>{
        const ic=idem(p,key,'saveProposal',idempotencyKey,payload);if(ic.replay)return ic.replay;
        const snap=current(key);if(!snap)fail('not_found');
        if(snap.version!==snapshotVersion||snap.snapshotHash!==snapshotHash)fail('stale_version');
        const prior=db.prepare('SELECT proposal_id FROM proposals WHERE tenant_id=? AND app_id=? AND case_id=? AND proposal_id=?').get(key.tenantId,key.appId,key.caseId,proposalId);
        if(prior)fail('already_exists');
        for(const w of writes){const row=db.prepare('SELECT version FROM records WHERE tenant_id=? AND app_id=? AND case_id=? AND record_type=? AND record_id=?').get(key.tenantId,key.appId,key.caseId,w.recordType,w.recordId);if((row?.version??null)!==w.expectedVersion)fail('record_version_mismatch');}
        const time=now();db.prepare('INSERT INTO proposals VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)')
          .run(key.tenantId,key.appId,key.caseId,proposalId,snapshotVersion,snapshotHash,sourceHash,json(proposal),json(writes),json(outbox),proposalHash,p.id,'proposed',time,time);
        const row=db.prepare('SELECT * FROM proposals WHERE tenant_id=? AND app_id=? AND case_id=? AND proposal_id=?').get(key.tenantId,key.appId,key.caseId,proposalId);
        const result=proposalRow(row);event(p,key,'proposal.saved',ic.payloadHash,{beforeVersion:snap.version,afterVersion:snap.version,lifecycle:'proposed',detail:{proposalId,proposalHash}});
        saveIdem(p,key,'saveProposal',idempotencyKey,ic.payloadHash,result,time);return result;
      });
    },
    readProposal(principalArg,keyArg,proposalIdArg) {
      const key=keyOf(keyArg);authorize(principalArg,key,'read');const proposalId=assertId(proposalIdArg);
      return proposalRow(db.prepare('SELECT * FROM proposals WHERE tenant_id=? AND app_id=? AND case_id=? AND proposal_id=?').get(key.tenantId,key.appId,key.caseId,proposalId));
    },
    approveProposal(principalArg,keyArg,proposalIdArg,binding,{idempotencyKey}={}) {
      const key=keyOf(keyArg),p=authorize(principalArg,key,'approve'),proposalId=assertId(proposalIdArg);
      if(!binding||typeof binding!=='object')fail('invalid_approval');
      const proposalHash=assertHash(binding.proposalHash),snapshotVersion=binding.snapshotVersion;
      if(!Number.isSafeInteger(snapshotVersion)||snapshotVersion<1)fail('invalid_version');
      const approvalId=assertId(binding.approvalId ?? `approval:${digest([proposalId,proposalHash,p.id]).slice(0,32)}`);
      const body={proposalId,proposalHash,snapshotVersion,approvalId};
      return transact(()=>{
        const ic=idem(p,key,'approveProposal',idempotencyKey,{key,...body});if(ic.replay)return ic.replay;
        const row=db.prepare('SELECT * FROM proposals WHERE tenant_id=? AND app_id=? AND case_id=? AND proposal_id=?').get(key.tenantId,key.appId,key.caseId,proposalId);
        const proposal=proposalRow(row);if(proposal.hash!==proposalHash||proposal.snapshotVersion!==snapshotVersion)fail('binding_mismatch');if(proposal.requester===p.id)fail('separation_of_duties');
        const snap=current(key);if(!snap)fail('not_found');
        if(snap.version!==snapshotVersion||snap.snapshotHash!==proposal.snapshotHash)fail('stale_version');
        if(proposal.status!=='proposed')fail('invalid_lifecycle');
        const time=now(),approvalHash=digest({...body,approver:p.id});
        db.prepare('INSERT INTO approvals VALUES(?,?,?,?,?,?,?,?,?,?)').run(key.tenantId,key.appId,key.caseId,approvalId,proposalId,proposalHash,snapshotVersion,p.id,approvalHash,time);
        db.prepare("UPDATE proposals SET status='approved',updated_at=? WHERE tenant_id=? AND app_id=? AND case_id=? AND proposal_id=? AND status='proposed'").run(time,key.tenantId,key.appId,key.caseId,proposalId);
        const result={id:approvalId,proposalId,proposalHash,snapshotVersion,approver:p.id,hash:approvalHash,createdAt:time};
        event(p,key,'proposal.approved',ic.payloadHash,{beforeVersion:snap.version,afterVersion:snap.version,lifecycle:'approved',detail:{proposalId,approvalId,approvalHash}});
        saveIdem(p,key,'approveProposal',idempotencyKey,ic.payloadHash,result,time);return result;
      });
    },
    commitProposal(principalArg,keyArg,proposalIdArg,binding,{idempotencyKey}={}) {
      const key=keyOf(keyArg),p=authorize(principalArg,key,'commit'),proposalId=assertId(proposalIdArg);
      if(!binding||typeof binding!=='object')fail('invalid_commit');
      const proposalHash=assertHash(binding.proposalHash),approvalId=assertId(binding.approvalId),approvalHash=assertHash(binding.approvalHash),snapshotVersion=binding.snapshotVersion;
      if(!Number.isSafeInteger(snapshotVersion)||snapshotVersion<1)fail('invalid_version');
      const body={proposalId,proposalHash,approvalId,approvalHash,snapshotVersion};
      return transact(()=>{
        const ic=idem(p,key,'commitProposal',idempotencyKey,{key,...body});if(ic.replay)return ic.replay;
        const row=db.prepare('SELECT * FROM proposals WHERE tenant_id=? AND app_id=? AND case_id=? AND proposal_id=?').get(key.tenantId,key.appId,key.caseId,proposalId),proposal=proposalRow(row);
        const approval=db.prepare('SELECT * FROM approvals WHERE tenant_id=? AND app_id=? AND case_id=? AND approval_id=? AND proposal_id=?').get(key.tenantId,key.appId,key.caseId,approvalId,proposalId);
        if(proposal.hash!==proposalHash||proposal.snapshotVersion!==snapshotVersion||!approval||approval.approval_hash!==approvalHash||approval.proposal_hash!==proposalHash||approval.snapshot_version!==snapshotVersion)fail('binding_mismatch');
        const snap=current(key);if(!snap)fail('not_found');
        if(snap.version!==snapshotVersion||snap.snapshotHash!==proposal.snapshotHash)fail('stale_version');
        if(proposal.status!=='approved')fail('invalid_lifecycle');
        const time=now(),newVersion=snap.version+1;
        for(const w of proposal.writes) {if(!w.value || typeof w.value!=='object' || w.value.id!==w.recordId || w.value.tenantId!==key.tenantId || w.value.version!==(w.expectedVersion??0)+1)fail('binding_mismatch');const existing=db.prepare('SELECT version FROM records WHERE tenant_id=? AND app_id=? AND case_id=? AND record_type=? AND record_id=?').get(key.tenantId,key.appId,key.caseId,w.recordType,w.recordId);if((existing?.version??null)!==w.expectedVersion)fail('record_version_mismatch');}
        for(const w of proposal.writes) {
          const recordVersion=(w.expectedVersion??0)+1;
          db.prepare(`INSERT INTO records VALUES(?,?,?,?,?,?,?) ON CONFLICT(tenant_id,app_id,case_id,record_type,record_id) DO UPDATE SET value_json=excluded.value_json,version=excluded.version`)
            .run(key.tenantId,key.appId,key.caseId,w.recordType,w.recordId,json(w.value),recordVersion);
        }
        const snapshotJson=json(snapshotInput({...snap,domainRecords:undefined,version:undefined,snapshotHash:undefined,createdAt:undefined,updatedAt:undefined}));
        const newHash=digest({version:newVersion,snapshot:parse(snapshotJson),domainRecords:[...db.prepare('SELECT record_type,record_id,value_json,version FROM records WHERE tenant_id=? AND app_id=? AND case_id=? ORDER BY record_type,record_id').all(key.tenantId,key.appId,key.caseId).map(r=>({recordType:r.record_type,recordId:r.record_id,value:parse(r.value_json),version:r.version}))]});
        db.prepare('UPDATE cases SET version=?,snapshot_hash=?,updated_at=? WHERE tenant_id=? AND app_id=? AND case_id=? AND version=?')
          .run(newVersion,newHash,time,key.tenantId,key.appId,key.caseId,snap.version);
        const inserted=[];
        for(const entry of proposal.outbox){db.prepare("INSERT INTO outbox(tenant_id,app_id,case_id,entry_key,topic,payload_json,status,created_at,updated_at) VALUES(?,?,?,?,?,?,'pending',?,?)")
          .run(key.tenantId,key.appId,key.caseId,entry.key,entry.topic,json(entry.payload),time,time);inserted.push({key:entry.key,topic:entry.topic});}
        const staleIds=db.prepare("SELECT proposal_id FROM proposals WHERE tenant_id=? AND app_id=? AND case_id=? AND proposal_id<>? AND status IN ('proposed','approved')").all(key.tenantId,key.appId,key.caseId,proposalId).map(r=>r.proposal_id);
        db.prepare("UPDATE proposals SET status='stale',updated_at=? WHERE tenant_id=? AND app_id=? AND case_id=? AND proposal_id<>? AND status IN ('proposed','approved')").run(time,key.tenantId,key.appId,key.caseId,proposalId);
        for(const staleId of staleIds) event(p,key,'proposal.stale',digest({proposalId:staleId,version:newVersion}),{beforeVersion:snap.version,afterVersion:newVersion,lifecycle:'stale',detail:{proposalId:staleId}});
        db.prepare("UPDATE proposals SET status='committed',updated_at=? WHERE tenant_id=? AND app_id=? AND case_id=? AND proposal_id=? AND status='approved'").run(time,key.tenantId,key.appId,key.caseId,proposalId);
        const result={proposalId,status:'committed',version:newVersion,snapshotHash:newHash,writes:proposal.writes.length,outbox:inserted};
        event(p,key,'proposal.committed',ic.payloadHash,{beforeVersion:snap.version,afterVersion:newVersion,lifecycle:'committed',detail:{proposalId,approvalId,proposalHash,outbox:inserted}});
        saveIdem(p,key,'commitProposal',idempotencyKey,ic.payloadHash,result,time);return result;
      });
    },
    rejectProposal(principalArg,keyArg,proposalIdArg,{proposalHash,reasonCode},{idempotencyKey}={}) {
      const key=keyOf(keyArg),p=authorize(principalArg,key,'reject'),proposalId=assertId(proposalIdArg),hash=assertHash(proposalHash),reason=assertId(reasonCode);
      return transact(()=>{
        const ic=idem(p,key,'rejectProposal',idempotencyKey,{key,proposalId,hash,reason});if(ic.replay)return ic.replay;
        const row=db.prepare('SELECT * FROM proposals WHERE tenant_id=? AND app_id=? AND case_id=? AND proposal_id=?').get(key.tenantId,key.appId,key.caseId,proposalId),proposal=proposalRow(row);
        if(proposal.hash!==hash)fail('binding_mismatch');if(!['proposed','approved'].includes(proposal.status))fail('invalid_lifecycle');
        const time=now();db.prepare("UPDATE proposals SET status='rejected',updated_at=? WHERE tenant_id=? AND app_id=? AND case_id=? AND proposal_id=?").run(time,key.tenantId,key.appId,key.caseId,proposalId);
        const result={...proposal,status:'rejected',updatedAt:time};event(p,key,'proposal.rejected',ic.payloadHash,{lifecycle:'rejected',detail:{proposalId,reasonCode:reason}});
        saveIdem(p,key,'rejectProposal',idempotencyKey,ic.payloadHash,result,time);return result;
      });
    },
    listEvents(principalArg,keyArg,{after=0,limit=DEFAULT_PAGE}={}) {
      const key=keyOf(keyArg),p=authorize(principalArg,key,'read'),pg=pageArgs({after,limit});
      const rows=db.prepare('SELECT * FROM events WHERE tenant_id=? AND app_id=? AND case_id=? AND seq>? ORDER BY seq LIMIT ?').all(key.tenantId,key.appId,key.caseId,pg.after,pg.limit+1);
      const items=rows.slice(0,pg.limit).map(r=>({id:r.seq,actor:r.actor,operation:r.operation,payloadHash:r.payload_hash,beforeVersion:r.before_version,afterVersion:r.after_version,lifecycle:r.lifecycle,detail:parse(r.detail_json),createdAt:r.created_at}));
      return {items,next:rows.length>pg.limit?items.at(-1)?.id:null};
    },
    claimOutbox(principalArg,{tenantId,appId,workerId,leaseMs=30_000,limit=10}={}) {
      const k={tenantId:assertId(tenantId),appId:assertId(appId),caseId:'outbox'},p=authorize(principalArg,k,'dispatch');assertId(workerId);
      if(!Number.isSafeInteger(leaseMs)||leaseMs<100||leaseMs>MAX_LEASE_MS)fail('invalid_lease');const pg=pageArgs({limit});
      return transact(()=>{
        const time=now();
        const expired=db.prepare("SELECT id,case_id,entry_key,worker_id FROM outbox WHERE tenant_id=? AND app_id=? AND status='leased' AND lease_until<=? ORDER BY id").all(k.tenantId,k.appId,time);
        db.prepare("UPDATE outbox SET status='unknown',worker_id=NULL,lease_until=NULL,updated_at=? WHERE tenant_id=? AND app_id=? AND status='leased' AND lease_until<=?")
          .run(time,k.tenantId,k.appId,time);
        for(const item of expired) event(p,{tenantId:k.tenantId,appId:k.appId,caseId:item.case_id},'outbox.unknown',digest({id:item.id,key:item.entry_key,previousWorker:item.worker_id}),{detail:{entryId:item.id,entryKey:item.entry_key,reason:'lease_expired'}});
        const rows=db.prepare("SELECT * FROM outbox WHERE tenant_id=? AND app_id=? AND status='pending' ORDER BY id LIMIT ?").all(k.tenantId,k.appId,pg.limit);
        const claim=db.prepare("UPDATE outbox SET status='leased',worker_id=?,lease_until=?,attempts=attempts+1,updated_at=? WHERE id=? AND status='pending'");
        return rows.map(r=>{claim.run(workerId,time+leaseMs,time,r.id);event(p,{tenantId:r.tenant_id,appId:r.app_id,caseId:r.case_id},'outbox.claimed',digest({entryId:r.id,workerId,attempt:r.attempts+1,leaseUntil:time+leaseMs}),{detail:{entryId:r.id,workerId,attempt:r.attempts+1,leaseUntil:time+leaseMs}});return {id:r.id,tenantId:r.tenant_id,appId:r.app_id,caseId:r.case_id,key:r.entry_key,topic:r.topic,payload:parse(r.payload_json),workerId,leaseUntil:time+leaseMs,attempt:r.attempts+1};});
      });
    },
    ackOutbox(principalArg,entryId,workerId,receipt) {
      const p=principalOf(principalArg);if(!Number.isSafeInteger(entryId)||entryId<1)fail('invalid_input');assertId(workerId);json(receipt);
      return transact(()=>{const r=db.prepare('SELECT * FROM outbox WHERE id=?').get(entryId);if(!r||r.tenant_id!==p.tenantId)fail('not_found');authorize(p,{tenantId:r.tenant_id,appId:r.app_id,caseId:r.case_id},'dispatch');
        if(r.status!=='leased'||r.worker_id!==workerId||r.lease_until<=now())fail('lease_lost');const time=now();db.prepare("UPDATE outbox SET status='delivered',receipt_json=?,worker_id=NULL,lease_until=NULL,updated_at=? WHERE id=? AND status='leased'").run(json(receipt),time,entryId);
        event(p,{tenantId:r.tenant_id,appId:r.app_id,caseId:r.case_id},'outbox.delivered',digest(receipt),{detail:{entryId}});return {id:entryId,status:'delivered'};});
    },
    failOutbox(principalArg,entryId,workerId,{code='delivery_failed',ambiguous=false}={}) {
      const p=principalOf(principalArg);if(!Number.isSafeInteger(entryId)||entryId<1)fail('invalid_input');assertId(workerId);assertId(code);
      return transact(()=>{const r=db.prepare('SELECT * FROM outbox WHERE id=?').get(entryId);if(!r||r.tenant_id!==p.tenantId)fail('not_found');authorize(p,{tenantId:r.tenant_id,appId:r.app_id,caseId:r.case_id},'dispatch');
        if(r.status!=='leased'||r.worker_id!==workerId||r.lease_until<=now())fail('lease_lost');const status=ambiguous?'unknown':'failed',time=now();
        db.prepare('UPDATE outbox SET status=?,failure_code=?,worker_id=NULL,lease_until=NULL,updated_at=? WHERE id=? AND status=\'leased\'').run(status,code,time,entryId);
        event(p,{tenantId:r.tenant_id,appId:r.app_id,caseId:r.case_id},`outbox.${status}`,digest({entryId,code}),{detail:{entryId,code}});return {id:entryId,status,code};});
    },
    reconcileOutbox(principalArg,entryId,{outcome,receipt,evidence}={}) {
      const p=principalOf(principalArg);if(!Number.isSafeInteger(entryId)||entryId<1)fail('invalid_input');
      if(!['delivered','not_sent'].includes(outcome))fail('invalid_reconciliation');
      if(outcome==='delivered')json(receipt);
      // This is a manual operator assertion, not cryptographic verification of the
      // remote system. Keep reconciliation authority separate from delivery workers
      // and persist the assertion text alongside the authenticated actor in events.
      if(!evidence||typeof evidence!=='object'||Array.isArray(evidence)||evidence.kind!=='operator_assertion'||
          typeof evidence.reference!=='string'||!evidence.reference.trim()||
          typeof evidence.assertion!=='string'||!evidence.assertion.trim()||evidence.assertion.length>4000)fail('evidence_required');
      json(evidence);
      return transact(()=>{const r=db.prepare('SELECT * FROM outbox WHERE id=?').get(entryId);if(!r||r.tenant_id!==p.tenantId)fail('not_found');authorize(p,{tenantId:r.tenant_id,appId:r.app_id,caseId:r.case_id},'reconcile');
        if(!['unknown','failed'].includes(r.status))fail('invalid_lifecycle');const time=now();
        if(outcome==='delivered')db.prepare("UPDATE outbox SET status='delivered',receipt_json=?,failure_code=NULL,reconciliation_json=?,updated_at=? WHERE id=? AND status IN ('unknown','failed')").run(json(receipt),json(evidence),time,entryId);
        else db.prepare("UPDATE outbox SET status='pending',failure_code=NULL,reconciliation_json=?,updated_at=? WHERE id=? AND status IN ('unknown','failed')").run(json(evidence),time,entryId);
        event(p,{tenantId:r.tenant_id,appId:r.app_id,caseId:r.case_id},'outbox.reconciled',digest({entryId,outcome,receipt:outcome==='delivered'?receipt:null,evidence}),{detail:{entryId,outcome,evidence}});
        return {id:entryId,status:outcome==='delivered'?'delivered':'pending',evidence};});
    },
    readOutbox(principalArg,entryId) {
      const p=principalOf(principalArg);if(!Number.isSafeInteger(entryId)||entryId<1)fail('invalid_input');const r=db.prepare('SELECT * FROM outbox WHERE id=?').get(entryId);
      if(!r||r.tenant_id!==p.tenantId)fail('not_found');authorize(p,{tenantId:r.tenant_id,appId:r.app_id,caseId:r.case_id},'dispatch');return {id:r.id,tenantId:r.tenant_id,appId:r.app_id,caseId:r.case_id,key:r.entry_key,topic:r.topic,payload:parse(r.payload_json),status:r.status,attempts:r.attempts,receipt:r.receipt_json?parse(r.receipt_json):null,failureCode:r.failure_code,reconciliation:r.reconciliation_json?parse(r.reconciliation_json):null,updatedAt:r.updated_at};
    },
    close() { db.close(); },
  };
}
function assertHash(value) { if(typeof value!=='string'||!/^[a-f0-9]{64}$/.test(value))fail('invalid_hash');return value; }
