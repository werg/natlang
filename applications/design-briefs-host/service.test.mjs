import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { loadApplications, nativeExecutor, jsonCopy } from './runtime.mjs';
import { openDesignBriefStore } from './store.mjs';
import { createApplicationService } from './service.mjs';
import { serveApplications } from './server.mjs';
import { dispatchOnce } from './delivery.mjs';
import { scriptedModel } from '../../ts-host/test/support/natlang.mjs';

const op = { id: 'operator', tenantId: 'tenant', roles: ['operator'] };
const reviewer = { id: 'reviewer', tenantId: 'tenant', roles: ['reviewer'] };
const worker = { id: 'delivery', tenantId: 'tenant', roles: ['dispatcher'] };
const capability = { read: ['operator','reviewer'], write: ['operator'],
  'invoke:postLedger': ['operator'], 'invoke:applyAdjustment': ['operator'], 'invoke:inspectPassages': ['operator'], dispatch: ['dispatcher'] };
const capabilities = { '001': capability, '002': capability };
const codeIs = name => error => error.code === name;
const key = { tenantId: 'tenant', appId: '001', caseId: 'engineering-case' };
const source = (args, returns, body) => `---\nargs: { ${args} }\nreturns: ${returns}\n---\n${body}\n`;

async function environment(t, overrides = {}) {
  const root = await mkdtemp(join(process.cwd(), '.test-native-interface-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const a = join(root, '001-ledger'), b = join(root, '002-search');
  await mkdir(a); await mkdir(b); await mkdir(join(a, 'postLedger'));
  const bundles = [
    [a, { schema:'natlang.application/1',id:'001',title:'Engineering ledger',entrypoint:'postLedger.nl',
      interfaces:[{name:'postLedger',path:'postLedger.nl',purpose:'Exercise a carried ledger state.',stateParameter:'ledger',stateResultPath:['next'],stateKey:'ledger'}] }, {
      'postLedger.nl': source('ledger: { balance: number }, delta: number','{ next: { balance: number }; posting: number }',
        'Call calculatePosting(delta), add its actual result to ledger.balance and return the next state.'),
      'postLedger/calculatePosting.nl': source('delta: number','number','Return the supplied delta as the engineering posting.'),
    }],
    [b, { schema:'natlang.application/1',id:'002',title:'Engineering text search',entrypoint:'inspectPassages.nl',
      interfaces:[{name:'inspectPassages',path:'inspectPassages.nl',purpose:'Exercise unrelated typed array inputs.'}] }, {
      'inspectPassages.nl': source('passages: string[], search: string','string[]','Select passages containing search exactly.'),
    }],
  ];
  overrides.configure?.(bundles);
  for (const [dir, manifest, files] of bundles) {
    manifest.source_files = Object.keys(files);
    await writeFile(join(dir,'application.json'), JSON.stringify(manifest));
    await writeFile(join(dir,'brief.json'), JSON.stringify({application_id:manifest.id,title:manifest.title}));
    for (const [path, text] of Object.entries(files)) await writeFile(join(dir,path), text);
  }
  const catalog = await loadApplications(root);
  const scripted = scriptedModel(opening => opening.includes('engineering posting') ? 'return delta;' :
    opening.includes('Select passages containing') ? 'return passages.filter(p => p.includes(search));' :
    opening.includes('Carry nested checkpoint') ? 'return { updated: { balance: dossier.checkpoint.balance + dossier.delta }, accepted: true };' :
    'const posting = await calculatePosting(delta); return { next: { balance: ledger.balance + posting }, posting };');
  const execute = overrides.execute ?? nativeExecutor({ driver: scripted.driver });
  const filename = join(root,'state.sqlite');
  let store = openDesignBriefStore({filename,capabilities,clock:overrides.clock});
  t.after(() => { try { store.close(); } catch {} });
  const service = createApplicationService({catalog,store,execute,capabilities,...overrides});
  return { root, filename, catalog, store, service, scripted, execute,
    reopen() { store.close(); store = openDesignBriefStore({filename,capabilities,clock:overrides.clock}); return {
      store, service:createApplicationService({catalog,store,execute,capabilities,...overrides}) }; } };
}
function create(service) {
  return service.create(op,'001',key.caseId,{interface:'postLedger',inputs:{ledger:{balance:7},delta:3},idempotencyKey:'create'});
}
test('real native runtime preserves unrelated per-program APIs and nested child results', async t => {
  const e=await environment(t);create(e.service);
  const p=await e.service.invoke(op,'001',key.caseId,{interface:'postLedger',expectedVersion:1,idempotencyKey:'post-1'});
  assert.deepEqual(p.invocation.result,{next:{balance:10},posting:3});
  assert.equal(e.scripted.openings.length,2,'the root and its native child must actually execute');
  assert.equal(e.store.readCase(op,key).domainRecords[0].value.state.balance,10,'the checked checkpoint is durably available to the next turn');
  e.service.create(op,'002','other',{interface:'inspectPassages',inputs:{passages:['alpha','beta','alphabet'],search:'alpha'},idempotencyKey:'text-create'});
  const q=await e.service.invoke(op,'002','other',{interface:'inspectPassages',expectedVersion:1,idempotencyKey:'text-call'});
  assert.deepEqual(q.invocation.result,['alpha','alphabet']);
  assert.deepEqual(e.service.describe(op,'001').interfaces[0].parameters.map(x=>x.name),['ledger','delta']);
  assert.deepEqual(e.service.describe(op,'002').interfaces[0].parameters.map(x=>x.name),['passages','search']);
});

test('native checkpoint resumes after restart; exact invocation replay avoids new inference', async t => {
  const e=await environment(t);create(e.service);
  const p=await e.service.invoke(op,'001',key.caseId,{interface:'postLedger',expectedVersion:1,idempotencyKey:'first'});
  const before=e.scripted.openings.length;
  const reopened=e.reopen();
  assert.equal((await reopened.service.invoke(op,'001',key.caseId,{interface:'postLedger',expectedVersion:1,idempotencyKey:'first'})).hash,p.hash);
  assert.equal(e.scripted.openings.length,before);
  const resumed=await reopened.service.invoke(op,'001',key.caseId,{interface:'postLedger',expectedVersion:2,idempotencyKey:'saved-source-resume'});
  assert.deepEqual(resumed.invocation.inputs,{delta:3,ledger:{balance:10}});
  assert.deepEqual(resumed.invocation.result,{next:{balance:13},posting:3});
  const q=await reopened.service.invoke(op,'001',key.caseId,{interface:'postLedger',inputs:{delta:4},expectedVersion:3,idempotencyKey:'second'});
  assert.deepEqual(q.invocation.inputs,{delta:4,ledger:{balance:13}});
  assert.deepEqual(q.invocation.result,{next:{balance:17},posting:4});
  await assert.rejects(reopened.service.invoke(op,'001',key.caseId,{interface:'postLedger',inputs:{ledger:{balance:0},delta:4},expectedVersion:4,idempotencyKey:'override'}),codeIs('state_override'));
  await assert.rejects(reopened.service.invoke(op,'001',key.caseId,{interface:'postLedger',inputs:{delta:5},expectedVersion:3,idempotencyKey:'second'}),codeIs('idempotency_conflict'));
});

test('source refresh during inference rejects the stale result without committing or dispatching', async t => {
  let release, started;const ready=new Promise(resolve=>{started=resolve;});
  const e=await environment(t,{execute:async()=>{started();return await new Promise(resolve=>{release=resolve;});}});create(e.service);
  const pending=e.service.invoke(op,'001',key.caseId,{interface:'postLedger',expectedVersion:1,idempotencyKey:'slow'});
  await ready;
  e.service.replace(op,'001',key.caseId,{interface:'postLedger',inputs:{ledger:{balance:9},delta:3},expectedVersion:1,idempotencyKey:'refresh'});
  release({next:{balance:10},posting:3});await assert.rejects(pending,codeIs('stale_version'));
  assert.equal(e.store.readCase(op,key).domainRecords.length,0);
});

test('invalid native data and wrong operation authority cannot mutate state', async t => {
  const e=await environment(t,{execute:async()=>({next:{balance:'invalid'},posting:3})});create(e.service);
  await assert.rejects(e.service.invoke(op,'001',key.caseId,{interface:'postLedger',inputs:null,expectedVersion:1,idempotencyKey:'null'}),codeIs('invalid_arguments'));
  await assert.rejects(e.service.invoke(op,'001',key.caseId,{interface:'postLedger',expectedVersion:1,idempotencyKey:'invalid'}),codeIs('invalid_result'));
  await assert.rejects(e.service.invoke(reviewer,'001',key.caseId,{interface:'postLedger',expectedVersion:1,idempotencyKey:'forbidden'}),codeIs('forbidden'));
  assert.throws(()=>e.service.create(op,'002','wrong',{interface:'inspectPassages',inputs:{ledger:{}},idempotencyKey:'wrong'}),codeIs('unknown_argument'));
  assert.deepEqual(e.store.listEvents(op,key).items.map(x=>x.operation),['case.created']);
});

test('external effects are atomically queued by an exact adapter and require a matching receiver receipt', async t => {
  const e=await environment(t,{effectAdapters:{'001':{postLedger:({result})=>[{key:'posting-1',topic:'ledger.posted',payload:{posting:result.posting}}]}}});create(e.service);
  const p=await e.service.invoke(op,'001',key.caseId,{interface:'postLedger',expectedVersion:1,idempotencyKey:'effect'});
  assert.equal(p.outbox.length,1);
  let calls=0;
  const results=await dispatchOnce({store:e.store,principal:worker,appId:'001',workerId:'worker-1',routes:{'001':{'ledger.posted':{endpoint:'https://receiver.invalid/receive'}}},
    fetchImpl:async(url,options)=>{calls++;assert.equal(options.redirect,'error');assert.match(options.headers['idempotency-key'],/tenant:001:engineering-case:posting-1/);return {ok:true,json:async()=>({status:'delivered',idempotencyKey:'posting-1',receiptId:'receipt-1'})};}});
  assert.equal(calls,1);assert.equal(results[0].status,'delivered');
  assert.equal(e.store.readOutbox(worker,1).receipt.receiptId,'receipt-1');
});

test('delivery validates configuration before claiming and leases each slow item at its actual send time', async t => {
  let time = 1_000_000;
  const e=await environment(t,{clock:()=>time,effectAdapters:{'001':{postLedger:()=>[
    {key:'one',topic:'ledger.posted',payload:{n:1}}, {key:'two',topic:'ledger.posted',payload:{n:2}},
    {key:'three',topic:'ledger.posted',payload:{n:3}},
  ]}}});create(e.service);
  const p=await e.service.invoke(op,'001',key.caseId,{interface:'postLedger',expectedVersion:1,idempotencyKey:'slow-deliveries'});
  await assert.rejects(dispatchOnce({store:e.store,principal:worker,appId:'001',workerId:'w',
    routes:{'001':{'ledger.posted':{endpoint:'not a URL'}}}}),codeIs('invalid_delivery_route'));
  assert.equal(e.store.readOutbox(worker,1).attempts,0);
  const result=await dispatchOnce({store:e.store,principal:worker,appId:'001',workerId:'w',timeoutMs:900,leaseMs:1000,limit:3,
    routes:{'001':{'ledger.posted':{endpoint:'https://receiver.invalid/delivery'}}},fetchImpl:async(_url,options)=>{
      time+=700;const body=JSON.parse(options.body);
      return {ok:true,json:async()=>({status:'delivered',idempotencyKey:body.idempotencyKey,receiptId:'ack-'+body.idempotencyKey})};
    }});
  assert.deepEqual(result.map(row=>row.status),['delivered','delivered','delivered']);
  for(const id of [1,2,3])assert.equal(e.store.readOutbox(worker,id).attempts,1);
});

test('HTTP binds the authenticated tenant and exposes no inference or configuration without a credential', async t => {
  const e=await environment(t);
  const token='engineering-test-token', tokenHash=createHash('sha256').update(token).digest('hex');
  const server=await serveApplications({service:e.service,credentials:[{tokenHash,principal:op}]});t.after(()=>server.close());
  assert.equal((await fetch(server.url+'/applications')).status,401);
  const headers={authorization:`Bearer ${token}`,'content-type':'application/json'};
  const described=await (await fetch(server.url+'/applications/002',{headers})).json();
  assert.equal(described.interfaces[0].name,'inspectPassages');
  const created=await fetch(server.url+'/applications/001/cases/engineering-case',{method:'POST',headers,
    body:JSON.stringify({interface:'postLedger',inputs:{ledger:{balance:7},delta:3},idempotencyKey:'http-create',tenantId:'attacker'})});
  assert.equal(created.status,201);assert.equal((await created.json()).tenantId,'tenant');
  const invalid=await fetch(server.url+'/applications/001/cases/bad',{method:'POST',headers,body:'{'});
  assert.equal(invalid.status,400);assert.deepEqual(await invalid.json(),{error:'invalid_json'});
});

test('different operations restore the same nested domain checkpoint without replacing other dossier inputs', async t => {
  const e=await environment(t,{configure:bundles=>{
    const [,manifest,files]=bundles[0];
    manifest.interfaces.push({name:'applyAdjustment',path:'applyAdjustment.nl',purpose:'Carry a checkpoint through an unrelated dossier field.',
      stateParameter:'dossier',stateInputPath:['checkpoint'],stateResultPath:['updated'],stateKey:'ledger'});
    files['applyAdjustment.nl']=source('dossier: { checkpoint?: { balance: number }; delta: number }',
      '{ updated: { balance: number }; accepted: boolean }','Carry nested checkpoint and add the supplied dossier delta.');
  }});
  create(e.service);
  await e.service.invoke(op,'001',key.caseId,{interface:'postLedger',expectedVersion:1,idempotencyKey:'initial'});
  const result=await e.service.invoke(op,'001',key.caseId,{interface:'applyAdjustment',inputs:{dossier:{delta:5}},
    expectedVersion:2,idempotencyKey:'adjust'});
  assert.deepEqual(result.invocation.inputs,{dossier:{checkpoint:{balance:10},delta:5}});
  assert.deepEqual(result.invocation.result,{updated:{balance:15},accepted:true});
  const resumed=await e.service.invoke(op,'001',key.caseId,{interface:'postLedger',inputs:{delta:2},expectedVersion:3,idempotencyKey:'back'});
  assert.equal(resumed.invocation.result.next.balance,17);
  await assert.rejects(e.service.invoke(op,'001',key.caseId,{interface:'applyAdjustment',
    inputs:{dossier:{checkpoint:{balance:0},delta:2}},expectedVersion:4,idempotencyKey:'overwritten'}),codeIs('state_override'));
});

test('a declared pending effect cannot be silently discarded when its application adapter is missing', async t => {
  const e=await environment(t,{configure:bundles=>{
    bundles[0][1].interfaces[0].effectsPath=['intentions'];
    bundles[0][2]['postLedger.nl']=source('ledger: { balance: number }, delta: number',
      '{ next: { balance: number }; posting: number; intentions: string[] }','Produce the typed engineering intention.');
  },execute:async()=>({next:{balance:10},posting:3,intentions:['deliver-this']})});
  create(e.service);
  await assert.rejects(e.service.invoke(op,'001',key.caseId,{interface:'postLedger',expectedVersion:1,idempotencyKey:'effect-missing'}),codeIs('effect_adapter_required'));
  assert.equal(e.store.readCase(op,key).version,1);
  assert.equal(e.store.readCase(op,key).domainRecords.length,0);
});

test('cancellation after inference or adapter completion cannot commit a partial invocation', async t => {
  const controller=new AbortController();
  const e=await environment(t,{execute:async()=>{controller.abort();return {next:{balance:10},posting:3};}});
  create(e.service);
  await assert.rejects(e.service.invoke(op,'001',key.caseId,{interface:'postLedger',expectedVersion:1,idempotencyKey:'cancelled'},
    {signal:controller.signal}),codeIs('invocation_cancelled'));
  assert.equal(e.store.readCase(op,key).version,1);
  assert.deepEqual(e.store.listEvents(op,key).items.map(e=>e.operation),['case.created']);
});

test('JSON custody rejects lossy values and adapters cannot modify the checked result', async t => {
  for (const bad of [undefined,NaN,Infinity,{x:undefined},[undefined],new Date()]) assert.throws(()=>jsonCopy(bad),codeIs('invalid_json'));
  const cycle={};cycle.self=cycle;assert.throws(()=>jsonCopy(cycle),codeIs('invalid_json'));
  const e=await environment(t,{effectAdapters:{'001':{postLedger:({result})=>{result.next.balance=0;return [];}}}});
  create(e.service);
  await assert.rejects(e.service.invoke(op,'001',key.caseId,{interface:'postLedger',expectedVersion:1,idempotencyKey:'mutation'}),TypeError);
  assert.equal(e.store.readCase(op,key).version,1);
});
