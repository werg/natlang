#!/usr/bin/env node
/**
 * Engineering pilot for the order workflow. `controlled` exercises the real
 * application decision table and durable service against local simulated
 * receipts. `model` is an explicitly separate selected-checkpoint route.
 * Neither mode connects to payment, inventory, shipping, or messaging systems.
 */
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const repoRoot=fileURLToPath(new URL('../..',import.meta.url));

let workflowModule;
async function workflowApp() {
  if(workflowModule) return workflowModule;
  const {buildProject,formatDiagnostics}=await import('../dist/index.js');
  const {resolve}=await import('node:path');
  const {pathToFileURL}=await import('node:url');
  const root=resolve(repoRoot,'applications/workflow'),runtime=resolve(repoRoot,'ts-host/dist/index.js');
  const built=buildProject({project:root,outDir:resolve(repoRoot,'applications/dist/workflow'),
    runtimeModule:{url:pathToFileURL(runtime).href,path:runtime,
      types:resolve(process.cwd(),'ts-host/dist/index.d.ts'),specifiers:['@natlang/node']}});
  if(!built.ok) throw new Error(formatDiagnostics(built.diagnostics));
  workflowModule=await import(new URL('../../applications/dist/workflow/index.js',import.meta.url));
  return workflowModule;
}

const flags = new Map();
for (let i=2;i<process.argv.length;i+=2) flags.set(process.argv[i],process.argv[i+1]);
const mode=flags.get('--mode')??'controlled';
if(!['controlled','reference','model'].includes(mode)) throw new TypeError('--mode must be controlled, reference, or model');

/** Use the production app API with a filesystem-backed local remote simulator. */
export async function runWorkflowPilot({root=undefined,origin=mode,policy='crisp',run=async()=>null,
  seed=17,checkpointIdentity=null}={}) {
  const {WorkflowService,stepFull}=await workflowApp();
  const temporary=root?null:await mkdtemp(join(repoRoot,'.workflow-pilot-'));
  const service=new WorkflowService(root??temporary);
  const journal=[];
  const record=async(order,event)=>{
    const result=await stepFull(service,order,event,{run,policy});
    journal.push({order,event,action:result.decision.action,reason:result.decision.reason,
      revision:result.state.revision,phase:result.state.phase,pending:result.state.pending});
    return result;
  };
  try {
    const journeys={};
    // Ordinary success: advance through the actual production decision table.
    await service.open('ordinary-success',1250);
    for(let n=0;n<4;n++) await record('ordinary-success',{kind:'continue'});
    journeys.success=await service.read('ordinary-success');
    // Definite charge rejection followed by bounded compensation.
    await service.open('rejected-operation',1250);
    await record('rejected-operation',{kind:'continue'});
    await record('rejected-operation',{kind:'continue',fault:'definite_failure'});
    await record('rejected-operation',{kind:'continue'});
    journeys.rejection=await service.read('rejected-operation');
    // Remote completes, acknowledgement is lost; a fresh service instance reconciles receipt.
    await service.open('lost-ack',1250);
    await record('lost-ack',{kind:'continue',fault:'lost_ack'});
    const restarted=new WorkflowService(root??temporary);
    const recovered=await stepFull(restarted,'lost-ack',{kind:'reconcile'}, {run,policy});
    journal.push({order:'lost-ack',event:{kind:'reconcile'},action:recovered.decision.action,
      revision:recovered.state.revision,phase:recovered.state.phase,pending:recovered.state.pending});
    journeys.lostAck=await restarted.read('lost-ack');
    // A request that never reached the remote remains pending across process restart.
    await service.open('pending-restart',1250);
    await record('pending-restart',{kind:'continue',fault:'lost_request'});
    const resumed=new WorkflowService(root??temporary);
    const pending=await stepFull(resumed,'pending-restart',{kind:'reconcile'}, {run,policy});
    journal.push({order:'pending-restart',event:{kind:'reconcile'},action:pending.decision.action,
      revision:pending.state.revision,phase:pending.state.phase,pending:pending.state.pending});
    journeys.pendingRestart=await resumed.read('pending-restart');
    // Cancellation after reservation compensates by releasing stock.
    await service.open('cancel',1250);
    await record('cancel',{kind:'continue'});
    await record('cancel',{kind:'cancel'});
    journeys.cancel=await service.read('cancel');
    // Cancellation after payment must refund before releasing reserved stock.
    await service.open('refund-release',1250);
    await record('refund-release',{kind:'continue'});
    await record('refund-release',{kind:'continue'});
    await record('refund-release',{kind:'cancel'});
    await record('refund-release',{kind:'cancel'});
    journeys.refundRelease=await service.read('refund-release');
    // Repeated retained event exercises service idempotence / settled wait.
    await service.open('repeated',1250);
    for(let n=0;n<5;n++) await record('repeated',{kind:'continue'});
    journeys.repeated=await service.read('repeated');
    const effects=await service.remoteEffects(),serviceEvents=service.drainEvents();
    if(temporary) await rm(temporary,{recursive:true,force:true});
    return {schema:'natlang-workflow-pilot/1',origin,checkpointIdentity,seed,policy,
      optimizerSteps:0,learnedSuccess:null,
      journeys,journal,trace:{orderedDecisions:journal,serviceEvents},effects,
      cleanup:{temporaryDirectoryRemoved:!!temporary}};
  } finally { if(temporary) await rm(temporary,{recursive:true,force:true}); }
}

/** Run the application's existing crisp policy source as a compiled selected guest module. */
export async function runSelectedPolicyOrder({select,origin='model',seed=17,checkpointIdentity=null,
  root=undefined,policySource=undefined}={}) {
  if(typeof select!=='function') throw new TypeError('Selected policy requires a semantic selector');
  const base=repoRoot;
  const crispSource=await readFile(join(base,'applications/workflow/handle/crisp.ts'),'utf8');
  if(policySource!==undefined&&policySource!==crispSource)
    throw new Error('Selected policy source differs from the application crisp.ts source');
  const {WorkflowService}=await workflowApp();
  const {createNatlangSelectedEnvironment}=await import('tracing-js-self/natlang-selected');
  const {installSelectedSemanticEnvironment,setDefaultEnvironmentFactory}=await import('../dist/runtime/node.js');
  const {moduleInstanceAsync,setModuleRealm}=await import('../dist/runtime/modules.js');
  const {parseModule,PATH_ONLY}=await import('../dist/runtime/loader.js');
  const {TypeScriptEnvironment}=await import('../dist/environment.js');
  const temporary=root?null:await mkdtemp(join(repoRoot,'.workflow-selected-'));
  const policyRecord=parseModule('/__natlang__/folder/crisp.ts',policySource??crispSource,{},PATH_ONLY);
  const prefixes=[];
  const factory=mode=>createNatlangSelectedEnvironment({mode,select:async prefix=>{
      const lineage={entry:prefix.entry,invocationId:prefix.invocationId,trackId:prefix.trackId,
        programId:prefix.programId,sourceDigest:prefix.sourceDigest,committed:prefix.committed.length};
      try {
        const selected=await select(prefix);
        prefixes.push({...lineage,selectedRecord:typeof selected==='string'?selected:selected?.record??null,
          selection:selected?.selection??null});
        return selected;
      } catch(error) {
        prefixes.push({...lineage,selectedRecord:null,failure:{category:error?.category??error?.name??'Error',
          message:String(error?.message??error),evidence:error?.evidence??null}});
        throw error;
      }
    }});
  let selectedEnvironment;
  try {
    let service=new WorkflowService(root??temporary);const journeys={},journal=[],serviceEventTrace=[];
    const limits={transientRetries:2,checksBeforeRetry:2,recheckMs:30000};
    let fatalModelFailure=null;
    const runStep=async(order,event)=>{
      const before=await service.read(order);
      const snapshot={state:before,event,receipt:before.pending?await service.receipt(before.pending):null};
      try {
        selectedEnvironment=factory('fresh');
        installSelectedSemanticEnvironment(mode=>mode==='fresh'?selectedEnvironment:factory(mode));
        const module=await moduleInstanceAsync(policyRecord,{crisp:policyRecord});
        const decision=await module.default(snapshot,limits);
        if(origin==='controlled') {
          const latest=prefixes.at(-1);
          const expected=select.referenceFor?.(latest?.programId,latest?.invocationId);
          if(!expected||JSON.stringify(decision)!==JSON.stringify(expected))
            throw new Error('Controlled selected projection differs from the production application reference');
        }
        const state=await service.apply(order,before.revision,event,decision);
        journal.push({order,event,decision:JSON.parse(JSON.stringify(decision)),state});return state;
      } catch(error) {
        const failure={name:error?.name??'Error',category:error?.category??null,
          message:String(error?.message??error),evidence:error?.evidence??null,events:error?.events??[]};
        if(origin==='model') fatalModelFailure=failure;
        journal.push({order,event,failure,state:before});throw Object.assign(new Error(failure.message),{failure});
      } finally {
        selectedEnvironment?.close();selectedEnvironment=undefined;
        setDefaultEnvironmentFactory(options=>new TypeScriptEnvironment({workspace:options.workspace,network:options.network}));
        setModuleRealm(()=>new TypeScriptEnvironment({mode:'retained'}));
      }
    };
    const journey=async(id,events)=>{
      await service.open(id,1250);
      if(origin==='model'&&fatalModelFailure) {
        journeys[id]={outcome:'not-run-after-model-rejection',state:await service.read(id)};return;
      }
      try { for(const event of events) await runStep(id,event); }
      catch(error) { journeys[id]={outcome:'rejected',failure:error.failure??{message:error.message},state:await service.read(id)};
        if(origin==='model') fatalModelFailure=error.failure??{message:error.message};return; }
      journeys[id]={outcome:'complete',state:await service.read(id)};
    };
    await journey('ordinary-success',Array.from({length:4},()=>({kind:'continue'})));
    await journey('rejected-operation',[{kind:'continue'},{kind:'continue',fault:'definite_failure'},{kind:'continue'}]);
    await service.open('lost-ack',1250);
    try { if(!fatalModelFailure) await runStep('lost-ack',{kind:'continue',fault:'lost_ack'}); }
    catch(error) { journeys['lost-ack']={outcome:'rejected',failure:error.failure??{message:error.message},state:await service.read('lost-ack')};fatalModelFailure=origin==='model'?error.failure??{message:error.message}:fatalModelFailure; }
    if(fatalModelFailure&&!journeys['lost-ack'])journeys['lost-ack']={outcome:'not-run-after-model-rejection',state:await service.read('lost-ack')};
    serviceEventTrace.push(...service.drainEvents());
    service=new WorkflowService(root??temporary);
    if(!journeys['lost-ack']) try { await runStep('lost-ack',{kind:'reconcile'});journeys['lost-ack']={outcome:'complete',state:await service.read('lost-ack')}; }
    catch(error) { journeys['lost-ack']={outcome:'rejected',failure:error.failure??{message:error.message},state:await service.read('lost-ack')}; }
    await service.open('pending-restart',1250);
    try { if(!fatalModelFailure) await runStep('pending-restart',{kind:'continue',fault:'lost_request'}); }
    catch(error) { journeys['pending-restart']={outcome:'rejected',failure:error.failure??{message:error.message},state:await service.read('pending-restart')};fatalModelFailure=origin==='model'?error.failure??{message:error.message}:fatalModelFailure; }
    if(fatalModelFailure&&!journeys['pending-restart'])journeys['pending-restart']={outcome:'not-run-after-model-rejection',state:await service.read('pending-restart')};
    serviceEventTrace.push(...service.drainEvents());
    service=new WorkflowService(root??temporary);
    if(!journeys['pending-restart']) try { await runStep('pending-restart',{kind:'reconcile'});journeys['pending-restart']={outcome:'complete',state:await service.read('pending-restart')}; }
    catch(error) { journeys['pending-restart']={outcome:'rejected',failure:error.failure??{message:error.message},state:await service.read('pending-restart')}; }
    await journey('cancel',[{kind:'continue'},{kind:'cancel'}]);
    await journey('refund-release',[{kind:'continue'},{kind:'continue'},{kind:'cancel'},{kind:'cancel'}]);
    await journey('repeated',Array.from({length:5},()=>({kind:'continue'})));
    const completed=Object.values(journeys).every(row=>row.outcome==='complete');
    const report={schema:'natlang-workflow-selected-order/1',origin,seed,checkpointIdentity,
      optimizerSteps:0,learnedSuccess:null,
      outcome:completed?'applied':'rejected',journeys,journal,effects:await service.remoteEffects(),
      serviceEvents:[...serviceEventTrace,...service.drainEvents()],selectedTrace:prefixes,
      controlledProjections:select.projections??[],cleanup:{temporaryDirectoryRemoved:!!temporary}};
    // Selected values are scoped proxies. Snapshot them before closing the
    // environment so the durable report never retains a live guest resource.
    return JSON.parse(JSON.stringify(report));
  } finally {
    if(temporary) await rm(temporary,{recursive:true,force:true});
    setDefaultEnvironmentFactory(options=>new TypeScriptEnvironment({workspace:options.workspace,network:options.network}));
    setModuleRealm(()=>new TypeScriptEnvironment({mode:'retained'}));
  }
}

/** Build controlled return-value projections from the existing crisp policy.
 * These are explicit engineering records, not complete source traces. */
export function controlledSourceProjectionSelector() {
  const cache=new Map(),projections=[];
  const selector=async prefix=>{
    const index=prefix.committed.filter(item=>item.kind==='semantic').length;
    if(prefix.entry==='initialize') {
      const {encodeWitness}=await import('tracing-js-self/training');
      return [
        encodeWitness('allocate',{name:'o99',kind:'closure',code:'crisp',properties:[],captures:[]}),
        encodeWitness('write',{object:'o0',key:'default',value:['ref','o99'],present:true}),
        encodeWitness('return',{value:['ref','o0']}),
        encodeWitness('end',{kind:'return'})
      ][index];
    }
    const cacheKey=`${prefix.programId}:${prefix.invocationId}`;
    let certified=cache.get(cacheKey);
    if(!certified) {
      const heap=new Map(prefix.selectedState.heap),locals=new Map(prefix.selectedState.locals),aliases=new Map(prefix.selectedState.captureAliases);
      const args=['snapshot','limits'].map(name=>selectedValue(locals.get(aliases.get(name)??name),heap));
      const source=await ownedPolicySource();
      const {executeWithNativeServices}=await import('tracing-js-self');
      const {compileEffectfulWitness,encodeWitness}=await import('tracing-js-self/training');
      let result,expected,verification='owned-tracer',ownedTracerBoundaryFailure=null;
      const nativeModule=await import(`data:text/javascript;base64,${Buffer.from(`${source}\nexport { crisp };`).toString('base64')}`);
      const nativeExpected=nativeModule.crisp(...args);
      try {
        const run=await executeWithNativeServices({entry:{text:source,functionName:'crisp',args}});
        result=compileEffectfulWitness(run.bundle,{sessionId:'workflow-pilot',programId:prefix.programId,
          invocationId:prefix.invocationId,trackId:prefix.trackId,realmId:prefix.realmId,triggerId:'workflow-pilot'});
        expected=selectedGraphValue(result.outcome.graph.roots.at(-1),result.outcome.graph.objects);
      } catch(error) {
        // The owned tracer deliberately has a narrower host-runtime surface
        // than this application. For unsupported native operations, invoke
        // the exact compiled source in an isolated V8 module; never invent an
        // action. The selected guest still has to return these source-derived
        // bytes and the production service validates/applies the decision.
        if(!/Unmodeled ordinary \[\[Get\]\] or exotic property/.test(String(error?.message??error))) throw error;
        expected=nativeExpected;verification='compiled-source-v8';
        ownedTracerBoundaryFailure=String(error?.message??error);
      }
      if(JSON.stringify(expected)!==JSON.stringify(nativeExpected))
        throw new Error('Owned tracer and independent crisp execution disagree');
      const objectIds=new Set(prefix.selectedState.heap.map(([id])=>id));
      let outputId=0;while(objectIds.has(`o${outputId}`))outputId++;
      const toWitness=value=>value===undefined?['undefined']:value===null?['null']:
        typeof value==='string'?['str',value]:typeof value==='number'?['num',value]:
        typeof value==='boolean'?['bool',value]:(()=>{throw new Error(`Decision contains unsupported ${typeof value}`)})();
      const projection=[
        encodeWitness('allocate',{name:`o${outputId}`,kind:'object',properties:Object.entries(expected).map(([key,value])=>({
          key,value:toWitness(value)
        }))}),
        encodeWitness('return',{value:['ref',`o${outputId}`],invocation:prefix.invocationId}),
        encodeWitness('end',{kind:'return'})
      ];
      const records=projection;
      const entrySourceDigest=typeof prefix.source==='string'?
        (await import('node:crypto')).createHash('sha256').update(prefix.source).digest('hex'):null;
      if(!entrySourceDigest||entrySourceDigest!==prefix.sourceDigest)
        throw new Error('Selected entry source bytes do not match the declared source digest');
      const {createHash}=await import('node:crypto');
      const originalCrisp=await readFile(join(repoRoot,'applications/workflow/handle/crisp.ts'));
      const sha=value=>createHash('sha256').update(value).digest('hex');
      const projectionDefinition='selected entry crisp(snapshot,limits) returns source-derived own enumerable primitive decision properties; controlled bootstrap maps module exports.default to closure crisp; no decision is authored by the fixture';
      const snapshotLimitsDigest=sha(JSON.stringify(args));
      certified={records,expected,sourceDigest:prefix.sourceDigest};cache.set(cacheKey,certified);
      projections.push({invocationId:prefix.invocationId,entry:prefix.entry,programId:prefix.programId,
        trackId:prefix.trackId,realmId:prefix.realmId,sourceDigest:prefix.sourceDigest,
        applicationSourceDigest:result?.exported.sidecar.compiler.sourceDigest??null,
        compiledSelectedSourceDigest:prefix.sourceDigest,
        trustedPolicyBinding:'exact application crisp.ts bytes enforced before selected compilation',
        observedEntrySourceDigest:entrySourceDigest,
        originalCrispSourceDigest:sha(originalCrisp),
        projectionDefinitionDigest:sha(projectionDefinition),
        snapshotLimitsDigest,
        independentExpected:nativeExpected,
        ownedInputRoots:result?.outcome.graph.roots.slice(0,2)??[],
        ownedRootProps:result?.outcome.graph.roots.slice(0,2).map(ref=>result.outcome.graph.objects.find(node=>node.id===ref[1])?.properties)??[],
        publicDigest:result?.exported.sidecar.compiler.publicDigest??null,
        generatedRecords:result?.exported.public.timeline.filter(item=>item.kind==='semantic').length??0,
        verification,
        ownedTracerBoundaryFailure,
        controlledProjectionRecords:records.length,expected});
    }
    const line=certified.records[index];
    if(typeof line!=='string') throw new Error('Owned tracer produced no controlled selected record at this semantic step');
    return line;
  };
  selector.referenceFor=(programId,invocationId)=>cache.get(`${programId}:${invocationId}`)?.expected;
  selector.projections=projections;
  return selector;
}

let ownedPolicySourcePromise;
async function ownedPolicySource() {
  if(ownedPolicySourcePromise) return ownedPolicySourcePromise;
  ownedPolicySourcePromise=(async()=>{
    const ts=await import('typescript');
    const original=await readFile(join(repoRoot,'applications/workflow/handle/crisp.ts'),'utf8');
    const source=original.replace(/^import type .*?;\s*/m,'').replace('export default function crisp','function crisp');
    const js=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.None,
      isolatedModules:true,removeComments:false}}).outputText;
    const ast=ts.createSourceFile('crisp.js',js,ts.ScriptTarget.ES2022,true,ts.ScriptKind.JS);
    const statements=ast.statements.map(statement=>{
      if(!ts.isVariableStatement(statement)) return statement;
      const declarations=statement.declarationList.declarations;
      if(declarations.length!==1||!ts.isIdentifier(declarations[0].name)||!declarations[0].initializer||
        !ts.isArrowFunction(declarations[0].initializer)) return statement;
      const arrow=declarations[0].initializer,body=ts.isBlock(arrow.body)?arrow.body:
        ts.factory.createBlock([ts.factory.createReturnStatement(arrow.body)],true);
      return ts.factory.createFunctionDeclaration(undefined,undefined,declarations[0].name.text,undefined,
        arrow.parameters,arrow.type,body);
    });
    const normalized=ts.factory.updateSourceFile(ast,statements);
    return '"use strict";\n'+ts.createPrinter({removeComments:false}).printFile(normalized);
  })();
  return ownedPolicySourcePromise;
}

function selectedValue(value,heap,seen=new Map()) {
  if(!Array.isArray(value)) return value;
  const [kind,atom]=value;
  if(kind==='undefined') return undefined;
  if(kind==='null') return null;
  if(['bool','str','num','bigint'].includes(kind)) return kind==='bigint'?BigInt(atom):atom;
  if(kind==='ref') {
    if(seen.has(atom)) return seen.get(atom);
    const node=heap.get(atom);if(!node)throw new Error(`Selected input graph is missing ${atom}`);
    const out=node.kind==='array'?[]:{};seen.set(atom,out);
    for(const [key,member] of node.properties) {
      if(key==='length'&&node.kind==='array') out.length=member[1];
      else out[key]=selectedValue(member,heap,seen);
    }
    return out;
  }
  throw new Error(`Unsupported selected input value ${kind}`);
}

function selectedGraphValue(value,objects,seen=new Map()) {
  if(!Array.isArray(value)) return value;
  const [kind,atom]=value;
  if(kind==='undefined') return undefined;
  if(kind==='null') return null;
  if(['bool','str','num','bigint'].includes(kind)) return kind==='bigint'?BigInt(atom):atom;
  if(kind==='ref') {
    if(seen.has(atom)) return seen.get(atom);
    const node=objects.find(item=>item.id===atom);if(!node)throw new Error(`Owned result graph is missing ${atom}`);
    const out=node.kind==='array'?[]:{};seen.set(atom,out);
    for(const [key,member] of node.properties) {
      if(key==='length'&&node.kind==='array') out.length=member[1];
      else out[key]=selectedGraphValue(member,objects,seen);
    }
    return out;
  }
  throw new Error(`Unsupported owned result value ${kind}`);
}

if(import.meta.url===new URL(process.argv[1],`file://${process.cwd()}/`).href){
  if(mode==='model'){
    const {createMegatokenProcessSelector}=await import('tracing-js-self/training');
    const command=flags.get('--model-command');
    const args=JSON.parse(flags.get('--model-args')??'[]');
    const timeoutMs=Number(flags.get('--rpc-timeout-ms')??120000);
    const selector=createMegatokenProcessSelector({command,args,cwd:flags.get('--model-cwd'),timeoutMs,representation:'explicit'});
    const checkpointIdentity=flags.get('--checkpoint-identity'),seed=Number(flags.get('--seed')??17);
    let identity=null,report,shutdown;
    try {
      identity=await selector.initialize({seed,checkpointIdentity});
      report=await runSelectedPolicyOrder({select:selector,origin:'model',seed,
        checkpointIdentity:identity.checkpointIdentity});
    } catch(error) {
      report={schema:'natlang-workflow-selected-order/1',origin:'model',seed,checkpointIdentity,
        optimizerSteps:0,learnedSuccess:null,
        outcome:'rejected',journeys:{initialization:{outcome:'rejected',failure:{name:error?.name??'Error',
          category:error?.category??null,message:String(error?.message??error),evidence:error?.evidence??null}}},
        journal:[],effects:[],serviceEvents:[],selectedTrace:error?.evidence?.prefix?[error.evidence.prefix]:[],
        controlledProjections:[],cleanup:{temporaryDirectoryRemoved:null}};
    } finally {
      const status=selector.status();
      if(!status.closed&&status.pending===0) {
        try { shutdown=await selector.shutdown({timeoutMs:10000}); }
        catch(error) { shutdown={graceful:false,error:String(error?.message??error)}; }
      } else selector.close();
    }
    process.stdout.write(JSON.stringify({...report,identity,processShutdown:shutdown??null},null,2)+'\n');
    if(report.outcome!=='applied') process.exitCode=1;
  } else if(mode==='reference') {
    const report=await runWorkflowPilot({origin:'controlled',seed:Number(flags.get('--seed')??17)});
    process.stdout.write(JSON.stringify(report,null,2)+'\n');
  } else {
    const seed=Number(flags.get('--seed')??17),selector=controlledSourceProjectionSelector();
    const report=await runSelectedPolicyOrder({select:selector,origin:'controlled',seed});
    process.stdout.write(JSON.stringify(report,null,2)+'\n');
    if(report.outcome!=='applied') process.exitCode=1;
  }
}
