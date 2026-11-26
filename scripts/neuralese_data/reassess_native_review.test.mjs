import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {reassessNativeReview} from './reassess_native_review.mjs';
const sha=value=>createHash('sha256').update(value).digest('hex');
const ref=value=>({complete:true,bytes:Buffer.byteLength(JSON.stringify(value)),hash:sha(JSON.stringify(value))});
function fixture(){
 const goal='Check this original worker and leave a private diagnostic.',entry={program:'chat',callable:'chat',arguments:{goal}};
 const source=Buffer.from('if(agent.participantId&&agent.participantId!==receipt.participantId) retainedWorkerRelationship(independentlyRead.events,newWorker.participantId) oldWorker.participantId===newWorker.participantId');
 const index={openingWorker:{objectKey:'original',participantId:'native-original'}},frames=[{channelId:'chat',worker:{revision:1}},{channelId:'chat',worker:{revision:3}}],report={disposition:{status:'ready'}};
 const input={actual:{goal,observations:{index,frames,evidence:{native:true}},artifacts:{artifact:'private diagnostic'},requestedReturn:report},expected:{goal,requestedReturnRequired:true},rubric:'Old review.'};
 const prior={caseId:'chat-continuity-case',goal,checks:{native:true},actualResult:report,index,frames,evidence:{native:true},artifact:'private diagnostic',semanticReviewRequired:true,semanticReviewInput:input,outcomeContract:{returnedValue:{required:true}}};
 const effects=[['chatContinuityIndex','list',[],index],...frames.map(frame=>['chatContinuityState','read',['chat'],frame])].map(([service,method,args,result])=>({service,method,by:'agent',args:ref(args),result:ref(result)}));
 const native={entry,report,rootCallIds:['root'],calls:[{callId:'root',record:{outcome:'done',output:ref(report),effects}}]};
 const packet={test:{name:'open-goal-chat'},result:{passed:true},execution:{diagnostics:{openGoalVerification:prior,natlangTeacher:native}}};
 const review={schema:'natlang.live-workflow-review/1',input,verdict:{accepted:false},exchange:{request:{},reply:{}}};
 const pins={teacher:'openai-codex:gpt-6-luna',repositories:{base:{dirty:false}},validator_sha256:sha(source)},task={scenario:packet.test.name,split:'train',interaction:'open-user-goal',entry,goalSha256:sha(JSON.stringify(goal))};
 const checker={observedChatWorkerIdentity:()=>({sameObjectKey:true,sameParticipantId:true}),chatContinuityReviewGuidance:'Native identity guidance',observedContentReview:(proof,contract,value,keys,artifacts)=>({semanticReviewRequired:true,semanticReviewInput:{actual:{goal:proof.goal,observations:Object.fromEntries(keys.map(key=>[key,proof[key]])),artifacts:Object.fromEntries(artifacts.map(key=>[key,proof[key]])),requestedReturn:value},expected:{goal:proof.goal,requestedReturnRequired:contract.returnedValue.required},rubric:'Judge actual mode. '}})};
 const bindInput=proof=>{assert.deepEqual(proof.semanticReviewInput.actual.observations,{index:proof.index,frames:proof.frames,evidence:proof.evidence});return proof.semanticReviewInput;};return {packet,pins,task,review,source,checker,bindInput};
}
const run=s=>reassessNativeReview(s.packet,s.pins,s.task,s.review,s.source,'chat-continuity',s.checker,entry=>entry,s.bindInput);
test('keeps all native checks and immutable original rejection while binding new source-derived facts',()=>{const s=fixture(),before=structuredClone(s.packet),r=run(s);assert.deepEqual(r.projected.checks,{native:true});assert.deepEqual(r.reviewInput.actual.observations.evidence.nativeWorkerIdentity,{sameObjectKey:true,sameParticipantId:true});assert.deepEqual(r.reviewInput.actual.requestedReturn,before.execution.diagnostics.natlangTeacher.report);assert.deepEqual(s.packet,before);assert.equal(s.review.verdict.accepted,false);});
for(const[label,change]of[
 ['native failure',s=>{s.packet.execution.diagnostics.openGoalVerification.checks.native=false;}],
 ['unbound review',s=>{s.review={...s.review,input:{actual:{goal:'different'}}};}],
 ['missing original rejection exchange',s=>{delete s.review.exchange;}],
 ['unbound original actor effect',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects.pop();}],
 ['unfinished original',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.outcome='error';}],
 ['altered source',s=>{s.source=Buffer.from('other');}],
 ['altered root output',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.output.hash='0'.repeat(64);}],
 ['held split',s=>{s.task.split='test';}],
 ['review waiver',s=>{s.checker.observedContentReview=()=>({semanticReviewRequired:false});}],
])test(`refuses ${label}`,()=>{const s=fixture();change(s);assert.throws(()=>run(s));});

function authorityFixture(){
 const s=fixture(), prior=s.packet.execution.diagnostics.openGoalVerification, native=s.packet.execution.diagnostics.natlangTeacher;
 s.source=Buffer.from('binding.services.assignedAuthorityRead.preflight(operationId) snapshots.push(observation)');s.pins.validator_sha256=sha(s.source);
 const observation={operationId:'site',decision:'acquirable',leaves:[{capability:'external.open',resourceKey:'https://review.example.invalid',tier:'gated',status:'acquirable'}]};
 prior.caseId='route-authority-case';prior.offer={request:{operations:[{id:'site'}],ceiling:[{capability:'external.open',resourceKey:'https://review.example.invalid'}]}};prior.snapshots=[observation];
 prior.authority={status:'needs-acquisition',protectedLeaves:prior.offer.request.ceiling,held:[],reviews:[{observation,outside:[],disposition:'needs-acquisition'}]};prior.choice=null;prior.readback=null;prior.artifact=null;
 const report={outcome:{status:'needs-approval',summary:'Native approval remains acquirable; no visit occurred.'},readback:null,handoff:null};prior.actualResult=native.report=report;native.calls[0].record.output=ref(report);
 native.calls[0].record.effects=[['routeEvidenceRead','discover',[],prior.offer],['assignedAuthorityRead','preflight',['site'],observation]].map(([service,method,args,result])=>({service,method,by:'crisp',args:ref(args),result:ref(result)}));
 s.checker.observedRouteAuthority=(rows,ceiling)=>({status:'needs-acquisition',protectedLeaves:ceiling,held:[],reviews:rows.map(observation=>({observation,outside:[],disposition:'needs-acquisition'}))});s.checker.routeAuthorityReviewGuidance='Exact acquirable preflight requires approval; do not visit to report it.';
 const keys=['offer','snapshots','authority','choice','readback'];
 prior.semanticReviewInput=s.checker.observedContentReview(prior,prior.outcomeContract,report,keys,['artifact']).semanticReviewInput;s.review.input=structuredClone(prior.semanticReviewInput);
 s.bindInput=proof=>{assert.deepEqual(proof.semanticReviewInput.actual.observations,Object.fromEntries(keys.map(key=>[key,proof[key]])));return proof.semanticReviewInput;};return s;
}
const authorityRun=s=>reassessNativeReview(s.packet,s.pins,s.task,s.review,s.source,'route-authority',s.checker,entry=>entry,s.bindInput);
test('binds an approval-state query to actual native offer, exact preflights and unchanged answer without a visit',()=>{const s=authorityFixture(),before=structuredClone(s.packet),r=authorityRun(s);assert.deepEqual(s.packet,before);assert.deepEqual(r.projected.checks, before.execution.diagnostics.openGoalVerification.checks);assert.deepEqual(r.reviewInput.actual.requestedReturn,s.packet.execution.diagnostics.natlangTeacher.report);assert.match(r.reviewInput.rubric,/do not visit/);assert.equal(s.review.verdict.accepted,false);});
for(const[label,change]of[
 ['missing native offer',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects.shift();}],
 ['missing native preflight',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects.pop();}],
 ['unobserved approval',s=>{s.packet.execution.diagnostics.openGoalVerification.snapshots[0].decision='allowed';}],
 ['forged aggregate authority',s=>{s.packet.execution.diagnostics.openGoalVerification.authority.status='eligible';}],
 ['duplicate preflight',s=>{const p=s.packet.execution.diagnostics.openGoalVerification;p.snapshots.push(structuredClone(p.snapshots[0]));}],
 ['foreign operation',s=>{s.packet.execution.diagnostics.openGoalVerification.offer.request.operations=[];}],
 ['missing domain guidance',s=>{delete s.checker.routeAuthorityReviewGuidance;}],
])test(`rejects approval reassessment with ${label}`,()=>{const s=authorityFixture();change(s);assert.throws(()=>authorityRun(s));});

function councilFixture(note=false){
 const s=fixture(),p=s.packet.execution.diagnostics.openGoalVerification,n=s.packet.execution.diagnostics.natlangTeacher;
 s.source=Buffer.from('Regency forecast basis changed during observation verifiedRegencyAdvance: verifiedPrivateRegencyNote: preservedRegencyWorkspace:');s.pins.validator_sha256=sha(s.source);
 p.caseId='regency-briefing-case';p.index={realm:'realm',memo:null};p.frontiers=[{realm:'realm',revision:'saved',accounts:{month:3},pending:'existing-turn'}];p.forecasts=[];p.witnesses=[];p.advance=null;p.note=null;p.artifact=null;
 if(note){p.artifact=JSON.stringify({realm:'realm',opening:p.frontiers[0],current:p.frontiers[0],advance:null});p.note={realm:'realm',revision:'saved',path:'/.tmp/owned/note.json',digest:sha(p.artifact)};}
 n.calls[0].record.effects=[['regencyAssignment','list',[],p.index],['regencySaved','read',['realm'],p.frontiers[0]],...(note?[['regencyNote','stage',[{}],p.note]]:[])].map(([service,method,args,result])=>({service,method,by:'crisp',args:ref(args),result:ref(result)}));
 s.checker.observedRegencyEvidence=(current,note,body)=>({currentCouncil:{revision:current.revision,pendingSettlement:current.pending},retainedPrivateNote:{receipt:note,bytesSha256:body===null?null:sha(body)}});s.checker.regencyReviewGuidance='Native current pending settlement and newly retained note bytes differ from an old instruction memo.';
 const keys=['index','frontiers','forecasts','witnesses','advance'];
 p.semanticReviewInput=s.checker.observedContentReview(p,p.outcomeContract,n.report,keys,['artifact']).semanticReviewInput;s.review.input=structuredClone(p.semanticReviewInput);
 s.oldBinder=proof=>{assert.deepEqual(proof.semanticReviewInput.actual.observations,Object.fromEntries(keys.map(key=>[key,proof[key]])));return proof.semanticReviewInput;};
 s.bindInput=proof=>{assert.deepEqual(proof.semanticReviewInput.actual.observations,Object.fromEntries([...keys,'nativeFacts'].map(key=>[key,proof[key]])));return proof.semanticReviewInput;};return s;
}
const councilRun=s=>reassessNativeReview(s.packet,s.pins,s.task,s.review,s.source,'regency-briefing',s.checker,entry=>entry,s.bindInput,s.oldBinder);
test('adds source-derived council pending and actual retained-note roles without changing original evidence',()=>{for(const note of [false,true]){const s=councilFixture(note),before=structuredClone(s.packet),r=councilRun(s);assert.deepEqual(s.packet,before);assert.equal(r.projected.nativeFacts.currentCouncil.pendingSettlement,'existing-turn');assert.equal(r.projected.nativeFacts.retainedPrivateNote.bytesSha256,note?sha(r.prior.artifact):null);assert.equal(s.review.verdict.accepted,false);}});
for(const[label,change]of[
 ['missing saved native call',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects.splice(1,1);}],
 ['missing retained native call',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects.pop();}],
 ['changed note bytes',s=>{s.packet.execution.diagnostics.openGoalVerification.artifact+=' ';}],
 ['forged note revision',s=>{s.packet.execution.diagnostics.openGoalVerification.note.revision='other';}],
 ['unobserved advance',s=>{s.packet.execution.diagnostics.openGoalVerification.advance={realm:'realm',basis:'saved',months:1,committed:{revision:'new'}};}],
])test(`refuses council recovery with ${label}`,()=>{const s=councilFixture(true);change(s);assert.throws(()=>councilRun(s));});
function gardenFixture(){
 const s=fixture(),p=s.packet.execution.diagnostics.openGoalVerification,n=s.packet.execution.diagnostics.natlangTeacher;
 s.source=Buffer.from('boundOwnedScene: keptCurrentAnchor: independentlyReadPrivateDraft: preservedNativeWorld:');s.pins.validator_sha256=sha(s.source);
 p.caseId='grimoire-scene-case';p.docket={id:'garden'};p.scope={ownerId:'owner',frontier:'old'};p.view={ownerId:'owner',frontier:'new'};p.current=structuredClone(p.view);p.gate={status:'current',frontier:'new'};p.receipt=null;p.readback={fileDigest:null};p.body=null;
 p.journal=[{service:'grimoireScene',method:'readScope',args:['garden'],result:p.scope},{service:'grimoireScene',method:'view',args:['garden'],result:p.view},{service:'grimoireScene',method:'confirm',args:['garden','new'],result:p.gate}];n.calls[0].record.effects=p.journal.map(row=>({...row,by:'crisp',args:ref(row.args),result:ref(row.result)}));
 s.checker.observedGrimoireAuthority=(scope,view,current,gate)=>({scopeFrontierMatchesCurrent:scope.frontier===current.frontier,viewedFrontierMatchesCurrent:view.frontier===current.frontier,gateConfirmsViewedFrontier:gate.frontier===view.frontier});s.checker.grimoireReviewGuidance='A current view does not renew the older scoped write authority.';
 const keys=['docket','view','current','scope','gate','readback','journal'];p.semanticReviewInput=s.checker.observedContentReview(p,p.outcomeContract,n.report,keys,['body']).semanticReviewInput;s.review.input=structuredClone(p.semanticReviewInput);
 s.oldBinder=proof=>{assert.deepEqual(proof.semanticReviewInput.actual.observations,Object.fromEntries(keys.map(key=>[key,proof[key]])));return proof.semanticReviewInput;};s.bindInput=proof=>{assert.deepEqual(proof.semanticReviewInput.actual.observations,Object.fromEntries([...keys,'nativeFacts'].map(key=>[key,proof[key]])));return proof.semanticReviewInput;};return s;
}
const gardenRun=s=>reassessNativeReview(s.packet,s.pins,s.task,s.review,s.source,'grimoire-scene',s.checker,entry=>entry,s.bindInput,s.oldBinder);
test('preserves stale caretaker scope while distinguishing current view confirmation',()=>{const s=gardenFixture(),before=structuredClone(s.packet),r=gardenRun(s);assert.deepEqual(s.packet,before);assert.deepEqual(r.projected.nativeFacts,{scopeFrontierMatchesCurrent:false,viewedFrontierMatchesCurrent:true,gateConfirmsViewedFrontier:true});});
for(const[label,change]of[
 ['missing native scope call',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects.shift();}],
 ['foreign scope owner',s=>{s.packet.execution.diagnostics.openGoalVerification.scope.ownerId='foreign';}],
 ['invented hold artifact',s=>{s.packet.execution.diagnostics.openGoalVerification.body='invented';}],
 ['unobserved confirmation',s=>{s.packet.execution.diagnostics.openGoalVerification.gate.status='changed';}],
])test(`refuses caretaker recovery with ${label}`,()=>{const s=gardenFixture();change(s);assert.throws(()=>gardenRun(s));});

/** TEST-only bounded protocol packets, never registered as teacher samples.
 * Actual Source48 packet preparation is a separate source-bound control. */
function premiseFixture(kind){
 const s=fixture(),p=s.packet.execution.diagnostics.openGoalVerification,n=s.packet.execution.diagnostics.natlangTeacher;
 const packageCase=kind==='installed-package';p.caseId=kind+'-case';
 s.source=Buffer.from(packageCase?'scope.workingRevision === workingRevision scope.workingDigest === local.digest decisionFor(current.observation) !== "restore-approved-field" verifiedLocalManifestEffect: preservedProtectedAndNeighbors:':'contextId !== ports.contextId directory: () => fs.mkdtemp("system-release-evidence") decode.decode(raw) === expectedBytes receipt.sha256 === hash(expectedBytes) sameEvidenceMembers(cases, reads)');s.pins.validator_sha256=sha(s.source);
 const oldKeys=packageCase?['effects','assignment','observed','plan','repair']:['assignment','runs','cases','effects'];const oldArtifacts=packageCase?[]:['handoffText'];let rows;
 if(packageCase){
  const scope={source:'workers/images',field:'native-suite',key:'native',grant:'restore',workingRevision:'before',workingDigest:'before-digest',baselineDigest:'approved'};
  p.assignment={operationId:'assigned',field:'native-suite',key:'native',scope};p.observed={operationId:'assigned',source:'workers/images',workingRevision:'before',workingDigest:'before-digest',baselineDigest:'approved',scopeMatches:true};p.plan={suite:'native',runtime:'native'};p.repair={status:'restored',observation:{...p.observed,workingRevision:'after',workingDigest:'after-digest',scopeMatches:false,targetFieldMatches:true,suiteRequest:{operationId:'plan'}}};
  n.report={finding:{reason:'The native suite is restored at its approved pre-action scope.'},verification:{plan:p.plan}};
  rows=[['installedPackageRead','discover',[],p.assignment],['installedPackageRead','inspect',['assigned'],p.observed],['installedPackageChange','restore',['assigned','before','before-digest'],p.repair],['testSuitePlanRead','resolve',['plan'],p.plan]];
  s.checker.openInstalledPackageWorlds=[{id:p.caseId,goal:p.goal,requestedReturnRequired:true}];s.checker.installedPackageGoalContract=()=>({returnedValue:{required:true}});s.checker.installedPackageAuthorization=({before,after,scope,action})=>({before,after,scope,action});
 }else{
  p.contextId='owned';p.assignment={id:'assigned',contextId:'owned'};p.runs={current:{id:'current'}};p.cases=[{name:'delivery',primary:{code:'DELIVERY_LATENCY',detail:'Actual retained delivery was late.'}}];
  p.handoffText=JSON.stringify({protocol:'vibestudio-local-system-release-review.v1',contextId:'owned',runId:'current',disposition:'stage-diagnostics',cases:p.cases});p.handoff={path:'/.tmp/native/release-evidence.json',bytes:Buffer.byteLength(p.handoffText),sha256:sha(p.handoffText)};n.report={assessment:{reason:'Actual delivery was late.'},handoff:p.handoff};
  rows=[['systemReleaseEvidence','discover',[],p.assignment],['systemReleaseEvidence','compare',['assigned'],p.runs],['systemReleaseEvidence','inspectCase',['assigned','current','delivery'],p.cases[0]],['systemReleaseHandoff','stage',['assigned','current','stage-diagnostics',p.cases],p.handoff]];
  s.checker.openSystemReleaseWorlds=[{id:p.caseId,goal:p.goal,requestedReturnRequired:false}];s.checker.systemReleaseGoalContract=()=>({returnedValue:{required:false}});s.checker.systemReleaseHandoffProtocol=({contextId,verifiedPrivateHandoff,receipt,text})=>verifiedPrivateHandoff&&contextId==='owned'&&receipt.bytes===Buffer.byteLength(text)&&receipt.sha256===sha(text)?{storage:'context-scoped-scratch',externalTransmission:false}:null;
 }
 p.actualResult=n.report;n.calls[0].record.output=ref(n.report);n.calls[0].record.effects=rows.map(([service,method,args,result])=>({service,method,args:ref(args),result:ref(result),by:'crisp'}));
 p.effects=rows.map(([service,method,args,result])=>({service,method,args,result}));p.checks={native:true,verifiedPrivateHandoff:true};p.observedChecks={...p.checks,answer:true};
 const oldInput={actual:{goal:p.goal,observations:Object.fromEntries(oldKeys.map(k=>[k,p[k]])),artifacts:Object.fromEntries(oldArtifacts.map(k=>[k,p[k]])),requestedReturn:null},expected:{goal:p.goal,requestedReturnRequired:false},rubric:'Original immutable review'};p.semanticReviewInput=oldInput;s.review.input=structuredClone(oldInput);
 s.oldBinder=proof=>proof.semanticReviewInput;s.bindInput=proof=>{assert.equal(proof.semanticReviewInput.actual.requestedReturn,packageCase?n.report:null);return proof.semanticReviewInput;};
 s.checker.observedContentReview=(proof,contract,value,keys,artifacts)=>({semanticReviewRequired:true,semanticReviewInput:{actual:{goal:proof.goal,observations:Object.fromEntries(keys.map(k=>[k,proof[k]])),artifacts:Object.fromEntries(artifacts.map(k=>[k,proof[k]])),requestedReturn:contract.returnedValue.required?value:null},expected:{goal:proof.goal,requestedReturnRequired:contract.returnedValue.required},rubric:'Exact native temporal and artifact semantics'}});
 s.checker.assessSemanticOutcome=(proof,contract)=>({...proof,passed:!contract.returnedValue.required||proof.observedChecks.answer===true,outcomeDimensions:{returnedValue:{required:contract.returnedValue.required,passed:proof.observedChecks.answer===true}}});return s;
}
const premiseRun=(s,kind)=>reassessNativeReview(s.packet,s.pins,s.task,s.review,s.source,kind,s.checker,entry=>entry,s.bindInput,s.oldBinder);
for(const kind of ['installed-package','system-release'])test(`${kind} binds source-owned chronology or exact local protocol without rewriting original`,()=>{const s=premiseFixture(kind),before=structuredClone(s.packet),r=premiseRun(s,kind);assert.deepEqual(s.packet,before);assert.equal(r.projected.passed,true);assert.equal(s.review.verdict.accepted,false);assert.equal(r.reviewInput.expected.requestedReturnRequired,kind==='installed-package');if(kind==='installed-package'){assert.equal(r.projected.authorization.before.scopeMatches,true);assert.equal(r.projected.authorization.after.scopeMatches,false);}else assert.equal(r.projected.handoffProtocol.externalTransmission,false);});
for(const[label,kind,change]of[
 ['withdrawn pre-action scope','installed-package',s=>{s.packet.execution.diagnostics.openGoalVerification.assignment.scope.grant='suspended';}],
 ['foreign pre-action revision','installed-package',s=>{s.packet.execution.diagnostics.openGoalVerification.assignment.scope.workingRevision='foreign';}],
 ['false pre-action matching scope','installed-package',s=>{s.packet.execution.diagnostics.openGoalVerification.observed.scopeMatches=false;}],
 ['missing native restore effect','installed-package',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects.splice(2,1);}],
 ['missing newly required answer','installed-package',s=>{s.packet.execution.diagnostics.openGoalVerification.observedChecks.answer=false;}],
 ['changed actual suite value','installed-package',s=>{s.packet.execution.diagnostics.openGoalVerification.plan.suite='browser';}],
 ['forged artifact bytes','system-release',s=>{s.packet.execution.diagnostics.openGoalVerification.handoffText+=' ';}],
 ['foreign artifact context','system-release',s=>{s.packet.execution.diagnostics.openGoalVerification.contextId='foreign';}],
 ['forged stage receipt','system-release',s=>{s.packet.execution.diagnostics.openGoalVerification.handoff.sha256='0'.repeat(64);}],
 ['missing native case read','system-release',s=>{s.packet.execution.diagnostics.natlangTeacher.calls[0].record.effects.splice(2,1);}],
 ['missing native local protocol','system-release',s=>{s.checker.systemReleaseHandoffProtocol=()=>null;}],
])test(`refuses ${label}`,()=>{const s=premiseFixture(kind);change(s);assert.throws(()=>premiseRun(s,kind));});

test('release case grouping may differ from canonical retained order when every actual native case is bound',()=>{
 const s=premiseFixture('system-release'),p=s.packet.execution.diagnostics.openGoalVerification,n=s.packet.execution.diagnostics.natlangTeacher;
 p.cases.push({name:'reopen',primary:null});const body=JSON.parse(p.handoffText);body.cases=p.cases;p.handoffText=JSON.stringify(body);p.handoff.bytes=Buffer.byteLength(p.handoffText);p.handoff.sha256=sha(p.handoffText);
 const args=['assigned','current','stage-diagnostics',[...p.cases].reverse()];p.effects=[...p.effects.filter(row=>row.method!=='stage'),{service:'systemReleaseHandoff',method:'stage',args,result:p.handoff}];
 n.calls[0].record.effects=[...p.effects.map(row=>({...row,args:ref(row.args),result:ref(row.result),by:'crisp'})),{service:'systemReleaseEvidence',method:'inspectCase',by:'crisp',args:ref(['assigned','current','reopen']),result:ref(p.cases[1])}];n.calls[0].record.output=ref(n.report);
 p.semanticReviewInput.actual.observations.effects=p.effects;p.semanticReviewInput.actual.observations.cases=p.cases;p.semanticReviewInput.actual.artifacts.handoffText=p.handoffText;s.review.input=structuredClone(p.semanticReviewInput);
 assert.equal(premiseRun(s,'system-release').projected.passed,true);
});
test('changing proof and reviewed sidecar together still cannot replace actual captured authority',()=>{const s=premiseFixture('installed-package'),p=s.packet.execution.diagnostics.openGoalVerification;p.assignment.scope.workingDigest='forged';s.review.input=structuredClone(p.semanticReviewInput);assert.throws(()=>premiseRun(s,'installed-package'));});
