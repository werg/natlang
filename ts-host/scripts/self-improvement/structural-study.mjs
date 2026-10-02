/** Concrete structural headroom, direct-rewrite control, and native iterative search. */
import {mkdir,readFile,writeFile,appendFile,cp,readdir,symlink} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
const [output,command='probe']=process.argv.slice(2);
if(!output||!['probe','flat-probe','native','separated','direct','confirm','draft'].includes(command))throw Error('usage: structural-study.mjs OUTPUT [probe|flat-probe|native|direct|confirm|draft]');
await mkdir(output,{recursive:true});
const path=join(output,'protocol.json');let protocol;
try{protocol=JSON.parse(await readFile(path,'utf8'));}catch(error){
 if(error.code!=='ENOENT')throw error;
 const header='---\nargs:\n  reviews: "Review[]"\nreturns: number\n---\n';
 const files={'recommending_reviews.nl':header+'How many of reviews recommend the product? Judge each review separately with an nl function, and count in code.\n','types.ts':'export type Review={id:string;text:string};\n'};
 const reference={...files,'recommending_reviews.nl':header+'Read each review and judge its overall recommendation yourself in this semantic call. Treat each independently by meaning, including negation, sarcasm and mixed sentiment. A positive detail does not override an overall rejection; a negative detail does not override an overall recommendation. Build one boolean per review in input order, then count true entries exactly in eval. Do not delegate this small task to another nl call or use keyword matching. Complete in one eval action: code explicitly returns the count, finish is true.\n'};
 const sets=[
  [['The hinge squeaks, but I use this every day and would gladly buy another.',true],['Excellent packaging. Shame the actual device is unusable; avoid it.',false],['I expected very little. It has quietly become my favourite purchase this year.',true],['If endless resets are your hobby, this is the perfect product. I sent mine back.',false],['Not the cheapest, yet I have no regrets and have suggested it to colleagues.',true],['It did work once. That is the nicest thing I can say about it.',false]],
  [['A few cosmetic flaws cannot outweigh how useful this has been. Worth buying.',true],['The specifications looked impressive. Real life was another matter; never again.',false],['I cannot recommend spending your money elsewhere; this one does the job beautifully.',true],['I would not tell a friend to buy this, even though the display looks lovely.',false],['Battery life is mediocre, but for my needs this is a keeper.',true],['Five stars for making me appreciate the old one. This replacement went straight back.',false]],
  [['I thought I would return it, but after a month I am buying a second for my sister.',true],['It is fast and attractive. Still, the random data loss makes it impossible to recommend.',false],['No flashy extras, just dependable performance. I am pleased I chose it.',true],['The box deserves an award; the contents deserve a refund.',false],['People complain about its weight. I prefer that sturdiness and would choose it again.',true],['I have stopped using it and wish I had kept the receipt.',false]],
  [['Despite a fiddly setup, it has earned a permanent place on my desk.',true],['I wanted to love it. Unfortunately I cannot suggest anyone buy it.',false],['The best thing about this purchase was the shop accepting the return.',false],['There is nothing revolutionary here, but I would happily recommend it.',true]],
  [['I would buy it again even with the same annoying fan noise.',true],['Calling it reliable would be generous. Mine failed twice and I gave up.',false],['I was sceptical; now my neighbours have bought one on my advice.',true],['Looks expensive, feels cheap, and was a mistake to buy.',false]],
  [['It falls short for gaming, but I bought it for writing and love it for that.',true],['A lovely screen does not compensate for losing my work. Stay away.',false],['The manual is poor. The product itself has been terrific and I recommend it.',true],['I cannot say this was a good purchase. It spends more time charging than working.',false]],
  [['Not bad at all: I would have no hesitation recommending it.',true],['I recommend the previous version. This one has been a disappointment.',false],['It is not perfect, but I am keeping it and would make the same choice again.',true],['Just what I needed: another expensive paperweight. Returned.',false]]
 ];
 const cases=sets.map((rows,i)=>({id:'semantic-batch-'+i,group:'fresh-review-batch-'+i,split:i<2?'train':i===2?'validation':'test',args:[rows.map(([text],j)=>({id:'r'+j,text}))],expected:rows.filter(([,positive])=>positive).length}));
 protocol={schema:'natlang.structural-study/1',optimizer:{provider:'openai-codex',model:'gpt-6-luna'},executor:{endpoint:'http://127.0.0.1:8081',model:'/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'},executorTimeoutMs:180000,files,reference,contract:{entry:'recommending_reviews.nl',exportName:'default',programId:'semantic-review-batching'},cases,policy:{maxExperiments:3,maxPopulation:4,mode:'structural',strategy:'adaptive',objective:'model-calls',allowedFiles:['recommending_reviews.nl','types.ts','aggregate.ts'],goal:'Preserve semantic recommendation judgment and exact counting. Simplify the execution structure to reduce model calls on correct answers; repair observed wrong answers first.'},budget:{maxModelCalls:450,maxRollouts:60,maxProposals:3,maxElapsedMs:1800000},design:'One frozen development study; reference headroom uses training only, reference source is never supplied to Luna. Direct receives the same baseline train feedback and one rewrite; native gets at most three evidenced experiments. Every attempt is reported. Four fresh test groups stay closed until all sources freeze; no claim of significant cost gain from this small study.'};
 await writeFile(path,JSON.stringify(protocol,null,2)+'\n');
}
// Every command executes the same copied runtime, including worker imports. Repository rebuilds cannot change a running study.
const frozenRuntime=resolve(output,'runtime');
try{await readFile(join(frozenRuntime,'runtime-identity.json'));}catch(error){
 if(error.code!=='ENOENT')throw error;
 await cp(new URL('../../dist/',import.meta.url),frozenRuntime,{recursive:true});
 await symlink(new URL('../../node_modules/',import.meta.url).pathname,join(frozenRuntime,'node_modules'),'dir');
 const hashes={};async function walk(directory){for(const entry of await readdir(directory,{withFileTypes:true})){const name=join(directory,entry.name);if(entry.isDirectory())await walk(name);else if(entry.name.endsWith('.js'))hashes[name.slice(frozenRuntime.length+1)]=createHash('sha256').update(await readFile(name)).digest('hex');}}await walk(frozenRuntime);
 await writeFile(join(frozenRuntime,'runtime-identity.json'),JSON.stringify(hashes,null,2)+'\n');
}
try{await readFile(resolve(output,'prelude.js'));}catch(error){if(error.code!=='ENOENT')throw error;await cp(new URL('../../prelude.js',import.meta.url),resolve(output,'prelude.js'));}
const runtimeImport=path=>import(pathToFileURL(join(frozenRuntime,path)).href);
const {Folder,SourceEvaluator,OperationJournal,improveProgram,openAICompatibleModelTurn,createNatlangRuntime}=await runtimeImport('index.js');
const {loadVirtualNatlang,compileVirtualProject}=await runtimeImport('runtime/virtual-project.js');
const {AUTHORED_IMPROVER}=await runtimeImport('improvement/authored-source.js');
const {TOOLS_PROMPT}=await runtimeImport('native/prompt.js');
const {UsageGateway}=await runtimeImport('evaluation/usage.js');
const {createPiModelBackend}=await runtimeImport('model/pi-provider.js');
const {fingerprint}=await runtimeImport('adaptation/identity.js');
const resultPath=join(output,command+'.json');try{await readFile(resultPath);console.log('Already completed '+command);process.exit(0);}catch(error){if(error.code!=='ENOENT')throw error;}
const allocation=new OperationJournal(join(output,command+'-allocation'));
const allocationIdentity=fingerprint({protocol,runtime:JSON.parse(await readFile(join(frozenRuntime,'runtime-identity.json'),'utf8'))});
const priorIdentity=allocation.read('identity')?.value;if(priorIdentity&&priorIdentity!==allocationIdentity)throw Error('Study allocation identity changed.');
allocation.record('identity',allocationIdentity);
const gateway=new UsageGateway(protocol.budget,allocation.read('ledger')?.value);gateway.onUpdate=ledger=>allocation.record('ledger',ledger);
const interrupted=new AbortController();
process.once('SIGTERM',()=>interrupted.abort(new Error('study assignment interrupted')));
const signal=AbortSignal.any([interrupted.signal,AbortSignal.timeout(Math.max(1,protocol.budget.maxElapsedMs-gateway.ledger.elapsedMs))]);
const rawStudent=openAICompatibleModelTurn({...protocol.executor,request:{temperature:0.2}});
const log=async(role,request,turn)=>{await appendFile(join(output,'exchanges.ndjson'),JSON.stringify({at:new Date().toISOString(),command,role,request,turn})+'\n');console.log(JSON.stringify({command,role,calls:turn.calls?.map(call=>call[0]),text:turn.text?.slice(0,80)}));};
const student=async(request,signal)=>{const turn=await rawStudent(request,signal);await log('student',request,turn);return turn;};
const evaluator=new SourceEvaluator(protocol.contract,protocol.cases,student,gateway,{executorId:protocol.executor.model,timeoutMs:protocol.executorTimeoutMs,signal,sourcePolicy:{baseline:protocol.files,mode:protocol.policy.mode,allowedFiles:protocol.policy.allowedFiles},journal:new OperationJournal(join(output,command+'-journal'))});
const baseline=Folder.fromFiles(protocol.files);let result;
// Reuse only exact completed baseline executions, preserving their original measured costs.
const baselineEvidenceFrom=command==='direct'?join(output,'native-journal'):['native','separated'].includes(command)?protocol.baselineEvidenceFrom:undefined;
if(baselineEvidenceFrom){
 const sourceJournal=new OperationJournal(baselineEvidenceFrom),targetJournal=new OperationJournal(join(output,command+'-journal'));
 const reused=[];
 for(const split of ['train','validation']){
  const rows=protocol.cases.filter(row=>row.split===split),reference=fingerprint({source:baseline.snapshot().digest,suite:evaluator.suiteVersion,split,ids:rows.map(row=>row.id),seed:0});
  for(const row of rows){const key=reference+':'+row.id,cached=sourceJournal.read(key);if(cached?.status==='done'){targetJournal.record(key,cached.value);reused.push(row.id);}}
 }
 await writeFile(join(output,command+'-baseline-reuse.json'),JSON.stringify({cases:reused,source:baselineEvidenceFrom,cost:'Original execution modelCalls retained; no new target request for a cached case.'},null,2)+'\n');
}

if(command==='probe'||command==='flat-probe'){
 const reference=command==='flat-probe'?{...protocol.files,'recommending_reviews.nl':protocol.files['recommending_reviews.nl'].split('---\n')[0]+'---\nargs:\n  reviews: \"Review[]\"\nreturns: number\n---\nRead every review and judge its overall recommendation yourself in this call. Treat each review independently by meaning, including negation, sarcasm and mixed sentiment. A positive detail does not override an overall rejection; a negative detail does not override an overall recommendation. Build one boolean per review in input order from that semantic judgment; count the true entries exactly in eval and explicitly return that number. Do not delegate this small task to another nl call, use keyword matching or guess the count. Finish with eval typed return followed by return_result {status: \"success\"} without a value in the same response when possible.\n'}:protocol.reference;
 await writeFile(join(output,command+'-reference.json'),JSON.stringify(reference,null,2)+'\n');
 const before=command==='flat-probe'?JSON.parse(await readFile(join(output,'probe.json'),'utf8')).baseline:await evaluator.evaluate(baseline.snapshot(),{split:'train'}),after=await evaluator.evaluate(Folder.fromFiles(reference).snapshot(),{split:'train'});
 result={baseline:before,reference:after,headroom:after.gatesPassed&&(after.quality>before.quality||after.quality===before.quality&&after.modelCalls<before.modelCalls),ledger:gateway.snapshot(),interpretation:'Training-only structural headroom probe, not held-out optimization gain.'};
}else if(command==='confirm'){
 const native=JSON.parse(await readFile(join(output,'native.json'),'utf8')),direct=JSON.parse(await readFile(join(output,'direct.json'),'utf8'));
 const freeze={protocol,prelude:createHash('sha256').update(await readFile(resolve(output,'prelude.js'))).digest('hex'),runtime:JSON.parse(await readFile(join(frozenRuntime,'runtime-identity.json'),'utf8')),sources:{baseline:protocol.files,native:native.sourceManifest.files,direct:direct.files}};
 const frozenPath=join(output,'freeze.json');try{const old=JSON.parse(await readFile(frozenPath,'utf8'));if(JSON.stringify(old)!==JSON.stringify(freeze))throw Error('Confirmation source changed.');}catch(error){if(error.code!=='ENOENT')throw error;await writeFile(frozenPath,JSON.stringify(freeze,null,2)+'\n');}
 const identity=createHash('sha256').update(JSON.stringify(freeze)).digest('hex');result={freeze:identity,comparisons:{}};
 for(const arm of ['native','direct']){
  const armEvaluator=new SourceEvaluator(protocol.contract,protocol.cases,student,gateway,{executorId:protocol.executor.model,timeoutMs:protocol.executorTimeoutMs,signal,journal:new OperationJournal(join(output,'confirmation-'+arm))});
  result.comparisons[arm]=await armEvaluator.confirmPair(baseline.snapshot(),Folder.fromFiles(freeze.sources[arm]).snapshot(),identity);
 }
 result.ledger=gateway.snapshot();
}else{
 const backend=createPiModelBackend(protocol.optimizer.provider,protocol.optimizer.model);
 const optimizer=async(request,signal)=>{const turn=await backend.turn(request,signal);await log('optimizer',request,turn);return turn;};
 try{
  if(command==='draft'){
   if(!protocol.observedTraining?.length)throw Error('Draft diagnostic requires pinned actual training observations.');
   const helpers=compileVirtualProject({files:{'main.ts':AUTHORED_IMPROVER['improveStep/context.ts'].replace("'../types'","'./types'"),'types.ts':AUTHORED_IMPROVER['types.ts']}},await runtimeImport('runtime/node.js'),{constrained:true,target:'node'});
   if(!helpers.ok)throw Error(JSON.stringify(helpers.diagnostics));
   const bookkeeping=helpers.require('main.ts');
   const authored=loadVirtualNatlang(AUTHORED_IMPROVER,'improveStep.nl');
   const sourceFiles=Object.entries(protocol.files).map(([path,text])=>({path,text})),evidence=protocol.observedTraining;
   const task=createNatlangRuntime({model:{driver:(request,signal)=>gateway.request(optimizer,request,signal,'reflection'),maxTurns:16,maxTokens:24000,turnTokens:2048,maxFailureRepairs:4},signal,codeEdits:'deny',network:false,onFolderProposal:()=>gateway.reserve('proposals',1,signal)});
   const proposal=await task.run(()=>baseline.propose(authored.rewriteProgram,bookkeeping.request(protocol.policy,'',evidence,sourceFiles)));
   const plan=proposal.value.summary;
   const checked=await evaluator.check(proposal.folder);
   result={plan,edit:proposal.value,checked,files:Object.fromEntries(proposal.folder.filePaths().map(path=>[path,new TextDecoder().decode(proposal.folder.readBytesSync(path))])),ledger:gateway.snapshot(),disposition:'draft-diagnostic',studentMeasured:false,interpretation:'Actual optimizer planning/editing with pinned prior development observations; compiler/edit-scope check only. No new student or held-out improvement claim.'};
  }else if(command==='native'||command==='separated'){
   const authoredSource=command==='separated'?protocol.separatedSource:protocol.improverSource;
   const improved=await improveProgram({folder:baseline,contract:protocol.contract,cases:protocol.cases,policy:protocol.policy,improver:optimizer,executor:student,executorId:protocol.executor.model,executorTimeoutMs:protocol.executorTimeoutMs,budget:protocol.budget,gateway,signal,...(authoredSource?{improverSource:Folder.fromFiles(authoredSource).snapshot()}:{}),directory:join(output,command+'-journal'),trace:trace=>{void appendFile(join(output,command+'-traces.ndjson'),JSON.stringify(trace)+'\n');}});
   const {folder,evaluator,...portable}=improved;result={...portable,source:folder.digest};
  }else{
   const train=await evaluator.evaluate(baseline.snapshot(),{split:'train'}),validation=await evaluator.evaluate(baseline.snapshot(),{split:'validation'});
   const editor=loadVirtualNatlang(protocol.improverSource??AUTHORED_IMPROVER,'improveStep.nl').rewriteProgram;
   const task=createNatlangRuntime({model:{driver:(request,signal)=>gateway.request(optimizer,request,signal,'reflection'),maxTurns:16,maxTokens:24000,turnTokens:2048,maxFailureRepairs:4},signal,codeEdits:'deny',network:false,seed:{mode:'derived',root:0},onFolderProposal:()=>gateway.reserve('proposals',1,signal)});
   const proposal=await task.run(()=>baseline.propose(editor,{brief:'Source: '+JSON.stringify(protocol.files)+'; actual training: '+JSON.stringify(evaluator.page(train.evidence)),objective:protocol.policy.objective??'quality',goal:protocol.policy.goal,mode:protocol.policy.mode,hypothesis:'Make the strongest coherent structural simplification supported by these actual student executions. Diagnose the traces yourself.',sourceFiles:baseline.snapshot().filePaths().map(path=>({path,text:new TextDecoder().decode(baseline.snapshot().readBytesSync(path))})),evidence:evaluator.page(train.evidence),allowedFiles:protocol.policy.allowedFiles}));
   const checked=await evaluator.check(proposal.folder);const candidate=checked.valid?await evaluator.evaluate(proposal.folder,{split:'validation'}):null;
   const candidateTrain=checked.valid?await evaluator.evaluate(proposal.folder,{split:'train'}):null;
   const accepted=!!candidate?.gatesPassed&&!!candidateTrain?.gatesPassed&&(candidate.quality>validation.quality||candidate.quality===validation.quality&&candidate.modelCalls<validation.modelCalls);
   result={files:accepted?Object.fromEntries(proposal.folder.filePaths().map(path=>[path,new TextDecoder().decode(proposal.folder.readBytesSync(path))])):protocol.files,accepted,baseline:validation,candidate,candidateTrain,checked,edit:proposal.value,ledger:gateway.snapshot()};
  }
 }finally{backend.close();}
}
if(command==='probe')await writeFile(join(output,'cases.jsonl'),JSON.stringify({version:'natlang.improvement-case/1',id:protocol.id??protocol.contract.programId,family:protocol.family??'semantic-review-batching',files:protocol.files,contract:protocol.contract,cases:protocol.cases,policy:protocol.policy,budget:protocol.budget,incidents:protocol.incidents??[{id:'observed-semantic-delegation-overhead',cluster:'semantic-delegation-overhead',split:'train',route:'structural-repair',reason:'Actual baseline train quality '+result.baseline.quality+' with '+result.baseline.modelCalls+' requests; independent reference quality '+result.reference.quality+' with '+result.reference.modelCalls+' requests.',source:{file:join(output,'probe.json'),hash:createHash('sha256').update(JSON.stringify(result)).digest('hex')}}],sourceGroups:protocol.sourceGroups??protocol.cases.filter(row=>row.split==='train').map(row=>row.group),provenance:{kind:result.baseline.quality<1?'observed-native-program-failure':'observed-structural-cost-opportunity',source:join(output,'probe.json'),oracle:'Independently authored semantic labels; exact count',probe:result,positiveSFT:false,disposition:'requires-native-trajectory-replay-and-admission'}})+'\n');
await writeFile(resultPath,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({command,status:'completed',headroom:result.headroom,disposition:result.disposition,accepted:result.accepted,ledger:result.ledger.roles}));
