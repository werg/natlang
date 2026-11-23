#!/usr/bin/env node
/** Recheck original source bytes and an independent typed review; no teacher rerun. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {gunzipSync} from 'node:zlib';
import {resolve,join,basename} from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {isDeepStrictEqual as same} from 'node:util';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const args=Object.fromEntries(process.argv.slice(2).reduce((rows,value,index,all)=>index%2===0?[...rows,[value.slice(2),all[index+1]]]:rows,[]));
for(const key of ['audit-root','packet','pins','catalog','review','out'])if(!args[key])throw Error(`missing --${key}`);
const root=resolve(args['audit-root']),out=resolve(args.out);
const raw=await readFile(args.packet),packet=JSON.parse(args.packet.endsWith('.gz')?gunzipSync(raw):raw);
const pins=JSON.parse(await readFile(args.pins,'utf8'));
const task=(await readFile(args.catalog,'utf8')).trim().split('\n').map(JSON.parse).find(row=>row.scenario===packet.test.name);
const d=packet.execution.diagnostics,prior=d.openGoalVerification,native=d.natlangTeacher;
const binding=prior.assignedPublicBrief.binding,shared=prior.assignedPublicBrief.shared;
const review=JSON.parse(await readFile(args.review,'utf8'));
if(!task || task.split!=='train' || packet.result.passed!==false || native.executionError || native.captureErrors?.length ||
 Object.values(pins.repositories).some(row=>row.dirty!==false))throw Error('requires original completed native execution with clean provenance');
if(!same(Object.entries(prior.checks).filter(([,value])=>value!==true).map(([key])=>key),['exact_citations']))
 throw Error('other original outcome failures remain unresolved');
if(review.schema!=='natlang.live-workflow-review/1' || review.model!=='openai-codex:gpt-5.6-terra' || review.teacher!=='openai-codex:gpt-6-luna' ||
 review.teacher!==pins.teacher || review.model===review.teacher || review.verdict?.accepted!==true ||
 review.semanticResult?.supported!==true || review.semanticResult?.fulfillsGoal!==true || review.semanticResult?.needsReview!==false ||
 !review.semanticResult.rationale || !review.exchanges?.length ||
 review.program?.sha256!==sha(review.program.source+'\n'+review.program.types) || review.inputSha256!==sha(JSON.stringify(review.input)))
 throw Error('independent reviewer identity, verdict, program or input custody is invalid');
const finalReview=review.exchanges.at(-1).reply;
if(review.exchanges.some(({reply})=>reply.raw_response?.provider!=='openai-codex'||reply.raw_response?.model!=='gpt-5.6-terra') ||
 !finalReview.calls?.some(([name,args])=>name==='return_result'&&args.status==='success'&&same(args.value,review.semanticResult)) ||
 !finalReview.raw_calls?.some(call=>call.function?.name==='return_result'&&same(JSON.parse(call.function.arguments).value,review.semanticResult)))
 throw Error('independent semantic verdict differs from original captured reviewer replies');
const sources=shared.reads.map(row=>{
 const {page}=row;
 if(page.truncated || Buffer.byteLength(page.head)!==page.size || sha(page.head)!==page.digest)throw Error('original raw source body is incomplete or hash-mismatched');
 return {requested:row.requested,url:page.url,title:page.title,sha256:page.digest,text:page.head};
});
const cites=native.report.brief.statements.flatMap(statement=>statement.cites.map(({url,quote})=>({url,quote})));
const actual=review.input?.actual;
if(review.input?.kind!=='public-claims' || actual?.goal!==prior.goal || !same(JSON.parse(actual.answer),native.report) ||
 !same(actual.sources,sources) || !same(actual.citations,cites) || !same(actual.sourceScope,binding.sources))
 throw Error('semantic review differs from actual original named application result and native sources');
await mkdir(out,{recursive:false});
async function store(name,bytes){await writeFile(join(out,name),bytes);return {path:name,sha256:sha(bytes)}}
const require=createRequire(join(root,'host','package.json')),{build}=require('esbuild');
const checker=join(root,'system-testing-followup/skills/system-testing/web-brief-managed.ts');
const stubs=new Map([
 ['@workspace/runtime','export const web={};'],
 ['@workspace-skills/web-research-bindings','export function bindAssignedResearch(){throw Error("offline verifier has no live services")}'],
 ['@vibestudio/shared/rpcMethods','export const createReceiverRpcMethods=()=>({});'],
 ['@workspace/web-core/contract','export const WEB_READ_MAX_LIMIT=65536;'],
]);
const bundled=await build({stdin:{contents:`export {checkManagedWebBriefEvidence} from ${JSON.stringify(checker)};`,resolveDir:root},bundle:true,platform:'node',format:'esm',write:false,metafile:true,
 alias:{'@vibestudio/content-addressing':'/home/werg/vibestudio/node_modules/@vibestudio/content-addressing/src/index.ts'},plugins:[{name:'no-live-services',setup(plugin){
 plugin.onResolve({filter:/.*/},args=>stubs.has(args.path)?{path:args.path,namespace:'offline'}:undefined);
 plugin.onLoad({filter:/.*/,namespace:'offline'},args=>({contents:stubs.get(args.path),loader:'js'}));}}]});
const bundle=await store('checker.bundle.mjs',bundled.outputFiles[0].contents);
const {checkManagedWebBriefEvidence}=await import(pathToFileURL(join(out,bundle.path)));
const cache=shared.reads.map((row,index)=>({requested:row.requested,digest:row.page.digest,size:row.page.size,
 verified:sha(sources[index].text)===row.page.digest,visible:sources[index].text}));
const crisp=checkManagedWebBriefEvidence({id:prior.caseId,goal:prior.goal,sources:binding.sources,facts:[]},native.report.brief,
 new Date(shared.request.retrievedAt).toISOString().slice(0,10),{reads:shared.reads,tails:shared.tails,cache});
if(Object.values(crisp).some(value=>value!==true))throw Error('corrected crisp native source/citation checks failed: '+JSON.stringify(crisp));
const sourcesBound=[bundle,await store('assessor.mjs',await readFile(fileURLToPath(import.meta.url)))];
for(const [index,path] of Object.keys(bundled.metafile.inputs).filter(path=>path!=='<stdin>'&&!path.startsWith('offline:')).entries())
 sourcesBound.push(await store(`checker-source-${index}-${basename(path)}`,await readFile(resolve(path))));
for(const relative of ['scripts/review-live-workflow-teacher.mjs','scripts/reviews/assessPublicClaims.nl','scripts/reviews/types.ts'])
 sourcesBound.push(await store('reviewer-'+basename(relative),await readFile(join(root,'semantic-review-admission',relative))));
const checks={exact_assigned_urls:crisp.assigned&&crisp.continuations,verified_cache:crisp.cache,exact_citations:crisp.citations,grounded_decision:crisp.outcome&&review.verdict.accepted};
if(!same(Object.keys(checks).sort(),Object.keys(prior.checks).sort()))throw Error('unreviewed original check dimensions');
const proof={schema:'natlang.live-outcome-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),
 goal:prior.goal,entry:native.entry,actualResult:native.report,passed:true,checks,
 rationale:'Original complete native source bodies independently match their UTF8 size/SHA256; exact normalized citation spans pass the corrected source checker. A distinct Terra typed assessor judged the complete actual native returned value against the actual user goal and source bytes, accepting both requested distinctions. Original failed outcome and Luna exchanges are retained.',
 checker_sources:sourcesBound,evidence:[await store('independent-review.json',await readFile(args.review)),await store('native-source-witness.json',JSON.stringify({request:shared.request,reads:shared.reads,tails:shared.tails,cache,crisp},null,2))]};
const completed=await store('completed-outcome.json',JSON.stringify(proof,null,2)+'\n');
await store('assessment.json',JSON.stringify({schema:'natlang.live-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),rationale:proof.rationale,completed_outcome:completed},null,2)+'\n');
console.log(JSON.stringify({task_id:task.taskId,assessment:join(out,'assessment.json'),completed_outcome:completed,teacher_rerun:false}));
