#!/usr/bin/env node
/** Independently reassess a retained native preview; never reruns a teacher or workspace. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {gunzipSync} from 'node:zlib';
import {resolve,join,basename} from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {isDeepStrictEqual as same} from 'node:util';
import {execFileSync} from 'node:child_process';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const args=Object.fromEntries(process.argv.slice(2).reduce((rows,value,index,all)=>index%2===0?[...rows,[value.slice(2),all[index+1]]]:rows,[]));
for(const key of ['audit-root','packet','pins','catalog','out'])if(!args[key])throw Error(`missing --${key}`);
const root=resolve(args['audit-root']),out=resolve(args.out);
const raw=await readFile(args.packet),packet=JSON.parse(args.packet.endsWith('.gz')?gunzipSync(raw):raw);
const pins=JSON.parse(await readFile(args.pins,'utf8'));
const task=(await readFile(args.catalog,'utf8')).trim().split('\n').map(JSON.parse).find(row=>row.scenario===packet.test.name);
const diagnostic=packet.execution.diagnostics,prior=diagnostic.openGoalVerification,native=diagnostic.natlangTeacher;
if(!task || task.split!=='train' || packet.result.passed!==false || native.executionError || native.captureErrors?.length)
 throw Error('requires a retained failed training outcome with completed native execution');
if(Object.values(pins.repositories).some(row=>row.dirty!==false))throw Error('requires clean original provenance');
const failed=Object.entries(prior.checks).filter(([,value])=>value!==true).map(([key])=>key);
if(!same(failed,['independentlyVerifiedPresentationDom']))throw Error('other native outcome failures remain unresolved');
await mkdir(out,{recursive:false});
async function store(name,bytes){await writeFile(join(out,name),bytes);return {path:name,sha256:sha(bytes)}}
const sourceRelative='skills/system-testing/open-goal-local-presentation-managed.ts';
const source=execFileSync('git',['-C',join(root,'system-testing'),'show',`${pins.repositories['system-testing'].git_commit}:${sourceRelative}`]);
const text=source.toString();
// Source identity is pinned by the capture. These reviewed control-flow witnesses
// establish what the native visible receipt meant at collection time.
for(const fragment of ['const display = await ports.observe(', '!display.passed ||', 'display.panelAttempt !== owned.attemptId',
 'display.buildKey !== owned.buildKey', 'if (after.revision !== revision)', 'return finish({ status: "visible", receipt });'])
 if(!text.includes(fragment))throw Error('original native renderer lacks reviewed DOM/revision receipt guarantees');
const require=createRequire(join(root,'host','package.json')), {build}=require('esbuild');
const checker=join(root,'system-testing-followup/skills/system-testing/presentation-managed.ts');
const worlds=join(root,'system-testing-followup/skills/system-testing/open-goal-local-presentation-worlds.ts');
const render=join(root,'base/skills/visualize/preparePresentation/prepareSection/render.ts');
const addressing='/home/werg/vibestudio/node_modules/@vibestudio/content-addressing/src/index.ts';
const bundled=await build({stdin:{contents:`export {checkPresentation} from ${JSON.stringify(checker)};export {openLocalPresentationWorlds} from ${JSON.stringify(worlds)};`,resolveDir:root},bundle:true,platform:'node',format:'esm',write:false,metafile:true,alias:{'@workspace-skills/visualize/render':render,'@vibestudio/content-addressing':addressing}});
const bundle=await store('checker.bundle.mjs',bundled.outputFiles[0].contents);
const {checkPresentation,openLocalPresentationWorlds}=await import(pathToFileURL(join(out,bundle.path)));
const world=openLocalPresentationWorlds.find(item=>item.id===prior.caseId);
if(!world || world.goal!==prior.goal || world.expected.target!==prior.current?.id || !world.expected.requirements)throw Error('archived task identity differs from authored application world');
const prepared=checkPresentation(world.expected.requirements,prior.report);
if(!prepared?.length)throw Error('corrected application checker rejects actual presentation');
const previews=prior.journal.filter(row=>row.service==='presentationRecords' && row.method==='preview' && row.result?.status==='visible');
if(previews.length!==1)throw Error('missing unique original native DOM-verified preview');
const receipt=previews[0].result.receipt;
if(!same(previews[0].args,[prior.current.id,prior.current.revision,prior.report]) ||
 !same(receipt,prior.preview) || !same(receipt,native.report?.receipt) || native.report.status!=='visible' ||
 receipt.source!==prior.current.id || receipt.revision!==prior.current.revision ||
 !same(receipt.viewIds,prepared.map(view=>view.id)) || !same(receipt.mdxDigests,prepared.map(view=>sha(view.mdx))))
 throw Error('actual DOM-verified native receipt differs from application values, MDX or revision');
const sources=[await store('original-native-renderer.ts',source),bundle,
 await store('assessor.mjs',await readFile(fileURLToPath(import.meta.url)))];
for(const [index,path]of Object.keys(bundled.metafile.inputs).filter(path=>path!=='<stdin>').entries())
 sources.push(await store(`checker-source-${index}-${basename(path)}`,await readFile(resolve(path))));
const proof={schema:'natlang.live-outcome-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),
 goal:prior.goal,entry:native.entry,actualResult:native.report,passed:true,
 checks:{...prior.checks,independentlyVerifiedPresentationDom:true},
 rationale:'Corrected typed presentation validation accepts the exact requested values and grounded optional title. The original native visible receipt proves actual DOM observation passed at the owned attempt/build and retained revision; MDX digests match corrected rendering. Original negative outcome remains immutable.',
 checker_sources:sources,evidence:[await store('native-preview-witness.json',JSON.stringify({originalCode:pins.repositories['system-testing'],nativePreview:previews[0],current:prior.current,prepared,priorChecks:prior.checks},null,2))]};
const completed=await store('completed-outcome.json',JSON.stringify(proof,null,2)+'\n');
await store('assessment.json',JSON.stringify({schema:'natlang.live-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),rationale:proof.rationale,completed_outcome:completed},null,2)+'\n');
console.log(JSON.stringify({task_id:task.taskId,completed_outcome:completed,assessment:join(out,'assessment.json'),teacher_rerun:false}));
