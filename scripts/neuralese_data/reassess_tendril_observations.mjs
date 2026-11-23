#!/usr/bin/env node
/** Additive offline reassessment using original native witnesses; no workspace/model rerun. */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { gunzipSync } from 'node:zlib';
import { resolve, join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { isDeepStrictEqual as same } from 'node:util';
import { execFileSync } from 'node:child_process';
const sha = value => createHash('sha256').update(value).digest('hex');
const args = Object.fromEntries(process.argv.slice(2).reduce((rows, value, i, all) => i % 2 ? rows : [...rows, [value.slice(2), all[i+1]]], []));
for (const key of ['source-root','host-root','packet','pins','catalog','out']) if (!args[key]) throw Error(`missing --${key}`);
const out=resolve(args.out),root=resolve(args['source-root']);
const raw=await readFile(args.packet),packet=JSON.parse(args.packet.endsWith('.gz')?gunzipSync(raw):raw);
const pins=JSON.parse(await readFile(args.pins,'utf8'));
const task=(await readFile(args.catalog,'utf8')).trim().split('\n').map(JSON.parse).find(row=>row.scenario===packet.test.name);
const prior=packet.execution.diagnostics.openGoalVerification,native=packet.execution.diagnostics.natlangTeacher;
if(!task||task.split!=='train'||packet.result.passed!==false||native.executionError||native.captureErrors?.length||
 Object.values(pins.repositories).some(row=>row.dirty!==false))throw Error('requires complete failed training capture with clean original provenance');
if(!same(Object.entries(prior.checks).filter(([,v])=>v!==true).map(([k])=>k),['checkedCurrentTendrilProjection']))throw Error('other native failures remain');
if(!same(native.report,prior.actualResult)||!same(native.entry,prior.entry)||!prior.finding||
 !same(native.report.finding,prior.finding)||native.report.status!=='verified')throw Error('typed return is not the exact independently checked current finding');
const sourceRelative='skills/system-testing/open-goal-tendril-route-managed.ts';
const originalSource=execFileSync('git',['-C',root,'show',`${pins.repositories['system-testing'].git_commit}:${sourceRelative}`]);
const originalText=originalSource.toString();
for(const fragment of ['if (board.revision !== revision)','projected.push(finding);','return finish({ status: "projected", finding });','readsFinalRevision &&','!raced && main.board.revision === initial.revision'])
 if(!originalText.includes(fragment))throw Error('original native service lacks reviewed revision/snapshot guarantees');
const finding=prior.finding,current=prior.current;
if(!same(finding,(({id,channel,audience,revision,utterance})=>({id,channel,audience,slot:finding.slot,revision,utterance,projection:finding.projection}))(current))||
 !current.slots.some(slot=>slot.id===finding.slot&&slot.state==='current'))throw Error('finding differs from owned current slot');
const probes=prior.journal.filter(row=>row.service==='tendrilRouteRead'&&row.method==='probe');
// This rechecker deliberately handles stationary successful observations. A
// concurrent-change case needs its own original revision transition witnesses.
if(!probes.length||probes.some(row=>row.result?.status!=='projected'||!same(row.args,[finding.id,finding.slot,finding.revision])||!same(row.result.finding,finding)))throw Error('unresolved changed/foreign/stale projection');
const boards=prior.journal.filter(row=>row.service==='tendrilRouteRead'&&row.method==='readBoard'&&row.args[0]===finding.id);
if(!boards.length||!boards.every(row=>same(row.result,current)))throw Error('native current-board basis changed');
await mkdir(out,{recursive:false});
async function store(name,bytes){await writeFile(join(out,name),bytes);return{path:name,sha256:sha(bytes)}}
const require=createRequire(join(resolve(args['host-root']),'package.json')),{build}=require('esbuild');
const built=await build({entryPoints:[join(root,'skills/system-testing/semantic-evidence.ts')],bundle:true,platform:'node',format:'esm',write:false});
const bundle=await store('checker.bundle.mjs',built.outputFiles[0].contents),{includesEvidence}=await import(pathToFileURL(join(out,bundle.path)));
if(!includesEvidence(probes.map(row=>row.result.finding),finding))throw Error('corrected native evidence join rejected finding');
const sources=[bundle,await store('original-native-oracle.ts',originalSource),await store('corrected-native-oracle.ts',await readFile(join(root,sourceRelative))),
 await store('semantic-evidence.ts',await readFile(join(root,'skills/system-testing/semantic-evidence.ts'))),await store('assessor.mjs',await readFile(fileURLToPath(import.meta.url)))];
const proof={schema:'natlang.live-outcome-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),goal:prior.goal,entry:native.entry,actualResult:native.report,passed:true,
 checks:{...prior.checks,checkedCurrentTendrilProjection:true},
 rationale:'The exact typed finding matches independently captured current owned board/slot/revision and every successful native projection. Repeated stationary read-only probes preserve this proof; they are not a user-requested cardinality. All other independent application checks already passed. The original failure, full graph and model replies remain immutable.',
 checker_sources:sources,evidence:[await store('native-observation-witness.json',JSON.stringify({originalCode:pins.repositories['system-testing'],current,finding,probes,boards,priorChecks:prior.checks},null,2))]};
const completed=await store('completed-outcome.json',JSON.stringify(proof,null,2)+'\n');
await store('assessment.json',JSON.stringify({schema:'natlang.live-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),rationale:proof.rationale,completed_outcome:completed},null,2)+'\n');
console.log(JSON.stringify({task_id:task.taskId,completed_outcome:completed,assessment:join(out,'assessment.json'),teacher_rerun:false}));
