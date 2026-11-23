#!/usr/bin/env node
/** Select actual successful subtrees using independent native result bytes, never call names alone. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import {join,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {isDeepStrictEqual as same} from 'node:util';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const ref=value=>{const bytes=JSON.stringify(value);return {complete:true,bytes:Buffer.byteLength(bytes),hash:sha(bytes)}};
const args=Object.fromEntries(process.argv.slice(2).reduce((rows,value,index,all)=>index%2===0?[...rows,[value.slice(2),all[index+1]]]:rows,[]));
for(const key of ['packet','catalog','out'])if(!args[key])throw Error(`missing --${key}`);
const raw=await readFile(args.packet),packet=JSON.parse(args.packet.endsWith('.gz')?gunzipSync(raw):raw);
const task=(await readFile(args.catalog,'utf8')).trim().split('\n').map(JSON.parse).find(row=>row.scenario===packet.test.name);
const d=packet.execution.diagnostics,proof=d.openGoalVerification,native=d.natlangTeacher;
if(!task||task.split!=='train'||packet.result.passed!==false)throw Error('requires original failed training capture');
const contracts={assetHistory:{discover:'boundAssetHistory',listFiles:'boundAssetHistory',inspect:'preservedOwnedAssetBasis'},
 tendrilRouteRead:{listBoards:'discoveredOwnedTendrilRoutes',readBoard:'joinedDurableTendrilAudience'}};
const selected=[];
const descendants=id=>native.calls.filter(call=>call.record.parent_call_id===id).flatMap(call=>[call,...descendants(call.callId)]);
const checkedEffect=effect=>{
 const check=contracts[effect.service]?.[effect.method];
 return check&&proof.checks[check]===true&&(proof.journal??[]).some(row=>row.service===effect.service&&row.method===effect.method&&same(effect.args,ref(row.args))&&same(effect.result,ref(row.result)));
};
for(const call of native.calls){
 const record=call.record;
 if(record.outcome!=='done'||record.executor?.kind==='crisp'||
  [call,...descendants(call.callId)].some(child=>(child.record.effects??[]).some(effect=>!checkedEffect(effect))))continue;
 const effects=record.effects??[];
 const matches=[];
 for(const effect of effects){
  const check=contracts[effect.service]?.[effect.method];
  if(!check || proof.checks[check]!==true)continue;
  for(const row of proof.journal??[])if(row.service===effect.service&&row.method===effect.method &&
   same(effect.args,ref(row.args))&&same(effect.result,ref(row.result)))matches.push({row,check});
 }
 // A native read wrapper is correct only when both actual effect coordinates and
 // its returned value equal the independently retained native service result.
 if(effects.length===1 && matches.some(({row})=>same(record.output,ref(row.result)))){
  selected.push({call_id:call.callId,checks:[...new Set(matches.map(({check})=>check))],
   evidenceKind:'exact-native-read-result',output:record.output,effects});continue;
 }
 if(proof.checks.boundAssetHistory===true && proof.checks.preservedOwnedAssetBasis===true){
  const versions=proof.journal.find(row=>row.service==='assetHistory'&&row.method==='discover')?.result??[];
  for(const version of versions){
   if(!same(record.inputs?.version,ref(version)))continue;
   const files=proof.journal.find(row=>row.service==='assetHistory'&&row.method==='listFiles'&&same(row.args,[version.id]))?.result;
   if(!Array.isArray(files))continue;
   const inspected=files.map(file=>proof.journal.find(row=>row.service==='assetHistory'&&row.method==='inspect'&&same(row.args,[version.id,file.id]))?.result);
   if(inspected.some(item=>!item))continue;
   const manifest=inspected.find(item=>item.id===version.dossier)?.manifest;
   if(manifest&&same(record.output,ref({version,files,manifest,inspected})))selected.push({call_id:call.callId,
    checks:['boundAssetHistory','preservedOwnedAssetBasis'],evidenceKind:'reconstructed-native-asset-version',output:record.output});
  }
 }
 if(proof.checks.joinedAssetIntegrityFinding===true && Array.isArray(proof.findings) &&
  proof.findings.some(finding=>same(record.output,ref(finding))))selected.push({call_id:call.callId,
   checks:['joinedAssetIntegrityFinding'],evidenceKind:'independently-checked-asset-finding',output:record.output});
}
if(!selected.length)throw Error('no independently proved native read subtrees');
const out=resolve(args.out);await mkdir(out,{recursive:false});
const source=await readFile(fileURLToPath(import.meta.url));await writeFile(join(out,'assessor.mjs'),source);
const assessment={schema:'natlang.live-reassessment/1',task_id:task.taskId,original_sha256:sha(raw),
 rationale:'Only original completed subtrees whose output and actual readonly effect coordinates exactly match independently retained native journal values, reconstructed native typed values, or an independently checked native finding are supervised. The original failed goal and every unapproved/error decision remain retained.',
 assessor:{path:'assessor.mjs',sha256:sha(source)},invocations:selected};
await writeFile(join(out,'assessment.json'),JSON.stringify(assessment,null,2)+'\n');
console.log(JSON.stringify({task_id:task.taskId,subtrees:selected.length,assessment:join(out,'assessment.json'),teacher_rerun:false}));
