/** Fresh development tasks for source repair, restructuring and further cost improvement. */
import {mkdir,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {loadVirtualNatlang,compileVirtualProject} from '../../dist/runtime/virtual-project.js';
import * as runtime from '../../dist/improvement/target-runtime.js';
import {digest} from './replay-runtime.mjs';
const header='---\nkind: directory-reducer\nargs: {}\nreturns: string\n---\n';
const type='type Root={files(pattern:string):Promise<{relativePath:string;readText():Promise<string>}[]>;file(path:string):{readText():Promise<string>;writeText(text:string):Promise<void>}};';
const domains=[
 ['active-urgency','currently requires immediate intervention','urgent','The outage is active. Intervene immediately.','The old urgent incident is resolved; next week is fine.','Immediate intervention is required despite the earlier low priority.','Urgency was suggested, but the final decision is routine scheduling.'],
 ['explicit-consent','has currently valid explicit consent','consent','I explicitly consent to this use.','Consent was requested, but I decline it.','The earlier refusal is superseded: I now explicitly consent.','I previously consented; I now withdraw that consent.'],
 ['completed-delivery','is finally confirmed delivered','delivered','Delivery is confirmed complete.','The delivered status was an error; the parcel is still missing.','After the earlier failure, receipt is now confirmed.','Delivery was predicted, but has not happened.'],
 ['active-license','has an active granted license','license','The license is granted and remains valid.','A license request is pending; none has been granted.','The earlier denial is superseded by a valid license grant.','The granted license has been revoked.'],
 ['final-cancellation','is finally cancelled','cancel','Cancel this booking; that is my final choice.','Cancellation was discussed. Keep the booking active.','Earlier I wanted to keep it; I now choose cancellation.','The cancellation request is withdrawn; retain the booking.'],
 ['overall-recommendation','recommends the item overall','recommend','I recommend this dependable item.','Nice packaging, but I cannot recommend this unreliable item.','Not perfect, but I would choose it again and recommend it.','I recommend the earlier model; avoid this one.'],
 ['authorized-access','has currently authorized access','access','Access is explicitly authorized and active.','Access was requested but authorization was denied.','The earlier restriction is superseded by explicit access authorization.','Access was authorized earlier, but is now revoked.'],
 ['resolved-support','is currently resolved','resolved','The issue is resolved and the customer confirms the fix.','Marked resolved by mistake; the problem persists.','The earlier failed repair is superseded by a confirmed resolution.','A resolution was proposed, but the customer says it still fails.'],
 ['confirmed-attendance','has confirmed attendance','attend','I confirm that I will attend.','I considered attending, but my final answer is no.','The earlier decline is superseded: I now confirm attendance.','My attendance confirmation is withdrawn; I will not attend.'],
 ['final-renewal','has a final confirmed renewal','renew','The renewal is confirmed and remains active.','A renewal offer was sent but declined.','The earlier cancellation is superseded by a confirmed renewal.','Renewal was initially approved but subsequently cancelled.']
];
export const SLATE_FAMILIES=[...domains.map(row=>row[0]),'identity-join','latest-revision','ranked-selection','exact-multi-edit','corrected-extraction','semantic-routing'];
const compiledSources=new Set();
function compile(row){for(const files of [row.files,row.reference]){const identity=digest(JSON.stringify(files));if(compiledSources.has(identity))continue;loadVirtualNatlang(files,'solve.nl');const built=compileVirtualProject({files},runtime,{constrained:true,target:'node'});if(!built.ok)throw Error(row.id+': '+JSON.stringify(built.diagnostics));compiledSources.add(identity);}}
export function trainingSlate({variants=12,seed=20261002}={}){
 if(!Number.isSafeInteger(variants)||variants<1||variants>256)throw Error('variants must be a finite integer in [1,256]');
 const rows=[];
 for(const family of SLATE_FAMILIES)for(let variant=0;variant<variants;variant++){
  const tag=digest(family+':'+seed+':'+variant).slice(0,10),id='optimizer-curriculum-'+family+'-'+tag;
  let goal,helper;const examples=[];
  const semantic=domains.find(row=>row[0]===family)??domains[1];
  for(let index=0;index<5;index++){
   const count=index===0?2:3+(variant+index)%4;
   const records=Array.from({length:count},(_,i)=>({id:tag+'-'+String.fromCharCode(122-i),amount:17+(variant*31+index*29+i*43)%701,revision:1,score:10+(i*7+variant)%29,note:'notes/'+(index===0?'':i%2?'east/':'west/')+tag+'-'+i+'.txt',yes:index===0?i===0:(i+variant+index)%3!==0}));
   const folder={'README.md':'Preserve this unrelated file.\n','archive/irrelevant.txt':'Archived material is not an input.\n'};
   if(family==='exact-multi-edit'){
    const changes=(index===0?records.slice(0,1):records).map((record,i)=>({path:'documents/'+record.id+'.txt',find:index===0?'draft':'  draft '+i+'\n',replace_with:index===0?'final':'\tfinal '+i+'\n\n'}));
    for(const change of changes)folder[change.path]='prefix\n'+change.find+'\nsuffix\n';
    folder['change-request.json']=JSON.stringify({changes});
    const effects={...folder};for(const change of changes)effects[change.path]=effects[change.path].replace(change.find,change.replace_with);
    examples.push({folder,effects});continue;
   }
   if(family==='corrected-extraction'){
    const owners=['Cedar','Maple','Willow','Aspen'],places=['Workshop','Depot','Atrium','Annex'];
    const extracted=records.filter(record=>record.yes).map((record,i)=>({id:record.id,owner:owners[(i+variant)%4],place:places[(i+index)%4]}));
    const inputs=records.map((record,i)=>{const value=extracted.find(value=>value.id===record.id);return {...record,text:value?(index===0?'Confirmed assignment: ':'Earlier assignment: Pine at Garage. Final confirmed assignment: ')+value.owner+' at '+value.place+'.':'Only proposed: Pine at Garage; not confirmed.'};});
    folder['records.json']=JSON.stringify(inputs.map(({yes,...record})=>record));
    examples.push({folder,effects:{'report.json':JSON.stringify({assignments:extracted.sort((a,b)=>a.id.localeCompare(b.id))})}});continue;
   }
   if(family==='semantic-routing'){
    const labels=['billing','technical','account'];
    const inputs=records.map((record,i)=>({...record,category:labels[(i+variant)%3],text:[index===0?'Please correct the invoice charge.':'The old login issue is fixed; my current problem is an incorrect invoice charge.',index===0?'Please fix the malfunctioning device.':'Billing is resolved; the device currently fails to start.',index===0?'Please restore my locked account.':'The equipment is fixed; I now need my locked account restored.'][(i+variant)%3]}));
    folder['records.json']=JSON.stringify(inputs.map(({yes,category,...record})=>record));
    examples.push({folder,effects:{'report.json':JSON.stringify({routes:inputs.map(record=>({id:record.id,category:record.category})).sort((a,b)=>a.id.localeCompare(b.id))})}});continue;
   }
   const texts=records.map((record,i)=>(index===0&&!record.yes?'No affirmative decision exists for this record; it is excluded.':semantic[index===0?3:(record.yes?5:6)])+(variant%2?' This is the current decision.':' Final status is as stated above.'));
   for(let i=0;i<records.length;i++)folder[records[i].note]=texts[i];
   let inputs=records;
   if(family==='latest-revision'&&index>0){inputs=records.flatMap((record,i)=>[{...record,revision:4,note:record.note},{...record,revision:1,note:'notes/old/'+record.id+'.txt'}]);for(const record of records)folder['notes/old/'+record.id+'.txt']=record.yes?semantic[6]:semantic[5];inputs.reverse();}
   folder['records.json']=JSON.stringify(inputs.map(({yes,...record})=>record));
   const selected=records.filter(record=>record.yes);const ordered=family==='ranked-selection'?selected.sort((a,b)=>b.score-a.score||a.id.localeCompare(b.id)):selected.sort((a,b)=>a.id.localeCompare(b.id));
   examples.push({folder,effects:{'report.json':JSON.stringify({selected:ordered.map(record=>record.id),total:ordered.reduce((sum,record)=>sum+record.amount,0)})}});
  }
  if(family==='exact-multi-edit'){
   goal='Read change-request.json. Apply EVERY change in order as one exact replacement, preserving the supplied strings, whitespace, final newlines and every other file. Return "done".';
   helper=`export async function run(folder:Root):Promise<string>{const request=JSON.parse(await folder.file('change-request.json').readText());const change=request.changes[0];const text=await folder.file(change.path).readText();await folder.file(change.path).writeText(text.replace(change.find.trim(),change.replace_with.trim()));return 'done';}`;
  }else if(family==='corrected-extraction'){
   goal='Read records.json. Extract each final confirmed assignment as {id,owner,place}, using the FINAL correction, ignoring proposals and superseded assignments. Write report.json as JSON.stringify({assignments: recordsSortedById}). Preserve other files; return "done".';
   helper=`export async function run(folder:Root):Promise<string>{const records=JSON.parse(await folder.file('records.json').readText());const assignments=[];for(const record of records){const text=record.text;const value=await nl<{owner:string;place:string}|null>\`Extract the first named owner and place if an assignment is confirmed, otherwise null.\`(text);if(value)assignments.push({id:record.id,...value});}await folder.file('report.json').writeText(JSON.stringify({assignments:assignments.sort((a,b)=>a.id.localeCompare(b.id))}));return 'done';}`;
  }else if(family==='semantic-routing'){
   goal='Read records.json. Route the CURRENT unresolved request by meaning to billing, technical or account. Ignore resolved earlier topics. Write report.json as JSON.stringify({routes: [{id,category}] sortedById}). Preserve all other files; return "done".';
   helper=`export async function run(folder:Root):Promise<string>{const records=JSON.parse(await folder.file('records.json').readText());const routes=[];for(const record of records){const text=record.text;const category=await nl<'billing'|'technical'|'account'>\`Classify the first topic mentioned in text.\`(text);routes.push({id:record.id,category});}await folder.file('report.json').writeText(JSON.stringify({routes:routes.sort((a,b)=>a.id.localeCompare(b.id))}));return 'done';}`;
  }else{
   const criterion=semantic[1];
   goal=`Read records.json and join each row to its explicit note path (including nested paths). Select records whose whole note ${criterion}, respecting final decisions, negation and withdrawals. ${family==='latest-revision'?'Use only the highest revision for each ID. ':''}Write report.json as JSON.stringify({selected: ${family==='ranked-selection'?'IDs ordered by descending score, ties by ID':'sortedUniqueIds'},total: exactIntegerSumOfSelectedAmounts}). Preserve all other files and return "done". Reduce unnecessary semantic delegation after correctness is established.`;
   const text=family==='identity-join'?"const notes=records.map(record=>record.note).sort();":"";
   const read=family==='identity-join'?"await folder.file(notes[index]).readText()":"await folder.file(record.note).readText()";
   helper=`export async function run(folder:Root):Promise<string>{const records=JSON.parse(await folder.file('records.json').readText());${text}const selected:string[]=[];let total=0;for(let index=0;index<records.length;index++){const record=records[index];const text=${read};const yes=await nl<boolean>\`Does the text contain any positive mention of ${semantic[2]}, including earlier or proposed states?\`(text);if(yes){selected.push(record.id);total+=record.amount;}}await folder.file('report.json').writeText(JSON.stringify({selected:selected.sort(),total}));return 'done';}`;
  }
  const files={'solve.nl':header+'Run workflow.run(folder) and return its result.\n','solve/workflow.ts':"import {nl} from '@natlang/node';\n"+type+'\n'+helper+'\n'};
  const reference={...files,'solve.nl':header+goal+' Judge the visible semantic inputs in this call, then perform exact joins, aggregation and writes.\n'};
  const row={version:'natlang.improvement-case/1',id,family,cohort:'development',files,reference,contract:{entry:'solve.nl',exportName:'default',programId:id},cases:examples.map((example,i)=>({id:id+'-'+i,group:'optimizer-curriculum:'+family+':'+tag+':'+i,split:i<3?'train':'validation',args:[],folder:example.folder,expected:'done',expectedFiles:{...example.folder,...example.effects}})),policy:{maxExperiments:3,maxPopulation:4,strategy:variant%2?'gepa':'adaptive',mode:'structural',objective:variant%3===0?'quality':'model-calls',goal,allowedFiles:['solve.nl','solve/workflow.ts','solve/judge.nl','solve/join.ts','solve/aggregate.ts','types.ts']},budget:{maxModelCalls:180,maxRollouts:32,maxProposals:3,maxElapsedMs:3600000},sourceGroups:['optimizer-curriculum:'+family+':'+tag],incidents:[{id:'reconstructed-'+family,split:'train',mechanisms:['incorrect semantic criterion','wrong evidence scope','excess delegation',family],reason:'Fresh reconstruction of generated-data failure mechanisms; not claimed as a reproduced historical student failure.'}],provenance:{kind:'independently-authored-development-curriculum',seed,variant,oracle:'Latent labelled decisions and exact file effects; labels absent from runtime folders',positiveSFT:false,referenceHiddenFromOptimizer:true}};
  compile(row);rows.push(row);
 }
 return rows;
}
if(process.argv[1]===new URL(import.meta.url).pathname){const output=resolve(process.argv[2]),variants=Number(process.argv[3]??12),seed=Number(process.argv[4]??20261002);const rows=trainingSlate({variants,seed});await mkdir(output,{recursive:true});await writeFile(join(output,'cases.jsonl'),rows.map(JSON.stringify).join('\n')+'\n');await writeFile(join(output,'manifest.json'),JSON.stringify({schema:'natlang.optimizer-training-slate/1',tasks:rows.length,families:SLATE_FAMILIES,variants,seed,sha256:digest(rows.map(JSON.stringify).join('\n')+'\n'),positiveSFT:false,design:'Development-only native source optimization; independent gold and source groups; no existing transfer/confirmation cases reused.'},null,2)+'\n');console.log(JSON.stringify({tasks:rows.length,families:SLATE_FAMILIES.length}));}
