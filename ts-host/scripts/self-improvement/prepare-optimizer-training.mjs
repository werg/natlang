/** Merge only verified positive optimizer decisions; exclude all transfer-family trajectories. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join,dirname,resolve} from 'node:path';
import {digest} from './replay-runtime.mjs';
export function trainingEligible(row){
 const definition=row.task?.program_ir?.semantics?.evaluation_fixture?.caseDefinition;
 if(definition?.cohort==='transfer')return false;
 if(row.split!=='train'||row.outcome?.accepted!==true||row.training_admission?.approved!==true||row.training_admission.kind!=='exact-native-runtime-oracle'||row.trace_admission?.admitted!==true)throw Error('Unverified optimizer target: '+row.id);
 if(definition?.cases?.some(example=>example.split==='test'))throw Error('Sealed cases in training fixture: '+row.id);
 return true;
}
export async function prepare(output,manifests){
 const rows=new Map(),sources=[],excluded=[];
 for(const path of manifests){
  const manifest=JSON.parse(await readFile(path,'utf8')),text=await readFile(join(dirname(path),'training-turns.jsonl'),'utf8');
  if(digest(text)!==manifest.sha256)throw Error('Training artifact changed: '+path);
  const parsed=text.trim().split('\n').filter(Boolean).map(JSON.parse);
  if(parsed.length!==manifest.rows)throw Error('Training row count changed: '+path);
  for(const row of parsed){
   if(!trainingEligible(row)){excluded.push({id:row.id,reason:'transfer family never trains the optimizer'});continue;}
   if(rows.has(row.id)&&JSON.stringify(rows.get(row.id))!==JSON.stringify(row))throw Error('Conflicting decision identity: '+row.id);
   rows.set(row.id,row);
  }
  sources.push({path,sha256:manifest.sha256,rows:manifest.rows});
 }
 await mkdir(output,{recursive:true});const text=[...rows.values()].map(JSON.stringify).join('\n')+'\n';
 await writeFile(join(output,'training-turns.jsonl'),text);
 const corrections=[...rows.values()].filter(row=>{const inputs=row.task.program_ir.semantics.inputs;return (inputs.state?.history??inputs.request?.history??[]).at(-1)?.accepted===false;}).map(row=>row.id);
 const manifest={schema:'natlang.optimizer-distillation-input/1',rows:rows.size,sha256:digest(text),sources,excluded,correctionContextTurns:corrections,model:'Sharp-MiniCPM5-2B',experiment:'Frozen untrained versus trained optimizer; Bonsai executor unchanged; no transfer-family or confirmation targets train weights.'};
 await writeFile(join(output,'manifest.json'),JSON.stringify(manifest,null,2));return manifest;
}
if(process.argv[1]===new URL(import.meta.url).pathname){const output=resolve(process.argv[2]),manifests=process.argv.slice(3).map(path=>resolve(path));if(!manifests.length)throw Error('usage: prepare-optimizer-training.mjs OUTPUT MANIFEST...');console.log(JSON.stringify(await prepare(output,manifests)));}
