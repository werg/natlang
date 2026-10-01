/** Replace retired authored templates; execution observations must be recollected, never invented. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {dirname} from 'node:path';
import {createHash} from 'node:crypto';
import {AUTHORED_IMPROVER} from '../../dist/improvement/authored-source.js';
import {fingerprint} from '../../dist/adaptation/identity.js';
export function migrateLifecycle(row){
 const copy=structuredClone(row);let changes=0,schemaChanges=0;
 const visit=value=>{
  if(!value||typeof value!=='object')return;
  if(value.function?.name==='eval'&&value.function.parameters?.properties?.code&&(!value.function.parameters.properties.finish||!value.function.parameters.properties.finish.description?.includes('final expression'))){
   value.function.parameters.properties.finish={type:'boolean',description:'Compute and finish the fresh typed final expression or explicit return in this eval. Omit or use false for inspection or staging.'};
   value.function.description='Run TypeScript in this call\'s persistent scope. A final expression inspects; a typed return stages; finish:true completes the fresh typed final expression or explicit return in one action.';
   schemaChanges++;
  }
  if(value.files&&typeof value.files==='object'&&typeof value.files['improveStep.nl']==='string'&&('improveStep/measureBaseline.nl'in value.files||'improveStep/selectCandidate.nl'in value.files||'improveStep/runExperiment.nl'in value.files||value.files['improveStep/planExperiment.nl']?.includes('frame: SearchFrame')||value.files['improveStep/planExperiment.nl']?.includes('returns: ExperimentPlan')||value.files['improveStep/planExperiment.nl']?.includes('context.brief'))){
   const prior=fingerprint(value.files);
   // The template is a training task, not an execution replay. Retain the old snapshot separately for audit.
   value.retiredAuthoredIdentity=prior;
   value.files=structuredClone(AUTHORED_IMPROVER);
   if('identity'in value)value.identity=fingerprint(value.files);
   changes++;
  }
  for(const child of Object.values(value))if(child&&typeof child==='object')visit(child);
 };
 visit(copy);
 if(changes||schemaChanges){
  copy.migration={version:'natlang.authored-lifecycle/2',sourceId:row.id,sourceHash:createHash('sha256').update(JSON.stringify(row)).digest('hex'),changes,schemaChanges,disposition:'recollect-required',reason:'Retired experiment plumbing and wrapped plans replaced by a primitive semantic hypothesis with a visible brief and one exact experiment; eval schema includes atomic completion. Old actions and observations are historical context, not executions of the new source.'};
  copy.training_admission={kind:'migration-recollection-required',approved:false,reason:copy.migration.reason};
  copy.trace_admission={admitted:false};
 }
 return {row:copy,changes,schemaChanges};
}
if(process.argv[1]&&new URL(import.meta.url).pathname===process.argv[1]){
 const [input,output]=process.argv.slice(2);if(!output)throw Error('usage: migrate-improver-lifecycle.mjs INPUT_JSONL OUTPUT_JSONL');
 const original=await readFile(input,'utf8'),rows=original.trim().split('\n').filter(Boolean).map(JSON.parse).map(migrateLifecycle);
 await mkdir(dirname(output),{recursive:true});await writeFile(output,rows.map(result=>JSON.stringify(result.row)).join('\n')+'\n');
 const manifest={version:'natlang.authored-lifecycle/2',input,output,inputHash:createHash('sha256').update(original).digest('hex'),rows:rows.length,migrated:rows.filter(result=>result.changes||result.schemaChanges).length,recollect:rows.filter(result=>result.changes||result.schemaChanges).map(result=>result.row.id),positiveSFT:false};
 await writeFile(output+'.manifest.json',JSON.stringify(manifest,null,2)+'\n');console.log(JSON.stringify({rows:manifest.rows,migrated:manifest.migrated,positiveSFT:false}));
}
