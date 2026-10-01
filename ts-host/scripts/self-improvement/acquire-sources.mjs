/** Bounded, pinned primary-source samples; reference/code acquisition never means SFT admission. */
import {mkdir,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
const output=process.argv[2];if(!output)throw Error('usage: acquire-sources.mjs OUTPUT_DIR');
await mkdir(output,{recursive:true});const records=[];
for(const [id,url,license,role]of [
 ['gepa-code','https://raw.githubusercontent.com/gepa-ai/gepa/main/src/gepa/optimize_anything.py','MIT','reflective source optimization reference'],
 ['gepa-license','https://raw.githubusercontent.com/gepa-ai/gepa/main/LICENSE','MIT','license evidence'],
 ['swe-bench-train','https://datasets-server.huggingface.co/rows?dataset=princeton-nlp%2FSWE-bench_Lite&config=default&split=dev&offset=0&length=3','dataset card and original repository licenses required','repair reconstruction sample']]){
 try{
  const response=await fetch(url,{signal:AbortSignal.timeout(30000)});if(!response.ok)throw Error('HTTP '+response.status);
  const text=await response.text();if(Buffer.byteLength(text)>512000)throw Error('bounded acquisition exceeds 512 KB');
  const sha256=createHash('sha256').update(text).digest('hex');await writeFile(join(output,id+'.txt'),text);
  const row={id,url,sha256,bytes:Buffer.byteLength(text),license,role,admission:'reference-only',retrievedAt:new Date().toISOString()};
  if(id==='swe-bench-train'){
   const sample=JSON.parse(text).rows.map(({row})=>({id:row.instance_id,repository:row.repo,baseRevision:row.base_commit,originalSplit:'dev',issue:row.problem_statement,oraclePatch:row.patch,testPatch:row.test_patch,disposition:'needs-native-project-and-independent-fixture-reconstruction',source:url,license}));
   await writeFile(join(output,'repair-samples.jsonl'),sample.map(JSON.stringify).join('\n')+'\n');row.samples=sample.length;
  }
  records.push(row);
 }catch(error){records.push({id,url,disposition:'acquisition-failed',error:String(error)});}
}
await writeFile(join(output,'manifest.json'),JSON.stringify({version:'natlang.primary-source-acquisition/1',records,rules:['No gold patches in improver contexts','Original held-out records never become train examples','Only executable independently verified converted cases enter training']},null,2));
console.log(JSON.stringify(records.map(row=>({id:row.id,bytes:row.bytes,samples:row.samples,disposition:row.disposition??row.admission}))));
