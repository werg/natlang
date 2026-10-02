import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {publish} from '../scripts/self-improvement/publish-optimizer-training.mjs';
import {digest} from '../scripts/self-improvement/replay-runtime.mjs';

test('optimizer publication preserves other lanes, carries forward decisions, and excludes transfer',async()=>{
 const repo=await mkdtemp(join(tmpdir(),'natlang-publication-test-'));
 try{
  const base=join(repo,'data/teacher/self-improvement');await mkdir(base,{recursive:true});
  const legacy={path:'unchanged.jsonl',sha256:'existing',lane:'other-producer'};
  await writeFile(join(base,'current-manifest.json'),JSON.stringify({artifacts:[legacy],historical_disposition:'retained'}));
  const source=async(id,content=id,cohort='development')=>{
   const directory=join(repo,'inputs',id+'-'+content+'-'+cohort);await mkdir(directory,{recursive:true});
   const row={id,split:'train',source:'original',messages:[{role:'user',content}],tools:[],target:{role:'assistant',content:'done'},outcome:{accepted:true},training_admission:{approved:true,kind:'exact-native-runtime-oracle'},trace_admission:{admitted:true},task:{program_ir:{semantics:{inputs:{},evaluation_fixture:{caseDefinition:{cohort,cases:[]}}}}}};
   const text=JSON.stringify(row)+'\n';await writeFile(join(directory,'training-turns.jsonl'),text);
   const manifest=join(directory,'manifest.json');await writeFile(manifest,JSON.stringify({rows:1,sha256:digest(text)}));return manifest;
  };
  const first=await publish(repo,[await source('one')]);
  const second=await publish(repo,[await source('two')]);assert.equal(second.rows,2);
  const registry=JSON.parse(await readFile(join(base,'current-manifest.json'),'utf8'));
  assert.deepEqual(registry.artifacts[0],legacy);assert.equal(registry.historical_disposition,'retained');
  assert.equal(registry.superseded_artifacts[0].path,first.path);
  assert.equal((await readFile(join(repo,second.path),'utf8')).trim().split('\n').length,2);
  assert.equal((await publish(repo,[await source('two')])).sha256,second.sha256);
  const conflict=await source('one','altered');await assert.rejects(()=>publish(repo,[conflict]),/Conflicting recorded optimizer decision/);
  assert.equal((await publish(repo,[await source('held','held','transfer')])).published,false);
  assert.equal(JSON.parse(await readFile(join(base,'current-manifest.json'),'utf8')).artifacts.at(-1).sha256,second.sha256);
 }finally{await rm(repo,{recursive:true,force:true});}
});
