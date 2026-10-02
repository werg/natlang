/** Publish admitted optimizer turns into the same registry consumed by the general recipe. */
import {mkdir,readFile,writeFile,open,rename,mkdtemp,rm,cp} from 'node:fs/promises';
import {join,resolve,relative} from 'node:path';
import {tmpdir} from 'node:os';
import {prepare} from './prepare-optimizer-training.mjs';
import {digest} from './replay-runtime.mjs';
export async function publish(repo,manifests){
 repo=resolve(repo);const base=join(repo,'data/teacher/self-improvement');await mkdir(base,{recursive:true});
 const temporary=await mkdtemp(join(tmpdir(),'natlang-optimizer-publication-'));
 const lockPath=join(base,'current-manifest.lock'),lock=await open(lockPath,'wx');
 try{
  const prepared=await prepare(temporary,manifests);
  if(!prepared.rows)return {rows:0,published:false};
  let previousCorpus;
  try{const registry=JSON.parse(await readFile(join(base,'current-manifest.json'),'utf8'));previousCorpus=registry.artifacts.find(artifact=>artifact.lane==='native-program-improvement');}catch(error){if(error.code!=='ENOENT')throw error;}
  let previousDirectory;
  if(previousCorpus){
   const bytes=await readFile(join(repo,previousCorpus.path));if(digest(bytes)!==previousCorpus.sha256)throw Error('Published optimizer data changed');
   previousDirectory=resolve(repo,previousCorpus.path,'..');
   const rows=new Map(bytes.toString().trim().split('\n').filter(Boolean).map(JSON.parse).map(row=>[row.id,row]));
   for(const row of (await readFile(join(temporary,'training-turns.jsonl'),'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse)){
    const previous=rows.get(row.id);
    const action=value=>JSON.stringify([value.messages,value.tools,value.target,value.source,value.teacher_reasoning]);
    if(previous&&action(previous)!==action(row))throw Error('Conflicting recorded optimizer decision: '+row.id);
    rows.set(row.id,row);
   }
   const text=[...rows.values()].map(JSON.stringify).join('\n')+'\n';
   prepared.rows=rows.size;prepared.sha256=digest(text);
   prepared.correctionContextTurns=[...rows.values()].filter(row=>{const inputs=row.task.program_ir.semantics.inputs;return (inputs.state?.history??inputs.request?.history??[]).at(-1)?.accepted===false;}).map(row=>row.id);
   await writeFile(join(temporary,'training-turns.jsonl'),text);
  }
  const directory=join(base,'native-program-improvement',prepared.sha256);
  await mkdir(directory,{recursive:true});
  const immutable=async(path,bytes)=>{try{await writeFile(path,bytes,{flag:'wx'});}catch(error){if(error.code!=='EEXIST')throw error;if(digest(await readFile(path))!==digest(bytes))throw Error('Published corpus changed: '+path);}};
  const turns=join(directory,'verified-turns.jsonl');await immutable(turns,await readFile(join(temporary,'training-turns.jsonl')));
  // Keep source admission receipts with the published data, even if run folders are archived.
  await mkdir(join(directory,'source-manifests'),{recursive:true});
  if(previousDirectory&&previousDirectory!==directory)await cp(join(previousDirectory,'source-manifests'),join(directory,'source-manifests'),{recursive:true,errorOnExist:false,force:false});
  for(const source of prepared.sources){const bytes=await readFile(source.path);await immutable(join(directory,'source-manifests',digest(bytes)+'.json'),bytes);}
  const receipt={...prepared,...(previousCorpus?{retainedCorpus:previousCorpus}:{}),publication:'general-training-recipe',transferPolicy:'All transfer-family and sealed confirmation examples excluded'};
  const receiptBytes=JSON.stringify(receipt,null,2)+'\n',receiptPath=join(directory,'manifest.json');try{await readFile(receiptPath);}catch(error){if(error.code!=='ENOENT')throw error;await immutable(receiptPath,receiptBytes);}
  {
   const registryPath=join(base,'current-manifest.json');let registry;
   try{registry=JSON.parse(await readFile(registryPath,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;registry={schema:'natlang.current-improvement-corpus/1',artifacts:[]};}
   const path=relative(repo,turns),lane='native-program-improvement';
   const previous=registry.artifacts.filter(artifact=>artifact.lane===lane&&artifact.path!==path);
   registry.superseded_artifacts=[...(registry.superseded_artifacts??[]),...previous.filter(artifact=>!(registry.superseded_artifacts??[]).some(old=>old.path===artifact.path))];
   registry.artifacts=registry.artifacts.filter(artifact=>artifact.lane!==lane);
   registry.artifacts.push({path,sha256:prepared.sha256,rows:prepared.rows,lane,runtime_api:'native-ordinary-runtime',verification_artifact:relative(repo,receiptPath),verification_sha256:digest(await readFile(receiptPath))});
   const staged=registryPath+'.optimizer.tmp';await writeFile(staged,JSON.stringify(registry,null,2)+'\n');await rename(staged,registryPath);
  }
  return {rows:prepared.rows,published:true,path:relative(repo,turns),sha256:prepared.sha256,correctionContextTurns:prepared.correctionContextTurns.length};
 }finally{await lock.close();await rm(lockPath);await rm(temporary,{recursive:true,force:true});}
}
if(process.argv[1]===new URL(import.meta.url).pathname){const [repo,...manifests]=process.argv.slice(2);if(!repo||!manifests.length)throw Error('usage: publish-optimizer-training.mjs REPO VERIFIED_MANIFEST...');console.log(JSON.stringify(await publish(repo,manifests.map(path=>resolve(path)))));}
