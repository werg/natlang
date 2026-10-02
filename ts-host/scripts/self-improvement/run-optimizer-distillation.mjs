/** Execute preparation, capped Sharp training and frozen transfer evaluation without manual gates. */
import {mkdir,readFile,writeFile,open} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {spawn} from 'node:child_process';
import {prepare} from './prepare-optimizer-training.mjs';
import {digest} from './replay-runtime.mjs';
const root=resolve(process.argv[2]),manifests=process.argv.slice(3).map(path=>resolve(path));
if(!manifests.length)throw Error('usage: run-optimizer-distillation.mjs CAMPAIGN EXISTING_VERIFIED_MANIFEST...');
const output=join(root,'sharp-training');await mkdir(output,{recursive:true});
const remote='/mnt/external/natlang-native-directory-20261002';
const model=JSON.parse(await readFile(join(output,'model-metadata.json'),'utf8'));
const protocol={schema:'natlang.sharp-optimizer-distillation/1',model:{repository:model.id,revision:model.sha,template:'Sharp-MiniCPM5-2B'},training:{jobs:1,steps:32,accum:4,rank:8,learningRate:0.0001,seed:20261002,maxLength:16384,cudaMemoryFraction:0.045,containerMemory:'16g',targetModules:'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'},evaluation:{checkpoints:['starting','trained'],families:['directory-eligibility-appeals','directory-confirmed-meetings'],experiments:1,modelCalls:600,caseExecutions:128},limits:{elapsedMs:43200000,trainingAttempts:1},sources:manifests,remote,design:'One optimizer weight-learning pilot. No teacher transfer trajectories train weights. No tuning or new cohorts after confirmation. Program-execution student remains Bonsai. All preparation, training and evaluation commands execute automatically.'};
const protocolPath=join(output,'protocol.json');try{if(JSON.stringify(JSON.parse(await readFile(protocolPath,'utf8')))!==JSON.stringify(protocol))throw Error('Frozen distillation protocol changed');}catch(error){if(error.code!=='ENOENT')throw error;await writeFile(protocolPath,JSON.stringify(protocol,null,2));}
let state;try{state=JSON.parse(await readFile(join(output,'state.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
state??={startedAt:Date.now(),completed:[]};
let active,forward,interrupted=false;process.once('SIGTERM',()=>{interrupted=true;active?.kill('SIGTERM');forward?.kill('SIGTERM');});
const save=()=>writeFile(join(output,'state.json'),JSON.stringify(state,null,2));
const quote=value=>"'"+String(value).replaceAll("'","'\\''")+"'";
async function run(id,argv,timeout=3600000,cleanup=false){
 if(state.completed.includes(id)&&!cleanup)return;
 if(interrupted&&!cleanup)throw Error('Distillation interrupted');
 const remaining=cleanup?timeout:Math.min(timeout,protocol.limits.elapsedMs-(Date.now()-state.startedAt));
 if(remaining<=0)throw Error('Distillation assignment exhausted');
 const log=await open(join(output,id+'.log'),'a');console.log(JSON.stringify({id,status:'running'}));
 let status;try{status=await new Promise(resolve=>{const child=spawn(argv[0],argv.slice(1),{cwd:new URL('../../',import.meta.url).pathname,stdio:['ignore',log.fd,log.fd]});active=child;let killTimer;const timer=setTimeout(()=>{child.kill('SIGTERM');killTimer=setTimeout(()=>child.kill('SIGKILL'),120000);},remaining);child.once('error',error=>resolve({error:String(error)}));child.once('exit',(code,signal)=>{active=undefined;clearTimeout(timer);clearTimeout(killTimer);resolve({code,signal});});});}finally{await log.close();}
 if(status.code!==0)throw Error(id+' failed: '+JSON.stringify(status));if(!state.completed.includes(id))state.completed.push(id);await save();
}
const ssh=command=>['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','dgx',command];
const docker=(name,args,gpu=false)=>['docker','run','--rm','--name',name,...(gpu?['--gpus','all','--memory','16g','--cpus','2']:['--memory','2g','--cpus','1']),'-v',remote+':/work','-e','HF_HOME=/work/cache','--workdir','/work/repo','--entrypoint','python','docker-train-bgkit2-p19o:latest',...args].map(quote).join(' ');
try{
 await save();
 // Wait only for this finite assignment's output, not for a quality threshold or extra cohort.
 for(;;){if(interrupted)throw Error('Distillation interrupted');if(Date.now()-state.startedAt>=protocol.limits.elapsedMs)throw Error('Distillation assignment exhausted');try{await readFile(join(root,'result.json'));break;}catch(error){if(error.code!=='ENOENT')throw error;}await new Promise(resolve=>setTimeout(resolve,30000));}
 await prepare(join(output,'prepared'),[...manifests,join(root,'training-current','manifest.json')]);
 const prepared=JSON.parse(await readFile(join(output,'prepared','manifest.json'),'utf8'));
 if(state.trainingData&&state.trainingData!==prepared.sha256)throw Error('Frozen training data changed');
 state.trainingData=prepared.sha256;await save();
 await run('copy-inputs',['rsync','-a',join(output,'prepared')+'/','dgx:'+remote+'/training/prepared/']);
 await run('copy-code',['rsync','-a',new URL('../../../scripts/',import.meta.url).pathname,'dgx:'+remote+'/repo/scripts/']);
 await run('render',ssh(docker('natlang-native-sharp-render',['/work/repo/scripts/render_training_corpus.py','--inputs','/work/training/prepared/training-turns.jsonl','--output','/work/training/rendered.jsonl','--end-token','<|im_end|>','--model','/work/model','--revision',model.sha])));
 await run('audit',ssh(docker('natlang-native-sharp-audit',['/work/repo/scripts/audit_training_corpus.py','--input','/work/training/rendered.jsonl','--output','/work/training/ready.jsonl','--model','/work/model','--revision',model.sha,'--max-len','16384'])));
 if(!state.completed.includes('train')){state.trainingStarted=true;await save();}
 await run('train',ssh(docker('natlang-native-sharp-train',['/work/repo/scripts/train_lora.py','/work/training/ready.jsonl','/work/training/run','--model','/work/model','--model-revision',model.sha,'--load-in-4bit','--no-kbit-upcast','--cuda-memory-fraction','0.045','--max-len','16384','--batch-tokens','16384','--rank','8','--lr','0.0001','--steps','32','--accum','4','--target-modules',protocol.training.targetModules,'--holdout','0','--skip-heldout-loss','--no-merge','--save-every','4','--seed','20261002','--require-audit'],true)),14400000);
 await run('retrieve',['rsync','-a','dgx:'+remote+'/training/',output+'/remote/']);
 const checkpointState=JSON.parse(await readFile(join(output,'remote','run','checkpoint','state.json'),'utf8'));if(checkpointState.trained_examples<checkpointState.corpus.target_examples)throw Error('Training stopped before its declared updates completed');
 const adapter=await readFile(join(output,'remote','run','checkpoint','weights','adapter_model.safetensors'));
 const checkpoints={starting:null,trained:{path:'/work/training/run/checkpoint/weights',sha256:digest(adapter)}};
 await writeFile(join(output,'checkpoint-identities.json'),JSON.stringify(checkpoints,null,2));
 await writeFile(join(output,'checkpoint-map.json'),JSON.stringify({starting:null,trained:checkpoints.trained.path},null,2));
 await run('copy-checkpoint-map',['rsync','-a',join(output,'checkpoint-map.json'),'dgx:'+remote+'/training/']);
 const server=['docker','run','--rm','-d','--name','natlang-native-sharp-eval','--gpus','all','--memory','16g','--cpus','2','-p','127.0.0.1:18184:18184','-v',remote+':/work','--workdir','/work/repo','--entrypoint','python','docker-train-bgkit2-p19o:latest','/work/repo/scripts/serve_improvement_student.py','--model','/work/model','--revision',model.sha,'--checkpoint-map','/work/training/checkpoint-map.json','--port','18184','--max-context','16384','--device','cuda','--load-in-4bit','--cuda-memory-fraction','0.045'];
 // A resumed controller may have cleaned up its earlier owned server. Restart it.
 state.completed=state.completed.filter(id=>!['serve','stop-evaluation-server'].includes(id));await save();
 await run('serve',ssh(server.map(quote).join(' ')));
 forward=spawn('ssh',['-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-N','-L','127.0.0.1:18084:127.0.0.1:18184','dgx'],{stdio:'ignore'});
 const readyUntil=Date.now()+600000;for(;;){try{const response=await fetch('http://127.0.0.1:18084/v1/models',{signal:AbortSignal.timeout(5000)});if(response.ok)break;}catch{}if(Date.now()>readyUntil)throw Error('Owned evaluation server did not become ready');await new Promise(resolve=>setTimeout(resolve,10000));}
 await run('evaluate',[process.execPath,'scripts/self-improvement/evaluate-optimizer-checkpoints.mjs',root,join(output,'checkpoint-identities.json'),'http://127.0.0.1:18084'],14400000);
 state.disposition='completed';
}catch(error){state.disposition=interrupted?'interrupted':'failed-or-exhausted';state.error=String(error);console.error(state.error);process.exitCode=1;}
finally{forward?.kill('SIGTERM');if(state.trainingStarted&&!state.completed.includes('train'))await run('stop-incomplete-training',ssh('docker stop -t 120 natlang-native-sharp-train'),180000,true).catch(()=>{});if(state.completed.includes('serve'))await run('stop-evaluation-server',ssh('docker stop -t 30 natlang-native-sharp-eval'),60000,true).catch(error=>{state.cleanupError=String(error);});await save();await writeFile(join(output,'result.json'),JSON.stringify({protocol,state},null,2));}
