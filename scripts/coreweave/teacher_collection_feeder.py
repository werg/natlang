"""Existing single-batch teacher feeder with an immutable optional source configuration.

Only the guardian allocates batch slots; this mechanism does not choose tasks.
"""
import pathlib,time,json,subprocess,os,hashlib,traceback
def load_source_configuration(path, expected_sha256=None, expected_bytes=None):
 raw=pathlib.Path(path).read_bytes()
 if expected_sha256 is not None:
  if (expected_bytes is None or len(raw)!=int(expected_bytes)
      or hashlib.sha256(raw).hexdigest()!=expected_sha256):
   raise ValueError('immutable feeder source configuration changed')
 elif expected_bytes is not None:raise ValueError('feeder source configuration lacks its digest')
 value=json.loads(raw)
 cases=pathlib.Path(value['cases']).read_bytes()
 if hashlib.sha256(cases).hexdigest()!=value['cases_sha256']:
  raise ValueError('feeder source cases changed')
 if type(value['rows'])is not int or value['rows']!=len([line for line in cases.splitlines()if line.strip()]):
  raise ValueError('feeder source row count differs from its immutable cases')
 return value

def prepare_collection_command(cases,out,rows,seed,configuration):
 sampling=collection_sampling(configuration)
 return ['python','-m','natlang_neuralese.rl.task_rollouts','prepare','--cases',str(cases),'--out',str(out),'--rows',str(rows),'--group',str(sampling['group']),'--temperature','0.6','--case-seconds','7200','--max-model-requests','128','--max-turns','160','--seed-base',str(seed),'--root-mode',sampling['root_mode']]


def resumed_collection_rows(entries,source):
 # Old sealed receipts have no explicit sampling declaration and always used two.
 configuration=source.get('source_provenance')or{}
 sampling=collection_sampling(configuration)
 if 'collection_sampling' in configuration:
  verify_collection_sampling(source,entries,configuration)
 if len(entries)%sampling['group']:
  raise ValueError('resumed collection queue has a partial sample group')
 return len(entries)//sampling['group']

root=pathlib.Path(os.environ['NATLANG_TEACHER_GENERATION_ROOT']);runtime=pathlib.Path(os.environ['NATLANG_REPO']);queue_worker=pathlib.Path(os.environ['NATLANG_TEACHER_QUEUE_WORKER']);node_bin=os.environ['NATLANG_TEACHER_NODE_BIN'];task_runtime=pathlib.Path(os.environ['NATLANG_TEACHER_TASK_RUNTIME']);teacher_identity=pathlib.Path(os.environ['NATLANG_TEACHER_IDENTITY']);controller=queue_worker.parent;batch=int(os.environ['NATLANG_TEACHER_START_BATCH']);worker_count=int(os.environ.get('NATLANG_TEACHER_WORKER_COUNT','24'));assert 1<=worker_count<=24
teacher_source=root/'continuations/batch14-after-csf-ordered-system-qualification-20261016-v1/teacher-source.json';teacher_source_raw=teacher_source.read_bytes();assert hashlib.sha256(teacher_source_raw).hexdigest()=='c67d81315c3f791c4d1a6f44da77e617bf8c86d4c216adbb09245782e59e0614';actual_teacher=json.loads(teacher_source_raw)['teacher'];assert actual_teacher['revision']=='b9ce18a6887e18c089fdea8509a33e63429a3300'

import sys
sys.path.insert(0,os.environ.get('NATLANG_TEACHER_GUARDIAN_MODULE_ROOT',str(controller)))
from teacher_generation_guardian import verify_teacher_identity,write
verify_teacher_identity(teacher_identity,server='http://127.0.0.1:8001',model_id='pennyroyal')
from teacher_batch_completion import batch_completion
from teacher_source_sequence import collection_sampling,verify_collection_sampling
controller_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()

resume=None
if os.environ.get('NATLANG_TEACHER_FANOUT_RESUME'):
 from teacher_generation_guardian import binding,fanout_frontier
 resume=json.loads(pathlib.Path(os.environ['NATLANG_TEACHER_FANOUT_RESUME']).read_text())
 while not pathlib.Path(resume['armed']).exists() or (controller/'DRAIN').exists():time.sleep(.2)
 assert binding(resume['frontier']['path'])==resume['frontier']
 assert binding(resume['source']['path'])==resume['source']
 frontier=json.loads(pathlib.Path(resume['frontier']['path']).read_text())
 assert frontier==fanout_frontier(frontier['queue']['path'],[r['path'] for r in frontier['journals']],worker_count)

def phase(value,**kw):
 r={'at':time.time(),'phase':value,**kw};target=root/'feeder-phases'/f'batch-{batch:04d}.json';target.parent.mkdir(exist_ok=True);write(target,r);print(json.dumps(r),flush=True)
try:
 phase('waiting_strong_server_ready')
 while not (root/'server-ready.json').exists():time.sleep(2)
 env={**os.environ,'PYTHONSAFEPATH':'1','PYTHONPATH':str(controller)+':'+str(runtime)+':'+str(runtime/'training/neuralese'),'NATLANG_REPO':str(runtime),'PATH':node_bin+os.pathsep+os.environ.get('PATH',''),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','CUDA_VISIBLE_DEVICES':''}
 while True:
  resuming=resume is not None and batch==resume['batch']
  if resuming:
   out=pathlib.Path(frontier['queue']['path']).parent
   entries=[json.loads(line) for line in (out/'queue.jsonl').read_text().splitlines() if line]
   binding=json.loads((out/'teacher-source.json').read_text());rows=resumed_collection_rows(entries,binding)
   for name in ['journal.jsonl','completion.json','score.json']:
    f=out/name;retained=out/(name+'.pre-fanout-v11')
    if f.exists():
     if retained.exists():assert retained.read_bytes()==f.read_bytes()
     else:retained.write_bytes(f.read_bytes())
   (out/'fanout-v11-execution.json').write_text(json.dumps({'frontier':resume['frontier'],'original_source':resume['source'],'actual_worker_count':worker_count,'actor_unchanged':True,'original_terminal_outcomes_preserved':True}))
  else:
   out=root/'evidence-collections-v2'/f'batch-{batch:04d}';cases=root/'selected-cases.jsonl';configured=pathlib.Path(os.environ.get('NATLANG_TEACHER_SOURCE_CONFIGURATION',str(root/'code-source-v3.json')))
   if configured.exists():
    conf=load_source_configuration(configured,os.environ.get('NATLANG_TEACHER_SOURCE_CONFIGURATION_SHA256'),os.environ.get('NATLANG_TEACHER_SOURCE_CONFIGURATION_BYTES'));cases=pathlib.Path(conf['cases']);rows=conf['rows']
   elif os.environ.get('NATLANG_TEACHER_SOURCE_CONFIGURATION'):raise ValueError('immutable feeder source configuration is missing')
   else:rows=32;conf={}
   packet=out/'source-cases.jsonl';out.mkdir(parents=True,exist_ok=True);packet.write_bytes(cases.read_bytes());cases=packet
   seed=2900000+batch*100000
   prep=prepare_collection_command(cases,out,rows,seed,conf);subprocess.run(prep,cwd=root,env=env,check=True)
   entries=[json.loads(line) for line in (out/'queue.jsonl').read_text().splitlines() if line]
   for e in entries:e.update(context_tokens=65536,text_neuralese_emulation=True)
   (out/'queue.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in entries));binding={'schema':'natlang.sandbox-strong-teacher-collection/1','at':time.time(),'batch':batch,'teacher':actual_teacher,'source_cases':str(cases),'source_cases_sha256':hashlib.sha256(cases.read_bytes()).hexdigest(),'seed_base':seed,'context_tokens':65536,'temperature':0.6,'root_mode':collection_sampling(conf)['root_mode'],'text_neuralese_emulation':True,'concurrency':worker_count,'policy_claim':'independent stronger teacher; not learner/onpolicy','protected_selection':conf.get('protected_selection',conf),'source_provenance':conf,'collection_controller':{'mechanism':'natlang.teacher-collection-feeder/1','helper_sha256':controller_sha256,'script':str(pathlib.Path(__file__)),'guardian_protocol':'natlang.teacher-generation-guardian/1'},'runtime_source_receipt':{'source_commit':'2ff598246378','root':str(runtime)},'actual_task_runtime':str(task_runtime),'actual_teacher_execution_identity':{'path':str(teacher_identity),'sha256':hashlib.sha256(teacher_identity.read_bytes()).hexdigest(),'bytes':teacher_identity.stat().st_size},'actor':json.loads(teacher_identity.read_text())['actor'],'teacher_configuration_source':{'path':str(teacher_source),'sha256':hashlib.sha256(teacher_source_raw).hexdigest(),'bytes':len(teacher_source_raw)}};binding.update({'collection_sampling':collection_sampling(conf)}if 'collection_sampling'in conf else {});verify_collection_sampling(binding,entries,conf);(out/'teacher-source.json').write_text(json.dumps(binding,indent=2));children=[]
  children=[]
  for worker in range(worker_count):
   label='fanout-v11-' if resuming else '';queue=out/f'queue-{label}worker-{worker}.jsonl';journal=out/f'journal-{label}worker-{worker}.jsonl';data=''.join(json.dumps(e)+'\n' for e in (frontier['shards'][worker] if resuming else entries[worker::worker_count]));assert not queue.exists() or queue.read_text()==data;queue.write_text(data);cmd=['python','-u',str(queue_worker),str(queue),str(journal),'--runtime',str(task_runtime),'--server','http://127.0.0.1:8001','--chat-completions-url','http://127.0.0.1:8001/v1/chat/completions','--model-id','pennyroyal','--model-concurrency','1','--max-batch-cases','1','--case-seconds','7200','--identical-turn-limit','8','--teacher-identity',str(teacher_identity),'--drain-file',str(controller/'DRAIN')];log=(out/f'{label}worker-{worker}.log').open('ab');c=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=log);children.append(c)
  phase('collecting',batch=batch,out=str(out),worker_pids=[c.pid for c in children],cases=rows,teacher=actual_teacher['model'],worker_count=worker_count)
  codes=[c.wait() for c in children];journal=[]
  for journal_path in sorted(out.glob('journal*worker-*.jsonl')):journal.extend(json.loads(line) for line in journal_path.read_text().splitlines() if line)
  journal.sort(key=lambda x:x.get('time',0));(out/'journal.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in journal));subprocess.run(['python','-m','natlang_neuralese.rl.task_rollouts','score','--run-dir',str(out),'--emit-rollouts',str(out/'rollouts.jsonl'),'--round',str(batch)],cwd=root,env=env,check=True);phase('batch_complete',batch=batch,score=json.loads((out/'score.json').read_text()),returncodes=codes)
  completion=batch_completion(entries,journal,codes,json.loads((out/'score.json').read_text()))
  completion.update(at=time.time(),controller_mechanism='natlang.teacher-collection-feeder/1',helper_sha256=controller_sha256)
  pending=out/'completion.pending';pending.write_text(json.dumps(completion));os.replace(pending,out/'completion.json')
  if not completion['can_advance']:raise RuntimeError('teacher batch incomplete; exact repair required: '+json.dumps(completion['repair_required']))
  if os.environ.get('NATLANG_TEACHER_SINGLE_BATCH')=='1':break
  batch+=1
except BaseException as e:
 write(root/'feeder-phases'/f'batch-{batch:04d}-error.json',{'at':time.time(),'error':str(e),'type':type(e).__name__});traceback.print_exc();raise
