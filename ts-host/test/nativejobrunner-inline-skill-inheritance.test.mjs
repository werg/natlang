import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, mkdir, readFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';

const dist = process.env.NATLANG_TEST_DIST ? resolve(process.env.NATLANG_TEST_DIST) : resolve('dist');
const {nativeJobRunner, expectedProvenance} = await import(pathToFileURL(join(dist,'teacher/collector.js')));
const {registerLocalEndpoint} = await import(pathToFileURL(join(dist,'model/chat-completion.js')));
const {TOOLS_PROMPT} = await import(pathToFileURL(join(dist,'native/prompt.js')));

test('nativeJobRunner passes caller bound skills through nested inline nl.with children and read_code', async t => {
  const root = await mkdtemp(join(tmpdir(),'natlang-inline-skill-'));
  t.after(() => rm(root,{recursive:true,force:true}));
  const jobs=join(root,'jobs');await mkdir(jobs);
  const skill = `---\nname: judge-against-criteria\ndescription: Use for eligibility judgments under explicit criteria.\n---\nRead the exact criterion, entity scope, and all current evidence. Apply only stated predicates.\n`;
  const expectedDocument=skill.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/,'').trim();
  const record={version:'natlang.program/2',id:'inline-skill-inheritance-nativejobrunner',kind:'lambda_source',split:'diagnostic',source_groups:['fixture:inline-skill-inheritance'],license:'project-authored',
    semantics:{root:'solve.nl',files:{'solve.nl':'---\nargs: {}\nreturns: number\n---\nApply the child to an exact arithmetic task.\n','solve/skills/judge-against-criteria/SKILL.md':skill},inputs:{},expected:4,oracle:'exact'}};
  const endpoint='http://127.0.0.1:65533';let rootEvalSent=false;
  const offered=new Set(),bodyRead=new Set(),capturedCriterionVisible=new Set();
  const requests=[];
  registerLocalEndpoint(endpoint,async(url,init)=>{
    assert.match(url,/\/v1\/chat\/completions$/);
    const body=JSON.parse(String(init.body)),messages=body.messages??[],names=(body.tools??[]).map(x=>x.function?.name);
    const serialized=JSON.stringify(messages);
    const level=serialized.includes('Delegate to a second inline child with the exact captured criterion.')?1:
      serialized.includes('Compute exactly two plus two under the supplied criterion.')?2:0;
    const child=level>0;
    if(child&&serialized.includes('judge-against-criteria'))offered.add(level);
    if(child&&(serialized.includes(expectedDocument)||serialized.includes(JSON.stringify(expectedDocument).slice(1,-1))))bodyRead.add(level);
    if(child&&serialized.includes('solve exact addition'))capturedCriterionVisible.add(level);
    let name,args;
    if(names.length===1&&names[0]==='execution_plan') {name='execution_plan';args={plan:'Use the available bound skill when it applies, then finish the exact task.'};}
    else if(child&&names.includes('read_code')&&!serialized.includes(expectedDocument)) {
      name='read_code';args={name:'skills.judge-against-criteria'};
    } else if(level===1&&names.includes('eval')) {
      name='eval';args={code:'const inner = nl.with<{criterion: string}, number>({criterion})`Compute exactly two plus two under the supplied criterion.`; return await inner();',finish:true};
    } else if(level===2&&names.includes('eval')) {name='eval';args={code:'2 + 2',finish:true};}
    else if(!rootEvalSent&&names.includes('eval')) {
      name='eval';args={code:'const outer = nl.with<{criterion: string}, number>({criterion: "solve exact addition"})`Delegate to a second inline child with the exact captured criterion.`; return await outer();',finish:true};rootEvalSent=true;
    } else throw new Error(`Unexpected model request; level=${level}; tools=${names.join(',')}`);
    requests.push({level,tool_names:names,selected:name,context_hash:createHash('sha256').update(serialized).digest('hex')});
    assert(names.includes(name),`fake response selected unoffered ${name}`);
    return new Response(JSON.stringify({id:`inline-proof-${requests.length}`,object:'chat.completion',created:1,model:'mock-luna',choices:[{index:0,
      message:{role:'assistant',content:null,tool_calls:[{id:`inline-proof-call-${requests.length}`,type:'function',function:{name,arguments:JSON.stringify(args)}}]},finish_reason:'tool_calls'}]}),
      {status:200,headers:{'content-type':'application/json'}});
  });
  const options={endpoint,modelId:'mock-luna',systemPrompt:TOOLS_PROMPT,contextTokens:8192,rootSeed:42,temperature:0,maxTurns:10,maxModelRequests:12,
    modelConcurrency:1,collectionRole:'teacher',textNeuraleseEmulation:false,executionPlans:true,request:{max_tokens:2048},jobs,output:join(root,'unused.jsonl'),workers:1,transportRetries:0};
  const runner=nativeJobRunner(options);
  const row=await runner({index:0,record},expectedProvenance(record,options),AbortSignal.timeout(30000));
  assert.equal(row.outcome.accepted,true,JSON.stringify(row.outcome));
  assert.equal(row.outcome.value,4);
  assert.deepEqual([...offered].sort(),[1,2],'both inline levels should see the caller-bound skill listing');
  assert.deepEqual([...bodyRead].sort(),[1,2],'both inline levels should read the exact bound skill body');
  assert.deepEqual([...capturedCriterionVisible].sort(),[1,2],'both inline levels should retain the typed captured criterion');
  const traceName=(await (await import('node:fs/promises')).readdir(jobs)).find(name=>name.endsWith('.trace.jsonl'));
  const trace=(await readFile(join(jobs,traceName),'utf8')).split('\n').filter(Boolean).map(JSON.parse);
  const starts=trace.filter(event=>event.kind==='invocation'&&event.phase==='start').map(event=>event.call_id);
  const childCallIds=new Set(starts.slice(1));
  const offeredIds=new Set(trace.filter(event=>event.kind==='skill_use'&&event.phase==='offered'&&event.skill_name==='judge-against-criteria').map(event=>event.invocation_id));
  const readIds=new Set(trace.filter(event=>event.kind==='skill_use'&&event.phase==='body_read'&&event.skill_name==='judge-against-criteria').map(event=>event.invocation_id));
  assert.equal(childCallIds.size,2,'the fixture should create two inline child invocations');
  for(const id of childCallIds){assert(offeredIds.has(id),`skill should be offered to inline child ${id}`);assert(readIds.has(id),`skill body should be read by inline child ${id}`);}
  assert(requests.some(request=>request.level===1&&request.selected==='read_code'));
  assert(requests.some(request=>request.level===2&&request.selected==='read_code'));
  assert(requests.some(request=>request.level===2&&request.context_hash));
  assert.equal(record.split,'diagnostic');
});
