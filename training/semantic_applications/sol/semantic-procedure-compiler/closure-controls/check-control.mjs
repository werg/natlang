/** External exact control checker; dependency/call meaning is independently reviewed NL work. */
import fs from 'node:fs';
const read=name=>JSON.parse(fs.readFileSync(new URL(name,import.meta.url),'utf8'));
const input=read('transitive-permission-bug.input.json'),oracle=read('transitive-permission-bug.oracle.json'),reference=read('transitive-permission-bug.reference.json');
const eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b);const set=a=>[...a].sort();
function verify(value){const issues=[];const require=(v,s)=>{if(!v)issues.push(s);};try{
 const c=value.closure,files=new Map(input.actual_files.map(x=>[x.path,x]));
 require(c.entryPath===input.entry_path,'original entry identity');require(eq(set(c.files.map(x=>x.path)),set(oracle.expected_visited_paths)),'actual visited file set');
 for(const f of c.files)require(eq(f,files.get(f.path)),'original generated file bytes');
 for(const r of [...c.references,...c.runtimeReads])require(r.generatedEvidence.length>0,'nonempty actual caller support');
 for(const r of [...c.references,...c.runtimeReads])for(const s of r.generatedEvidence){const f=files.get(s.path);require(s.path===r.callerPath&&f&&s.quote.length>0&&f.text.includes(s.quote),'exact caller span');}
 require(c.references.every(r=>r.status==='resolved'&&files.has(r.callerPath)&&files.has(r.targetPath)),'resolved original path domains');
 for(const r of c.references)if(r.kind==='type'){require(r.targetEvidence.length>0,'type defining evidence');for(const s of r.targetEvidence){const f=files.get(s.path);require(s.path===r.targetPath&&f?.kind==='types'&&s.quote.length>0&&f.text.includes(s.quote),'exact defining type span');}}
 const actualCalls=c.references.filter(r=>r.kind==='call').map(r=>[r.callerPath,r.targetPath]);require(eq(actualCalls.map(JSON.stringify).sort(),oracle.required_call_edges.map(JSON.stringify).sort()),'actual transitive call edges');
 const reachable=new Set([c.entryPath]);for(let n=0;n<c.files.length;n++)for(const r of c.references)if(reachable.has(r.callerPath))reachable.add(r.targetPath);require(c.files.every(f=>reachable.has(f.path)),'closure reachability');
 require(c.runtimeReads.length===1&&!c.runtimeReads[0].wholeCollection&&eq(set(c.runtimeReads[0].clauseIds),set(oracle.expected_runtime_clause_ids)),'actual explicit runtime read');
 require(eq(c.runtimeClauses,input.selected_clauses.filter(x=>oracle.expected_runtime_clause_ids.includes(x.id))),'original runtime clause values');
 require(c.unresolved.length===0,'decisive supplied closure');require(value.actualEmittedPermission===oracle.actual_emitted_permission&&value.sourceRequiredPermission===oracle.source_required_permission&&value.matchesSource===oracle.matches_source,'independent emitted/source invariant');
 }catch(e){issues.push('malformed closure witness: '+e.message);}return {ok:issues.length===0,issues};}
const checks=[];function check(name,expected,mutate){const value=structuredClone(reference);mutate?.(value);const result=verify(value);checks.push({name,expected,actual:result.ok,passed:result.ok===expected,issues:result.issues});}
check('canonical connected closure witness',true);
check('unordered actual file and dependency sets',true,v=>{v.closure.files.reverse();v.closure.references.reverse();});
check('full exact caller quotations',true,v=>{for(const r of [...v.closure.references,...v.closure.runtimeReads])for(const s of r.generatedEvidence)s.quote=input.actual_files.find(f=>f.path===s.path).text;});
check('transitive grandchild missing',false,v=>v.closure.files=v.closure.files.filter(f=>!f.path.endsWith('/permit.nl')));
check('uncalled ideal decoy included',false,v=>v.closure.files.push(input.actual_files.find(f=>f.path==='decoys/explainIdeal.nl')));
check('grandchild quote rebound to parent',false,v=>v.closure.runtimeReads[0].generatedEvidence[0].path='decision.nl');
check('ideal source silently corrects emitted permission',false,v=>v.actualEmittedPermission=false);
check('actual emitted value contaminates required permission',false,v=>v.sourceRequiredPermission=true);
check('unread source clause introduced',false,v=>v.closure.runtimeClauses.push(input.selected_clauses.find(x=>x.id==='WAIVER')));
check('decided call without exact supporting quote',false,v=>v.closure.references.find(r=>r.kind==='call').generatedEvidence=[]);
check('runtime read without exact supporting quote',false,v=>v.closure.runtimeReads[0].generatedEvidence=[]);
check('fabricated defining type quotation',false,v=>v.closure.references.find(r=>r.kind==='type').targetEvidence[0].quote='No actual definition here');
check('actual call edge erased',false,v=>v.closure.references=v.closure.references.filter(r=>r.targetPath!=='decision/readGate/permit.nl'));
console.log(JSON.stringify({scope:'Authored connected control witness exact checks, not NL closure/guard execution',passed:checks.every(x=>x.passed),checks},null,2));if(checks.some(x=>!x.passed))process.exitCode=1;
