/** Trusted exact source/reference checker; semantic policy remains in authored/generated NL. */
import {createHash} from 'node:crypto';
import {compileVirtualProject} from '../../../../ts-host/dist/runtime/virtual-project.js';
const canonical=x=>Array.isArray(x)?x.map(canonical):x&&typeof x==='object'?Object.fromEntries(Object.keys(x).sort().map(k=>[k,canonical(x[k])])):x;
const equal=(a,b)=>JSON.stringify(canonical(a))===JSON.stringify(canonical(b));
const bag=x=>JSON.stringify((x??[]).map(v=>JSON.stringify(canonical(v))).sort());
const safePath=x=>typeof x==='string'&&!x.startsWith('/')&&!x.split('/').some(p=>!p||p==='.'||p==='..')&&!x.includes('\\');
const atomKey=a=>JSON.stringify([a.key,a.subject]);
const atomFacts=a=>({key:a.key,subject:a.subject,value:a.value,known:a.known,establishedAt:a.establishedAt});
export function verifyCandidate({value:r,oracle,inputs,fixture}){
 const issues=[];const demand=(ok,msg)=>{if(!ok)issues.push(msg);};
 try{
  const snapshot=inputs.snapshot,request=inputs.request;demand(r.requestId===request.id&&r.requestId===oracle.requestId,'request identity');
  const originals=new Map(snapshot.records.map(d=>[d.id,d]));
  demand(originals.size===snapshot.records.length,'original source ID uniqueness');
  demand(snapshot.records.every(d=>fixture[d.path]===d.text&&createHash('sha256').update(d.text).digest('hex')===d.sha256),'source fixture/hash identity');
  const sourceClauses=new Set(snapshot.records.flatMap(d=>[...d.text.matchAll(/^\[([^\]]+)\]/gm)].map(m=>m[1])));
  const ruleRows=[...r.model.requirements.rules,...r.model.requirements.permissions],ruleIds=new Set(ruleRows.map(row=>row.id));
  const modules=new Map(r.program.modules.map(m=>[m.path,m]));
  demand(modules.size===r.program.modules.length,'generated module uniqueness');
  const files=[...r.program.modules.map(m=>[m.path,m.text]),...r.program.typeFiles.map(f=>[f.path,f.text])];
  demand(files.every(([path,text])=>safePath(path)&&typeof text==='string')&&new Set(files.map(([p])=>p)).size===files.length,'generated file paths');
  demand(modules.has(r.program.entryPath),'generated entry exists');
  for(const m of modules.values()){
   demand(m.path===m.plan.path&&m.path===m.contributionAudit.path,'generated plan/audit ownership');
   demand(m.plan.calls.every(path=>modules.has(path)),'generated child reference domain');
   demand(m.plan.ruleIds.every(id=>ruleIds.has(id)),'generated source rule domain');
   demand(m.plan.ownedFields.length>0&&m.contributionAudit.substantive===true,'authored contribution presence');
  }
  const built=compileVirtualProject({files:Object.fromEntries(files)},{},{constrained:true,target:'node'});
  demand(built.ok,'generated parser/type validity');
  const selected=r.model.requirements.selection.selectedClauses;
  demand(new Set(selected.map(c=>c.id)).size===selected.length&&selected.every(c=>sourceClauses.has(c.id)),'governed clause identity domain');
  const old=selected.find(c=>c.id==='V1'),current=selected.find(c=>c.id==='V2');
  demand(old?.effectiveUntil===oracle.governed_rule_invariants.ONE_before&&current?.effectiveFrom===oracle.governed_rule_invariants.TWO_from,'governed effective boundary');
  demand(bag(r.model.requirements.selection.replacedClauseIds)===bag(oracle.governed_rule_invariants.replaced),'replacement scope');
  demand(oracle.governed_rule_invariants.preserved.every(id=>r.model.requirements.selection.preservedClauseIds.includes(id)),'preserved independent rules');
  demand(oracle.governed_rule_invariants.noncontrolling.every(id=>!selected.some(c=>c.id===id)),'habit not governing');
  const requiredRuleIds=selected.map(c=>c.id);
  demand(requiredRuleIds.every(id=>r.program.sourceToModule.some(row=>row.clauseId===id&&row.paths.length>0&&row.paths.every(p=>modules.has(p)))),'source/module coverage');
  for(const rule of [...r.model.requirements.rules,...r.model.requirements.permissions]){
   demand(rule.clauseIds.every(id=>sourceClauses.has(id)),'rule source reference domain');
   for(const exception of rule.exceptionBindings){demand(sourceClauses.has(exception.exceptionClauseId),'exception source domain');demand(oracle.governed_rule_invariants.queue_waiver_does_not_waive.every(id=>exception.retainedDutyIds.includes(id)),'waiver retained duties');}
  }
  function evidence(v){
   if(!v||typeof v!=='object')return;
   if('sourceId'in v&&'path'in v&&'quote'in v){const d=originals.get(v.sourceId);demand(d&&d.path===v.path&&typeof v.quote==='string'&&v.quote.length>0&&d.text.includes(v.quote),'exact original witness');}
   if('path'in v&&'quote'in v&&!('sourceId'in v)){const m=modules.get(v.path);demand(m&&typeof v.quote==='string'&&v.quote.length>0&&m.text.includes(v.quote),'exact generated span');}
   for(const x of Object.values(v))if(Array.isArray(x))x.forEach(evidence);else if(x&&typeof x==='object')evidence(x);
  }evidence(r);
  const verdicts=r.suite.verdicts;demand(bag(verdicts.map(v=>v.scenarioId))===bag(request.requiredScenarios),'requested scenario coverage');
  const attemptValues=new Map();
  for(const verdict of verdicts){
   const source=snapshot.scenarios.find(s=>s.id===verdict.scenarioId);demand(!!source,'original scenario identity');if(!source)continue;
   demand(verdict.valid===true&&verdict.violatedObservables.length===0,'scenario verdict');
   demand(equal(verdict.traces.map(t=>t.eventId),source.events.map(e=>e.id)),'source event sequence');
   let previous=source.initial,sourcePrevious=source.initial;
   for(const trace of verdict.traces){
    const event=source.events.find(e=>e.id===trace.eventId);demand(trace.scenarioId===verdict.scenarioId,'trace scenario ownership');
    demand(bag(trace.stateBefore.map(atomFacts))===bag(previous.map(atomFacts))&&bag(trace.sourceStateBefore.map(atomFacts))===bag(sourcePrevious.map(atomFacts)),'independent state continuity');
    for(const atoms of [trace.stateBefore,trace.stateAfter,trace.sourceStateBefore,trace.sourceStateAfter])demand(new Set(atoms.map(atomKey)).size===atoms.length,'state key/owner uniqueness');
    demand(trace.decisions.length>0,'actual decision contribution present');
    for(const d of trace.decisions){
     demand(d.eventId===event?.id&&ruleIds.has(d.ruleId),'guard event/rule identity');
     demand(d.matchesSource===true&&d.localDecision&&d.sourceLocalDecision,'independent comparison contributions');
     demand(d.generatedEvidence.length>0&&d.witnesses.length>0,'separate source/generated support');
     demand(equal(d.value,d.sourceRequiredValue)&&bag(d.effects.map(atomFacts))===bag(d.sourceRequiredEffects.map(atomFacts)),'declared channel consistency');
    }
    if(Object.hasOwn(oracle.permission_invariants,trace.eventId)){
     const expected=oracle.permission_invariants[trace.eventId];
     for(const state of [trace.stateAfter,trace.sourceStateAfter]){
      const atom=state.find(a=>a.key==='permission'&&a.subject===event.actor);
      demand(atom&&equal(atom.value,expected)&&atom.known===(expected!==null)&&atom.establishedAt===event.time,'aggregate permission state:'+trace.eventId);
     }
     attemptValues.set(trace.eventId,expected);
    }
    // Exact effects must be what the report applies; this checks plumbing, not effect entailment.
    for(const [before,effects,after]of[[trace.stateBefore,trace.decisions.flatMap(d=>d.effects),trace.stateAfter],[trace.sourceStateBefore,trace.decisions.flatMap(d=>d.sourceRequiredEffects),trace.sourceStateAfter]]){
     const applied=new Map(before.map(a=>[atomKey(a),atomFacts(a)]));
     for(const effect of effects)applied.set(atomKey(effect),atomFacts(effect));
     demand(bag(after.map(atomFacts))===bag([...applied.values()]),'exact returned-effect application');
    }
    previous=trace.stateAfter;sourcePrevious=trace.sourceStateAfter;
   }
  }
  demand(bag([...attemptValues.keys()])===bag(Object.keys(oracle.permission_invariants)),'permission attempt coverage');
  const allTraces=verdicts.flatMap(v=>v.traces),trace=id=>allTraces.find(t=>t.eventId===id),after=(id,key,subject)=>trace(id)?.stateAfter.find(a=>a.key===key&&a.subject===subject);
  demand(after('N_TRANSFER','custody','Bo')?.value==='ST'&&after('N_TRANSFER','custody','Ada')?.value===null,'actual unique custody transfer');
  demand(after('N_TRANSFER','briefAcknowledged','Bo')?.value===false,'personal brief not transferred');
  demand(after('EM_ATTEMPT','queueAccepted','Bo')?.value===false,'waiver not queue completion');
  demand(after('R_ATTEMPT','briefAcknowledged','Cy')?.known===false&&after('R_ATTEMPT','briefAcknowledged','Cy')?.value===null,'receipt preserves unknown readiness');
  demand(after('H_CLEAR','unclearedHazard','ST')?.value===false&&after('H_CLEAR','unclearedHazard','ST')?.establishedAt===8,'station-global clearance time');
  demand(after('P_PROPOSAL','custody','Ada')?.value==='ST'&&after('P_PROPOSAL','custody','Cy')?.value==='RT','proposal/rehearsal not actual ST transfer');
  for(const finding of r.audit.interfaces)demand(modules.has(finding.caller)&&modules.has(finding.callee)&&modules.get(finding.caller).plan.calls.includes(finding.callee),'interface reference domain');
  demand([...modules.values()].every(m=>m.plan.calls.every(callee=>r.audit.interfaces.some(f=>f.caller===m.path&&f.callee===callee))),'interface edge coverage');
  demand(r.audit.sourceComplete===true&&r.audit.scenarioComplete===true&&r.audit.contributionComplete===true,'authored complete audit');
  demand(r.unresolved.length===0&&r.program.unresolved.length===0&&r.model.unresolved.length===0,'unresolved compile packet');
 }catch(error){issues.push('malformed packet: '+error.message);}
 return{ok:issues.length===0,issues,semantic_review_required:true,scope:'Exact source/file/type/evidence/reference/declared scenario-state invariants only; emitted NL entailment, symbolic judgments and live compiler/generated callable execution remain separate.',training_admission:false};
}
