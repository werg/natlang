/** Trusted exact fixture invariant checker. No probe, diagnosis or repair policy is implemented here. */
import * as lab from '../../../../applications/protocol-meaning-debugger/host/measurementLab.ts';
const normalize=x=>x===undefined?{$missing:true}:Array.isArray(x)?x.map(normalize):x&&typeof x==='object'?Object.fromEntries(Object.entries(x).sort().map(([k,v])=>[k,normalize(v)])):x;
const canonical=x=>JSON.stringify(normalize(x));
const same=(a,b)=>canonical(a)===canonical(b);
const bag=xs=>JSON.stringify((xs??[]).map(canonical).sort());
function rawTrial(t){return{id:t?.id,units:(t?.units??[]).map(u=>({unitId:u.unitId,firstTelemetryTime:u.firstTelemetryTime,commissioningEvents:u.commissioningEvents.map(({eventId,time,accepted})=>({eventId,time,accepted})),calibrationEvents:u.calibrationEvents.map(({eventId,time,accepted})=>({eventId,time,accepted}))})).sort((a,b)=>a.unitId.localeCompare(b.unitId)),readings:(t?.readings??[]).map(({id,unitId,time,outcome})=>({id,unitId,time,outcome})).sort((a,b)=>a.id.localeCompare(b.id))};}
export function verifyCandidate({value,oracle,inputs,fixture}){
 const issues=[];const demand=(ok,code)=>{if(!ok)issues.push(code);};
 const input=inputs?.input,request=inputs?.request;
 if(!input||!request||!value)return{ok:false,issues:['Missing typed input or report.'],semantic_review_required:true};
 demand(value.caseId===oracle.caseId&&request.caseId===input.caseId,'case-identity');
 demand(input.documents.every(d=>fixture?.[d.path]===d.text),'source-fixture-identity');
 const trials=Object.fromEntries([input.baseline,...input.probeSeeds.map(s=>s.data)].map(t=>[t.id,t]));
 function checkState(state,invariants,label,documents){
  if(!state||!invariants){demand(false,label+':missing-state');return;}
  demand(same(rawTrial(state.trial),rawTrial(trials[state.trial?.id])),label+':raw-trial-identity');
  demand(bag(state.receipts?.map(r=>r.implementationId))===bag(['TARGET','L','U']),label+':receipt-owner-set');
  demand(bag((state.lineage?.links??[]).map(({implementationId,consumerClauseId,definitionClauseId,role})=>({implementationId,consumerClauseId,definitionClauseId,role})))===bag(oracle.lineage_link_invariants),label+':selected-definition-links');
  demand(oracle.selected_clause_ids.every(id=>(state.lineage?.selectedClauses??[]).filter(c=>c.id===id).length===1)&&!state.lineage?.selectedClauses?.some(c=>c.id==='OLD_SUCCESS'),label+':selected-clause-identities');
  for(const [field,owner]of[['target','TARGET'],['left','L'],['right','U']]){
   const p=state[field],expected=invariants[owner];if(!p){demand(false,label+':missing-'+owner);continue;}
   demand(p.implementationId===owner,label+':plan-owner:'+owner);
   demand(bag(p.population?.includedUnitIds)===bag(expected.includedUnitIds),label+':population:'+owner);
   demand(p.population?.definitionId===expected.populationDefinitionId,label+':population-definition:'+owner);
   demand(bag((p.timing?.anchorByUnit??[]).map(({unitId,time,eventId})=>({unitId,time,eventId})))===bag(expected.anchors),label+':anchors:'+owner);
   demand(p.metric?.numeratorOutcome===expected.executable.numeratorOutcome&&p.metric?.denominator===expected.metricDenominator&&p.metric?.aggregation===expected.metricAggregation&&p.metric?.definitionId===expected.metricDefinitionId,label+':metric:'+owner);
   demand(same(p.executable?.window,expected.executable.window)&&same(p.executable?.anchorByUnit,expected.executable.anchorByUnit)&&bag(p.executable?.includedUnitIds)===bag(expected.includedUnitIds)&&p.executable?.numeratorOutcome===expected.executable.numeratorOutcome,label+':executable:'+owner);
   const receipts=state.receipts?.filter(r=>r.implementationId===owner)??[];demand(receipts.length===1,label+':receipt-cardinality:'+owner);
   if(receipts.length===1){const r=receipts[0];demand(r.trialId===state.trial?.id&&bag(r.selectedReadingIds)===bag(expected.receipt.selectedReadingIds)&&r.numerator===expected.receipt.numerator&&r.denominator===expected.receipt.denominator&&r.percent===expected.receipt.percent,label+':exact-numeric:'+owner);}
   if(p.executable)try{const actual=lab.evaluate(trials[state.trial.id],p.executable);demand(bag(actual.selectedReadingIds)===bag(expected.receipt.selectedReadingIds)&&actual.numerator===expected.receipt.numerator&&actual.denominator===expected.receipt.denominator&&actual.percent===expected.receipt.percent,label+':exact-service-replay:'+owner);}catch{demand(false,label+':nonexecutable:'+owner);}
  }
  function witnesses(x){if(!x||typeof x!=='object')return;if('sourceId'in x&&'path'in x&&'quote'in x)demand(lab.checkWitnesses(documents,[x]).every(c=>c.valid),label+':exact-witness');for(const child of Object.values(x))if(typeof child==='object'){if(Array.isArray(child))child.forEach(witnesses);else witnesses(child);}}
  witnesses(state);
 }
 checkState(value.baseline,oracle.plan_invariants.BASE,'baseline',input.documents);
 const outcomes=value.outcomes??[];demand(bag(outcomes.map(o=>o.seedId))===bag(oracle.required_probe_ids),'required-probe-set');
 for(const outcome of outcomes){demand(!!outcome.localDecision,'probe-local-contribution:'+outcome.seedId);checkState(outcome.prediction?.plans,oracle.plan_invariants[outcome.seedId],'probe:'+outcome.seedId,input.documents);}
 const roots=value.diagnosis?.rootCauses??[];demand(bag(roots.map(c=>c.dimension))===bag(Object.keys(oracle.root_cause_definitions)),'root-cause-dimensions');
 for(const [dimension,id]of Object.entries(oracle.root_cause_definitions)){const found=roots.filter(c=>c.dimension===dimension);demand(found.length===1&&bag(found[0].definitionIds)===bag([id])&&found[0].probeIds?.length>0,'root-cause-identity:'+dimension);}
 const contrasts=value.contrasts??[];demand(bag(contrasts.map(c=>c.id))===bag(['CP_POP','CP_ANCHOR','CP_WINDOW','CP_METRIC']),'contrast-set');
 demand(contrasts.find(c=>c.id==='CP_WINDOW')?.relation==='equivalent'&&contrasts.filter(c=>c.id!=='CP_WINDOW').every(c=>c.relation==='different'),'semantic-role-relations');
 const repair=value.repair;demand(repair?.verified===true,'verified-repair');
 if(repair){
  const edits=repair.proposal?.edits??[];demand(edits.length===oracle.minimum_definition_edits&&new Set(edits.map(e=>e.path)).size===oracle.minimum_definition_edits,'minimal-definition-sites');
  demand(bag(edits.map(e=>e.targetDefinitionId))===bag(Object.values(oracle.root_cause_definitions)),'repair-definition-identities');
  demand(lab.checkRepairs(input.documents,edits,request.allowedRepairPaths).every(c=>c.valid),'exact-repair-spans');
  let edited=input.documents;try{edited=lab.applyDefinitions(input.documents,edits,request.allowedRepairPaths);}catch{demand(false,'repair-nonexecutable');}
  checkState(repair.baseline,oracle.repaired_plan_invariants.BASE,'repair:baseline',edited);
  demand(bag(repair.probes?.map(p=>p.seedId))===bag(oracle.required_probe_ids),'repair-probe-set');
  for(const prediction of repair.probes??[])checkState(prediction.plans,oracle.repaired_plan_invariants[prediction.seedId],'repair:'+prediction.seedId,edited);
  demand(bag(repair.findings?.map(f=>f.contrastId))===bag(['CP_POP','CP_ANCHOR','CP_METRIC'])&&repair.findings.every(f=>f.status==='satisfied'),'repair-coverage');
 }
 demand((value.unresolved??[]).length===0&&(value.diagnosis?.unresolved??[]).length===0,'unresolved-report');
 return{ok:issues.length===0,issues,semantic_review_required:true,semantic_witness_entailment:'requires independent source-grounded judgment',actual_literal_execution:'requires live trace',training_admission:false};
}
