/** Exact certificate/state/provenance checks for the authored closed world; not NL entailment. */
const canon=x=>Array.isArray(x)?x.map(canon):x&&typeof x==='object'?Object.fromEntries(Object.keys(x).sort().map(k=>[k,canon(x[k])])):x;
const eq=(a,b)=>JSON.stringify(canon(a))===JSON.stringify(canon(b));
const bag=x=>JSON.stringify(x.map(v=>JSON.stringify(canon(v))).sort());
const core=o=>Object.fromEntries(Object.entries(o).filter(([k])=>k!=='witnesses'&&k!=='reason'));
const objectFacts=o=>core(o);
const boardFacts=b=>({objects:bag(b.objects.map(objectFacts)),resources:bag(b.resources.map(core)),tokens:bag(b.tokens.map(t=>({id:t.id,holderCardId:t.holderCardId,status:t.status}))),unresolved:b.unresolved});
const partFacts=p=>({...core(p),objectValue:p.objectValue?objectFacts(p.objectValue):null});
const triggerFacts=d=>({...core(d),capacityTokenIds:[...d.capacityTokenIds].sort(),tokenIds:[...d.tokenIds].sort(),qualifyingPartIds:[...d.qualifyingPartIds].sort(),consequenceParts:bag(d.consequenceParts.map(partFacts))});
export function verifyCandidate({value:r,oracle:o,inputs,fixture}){
 const issues=[];const require=(condition,message)=>{if(!condition)issues.push(message);};
 try{
  require(r.caseId===o.caseId&&r.caseId===inputs.request.caseId,'case identity');
  const manifest=JSON.parse(fixture['manifest.json']);const documents=new Map(manifest.documents.map(d=>[d.id,{...d,text:fixture[d.path]}]));
  const clauses=new Map();for(const d of documents.values())for(const line of d.text.split('\n')){const m=line.match(/^\[([^\]]+)\]/);if(m)clauses.set(m[1],{sourceId:d.id,path:d.path,quote:line});}
  function witnessWalk(v){if(!v||typeof v!=='object')return;if('sourceId'in v&&'path'in v&&'quote'in v){const d=documents.get(v.sourceId);require(d&&d.path===v.path&&typeof v.quote==='string'&&v.quote.length>0&&d.text.includes(v.quote),'exact source witness');}for(const x of Object.values(v))if(Array.isArray(x))x.forEach(witnessWalk);else if(x&&typeof x==='object')witnessWalk(x);}witnessWalk(r);
  const supports=(v,id)=>v.witnesses.some(w=>w.sourceId===clauses.get(id)?.sourceId&&w.path===clauses.get(id)?.path&&clauses.get(id)?.quote.includes(w.quote)&&w.quote.length>0);
  const expectedCoverage=manifest.documents.map(d=>({sourceId:d.id,clauseIds:[...clauses.entries()].filter(([,c])=>c.sourceId===d.id).map(([id])=>id).sort()}));
  require(bag(r.sourceCoverage.map(c=>({...c,clauseIds:[...c.clauseIds].sort()})))===bag(expectedCoverage),'complete original clause coverage');
  require(r.unresolved.length===0&&!r.findings.some(f=>f.kind!=='defended'),'determinate audited certificate');
  require(r.announcements.length===1&&r.announcements[0].intentId===inputs.request.announcedIntentId,'announced intent identity');
  const announcement=r.announcements[0];require(bag(announcement.parts.map(p=>[p.partId,p.printedOrder,p.targetInstanceId,p.amount]))===bag([['PA',1,'A1',3],['PM',2,'M1',2]]),'announced parts');
  for(const [pid,controller]of Object.entries(o.capturedControllers))require(announcement.references.some(b=>b.partId===pid&&b.controllerBasis==='captured'&&b.capturedController===controller),'captured controller '+pid);
  const triggers=r.triggerDecisions;require(triggers.length===4,'four printed trigger judgments');
  const byCapacity=new Map();for(const d of triggers){require(d.capacityTokenIds.length===1,'trigger capacity ownership');const token=d.capacityTokenIds[0];require(!byCapacity.has(token),'trigger capacity uniqueness');byCapacity.set(token,d);}
  require(bag([...byCapacity.keys()])===bag(['RETURN','SCRIBE','MIRROR','REPRISAL']),'trigger capacity domain');
  for(const token of ['RETURN','SCRIBE','MIRROR','REPRISAL']){const d=byCapacity.get(token);if(!d)continue;const active=o.emittedTriggers.includes(token);require(d.eligible===active,'trigger eligibility '+token);require(eq(d.tokenIds,active?[token]:[]),'trigger consumption '+token);require(active?d.consequenceParts.length===1:d.consequenceParts.length===0,'trigger consequences '+token);}
  const ret=byCapacity.get('RETURN'),scribe=byCapacity.get('SCRIBE');
  require(ret?.controllerBasis==='last-known'&&ret?.recipient==='Rowan'&&ret?.priority===10,'return binding');require(scribe?.controllerBasis==='captured'&&scribe?.recipient==='Ilex'&&scribe?.priority===20,'scribe binding');
  require(eq(ret?.qualifyingPartIds,['PA'])&&eq(scribe?.qualifyingPartIds,['PA']),'qualifying resolved part');
  const roles=r.windows.map(w=>w.kind==='trigger'?[...byCapacity].find(([,d])=>d.ruleId===w.sourceIdentity)?.[0]:w.sourceIdentity);
  require(eq(roles,o.windowOrder),'source-governed window chronology');require(new Set(r.windows.map(w=>w.slotId)).size===r.windows.length,'window slot uniqueness');
  const initial=r.windows[0]?.before;require(initial&&eq(boardFacts(initial),boardFacts(o.initialBoardFacts)),'initial source board');
  let previous=initial;const obj=(b,id)=>b.objects.find(x=>x.instanceId===id);const spend=(b,ids)=>{for(const id of ids){const t=b.tokens.find(x=>x.id===id);require(t&&t.status==='available','one-use token '+id);if(t)t.status='spent';}};
  for(let i=0;i<r.windows.length;i++){
   const w=r.windows[i],role=roles[i];require(eq(boardFacts(w.before),boardFacts(previous))&&w.before.revision===previous.revision,'window state continuity');require(w.after.revision!==w.before.revision,'new committed revision');
   for(const b of [w.before,w.after]){require(new Set(b.objects.map(x=>x.instanceId)).size===b.objects.length,'object identity uniqueness');require(new Set(b.tokens.map(x=>x.id)).size===b.tokens.length,'token identity uniqueness');}
   const expected=structuredClone(w.before);
   if(role==='TAKE'){require(w.kind==='response'&&w.batch===null&&w.bundle===null,'response shape TAKE');obj(expected,'A1').controller='Rowan';}
   else if(role==='MOVE'){require(w.kind==='response'&&w.batch===null&&w.bundle===null,'response shape MOVE');obj(expected,'W1').attachedTo='M1';}
   else{
    require(w.bundle&&w.batch&&w.bundle.bundleId===w.batch.bundleId&&w.bundle.causeSlotId===w.slotId,'batch identity');
    require(w.bundle.beforeRevision===w.before.revision&&w.batch.beforeRevision===w.before.revision,'common before-batch revision');
    require(w.batch.unresolved.length===0&&w.batch.outcomes.every(p=>p.unresolved.length===0),'resolved batch');
    require(bag(w.batch.outcomes.map(x=>partFacts(x.part)))===bag(w.bundle.parts.map(partFacts)),'actual final parts consumed');
    const applied=structuredClone(w.before);const changed=new Set();
    for(const c of w.batch.objects){require(!changed.has(c.instanceId),'one merged object delta');changed.add(c.instanceId);const before=obj(applied,c.instanceId);require(c.before===null?!before:before&&eq(objectFacts(c.before),objectFacts(before)),'delta before object');require(c.after.instanceId===c.instanceId,'delta object identity');if(before)Object.assign(before,structuredClone(c.after));else applied.objects.push(structuredClone(c.after));}
    const changedResources=new Set();for(const c of w.batch.resources){require(!changedResources.has(c.player),'one merged resource delta');changedResources.add(c.player);const a=applied.resources.find(x=>x.player===c.player);require(a&&a.ink===c.beforeInk,'delta before resource');if(a)a.ink=c.afterInk;}
    spend(applied,w.batch.consumedTokenIds);spend(applied,w.triggers.filter(d=>d.eligible===true).flatMap(d=>d.tokenIds));
    require(eq(boardFacts(applied),boardFacts(w.after)),'actual declared deltas applied');
    if(role==='FORK'){
     require(w.kind==='announcement'&&w.bundle.intentId==='FORK'&&w.causeSlotId===null,'Fork identity');require(bag(w.triggers.map(triggerFacts))===bag(triggers.map(triggerFacts)),'actual trigger decisions retained');
     const current=new Map(announcement.parts.map(p=>[p.partId,{partId:p.partId,lineagePartId:p.partId,printedOrder:p.printedOrder,kind:'damage',targetInstanceId:p.targetInstanceId,amount:p.amount,recipient:null,objectValue:null,attachmentTarget:null,appliedRuleIds:[]} ]));
     for(const d of w.replacements){require(d.beforeParts.length===1&&d.afterParts.length===1&&eq(partFacts(d.beforeParts[0]),current.get(d.lineagePartId)),'rewrite actual before lineage');const before=d.beforeParts[0],after=d.afterParts[0];require(after.lineagePartId===d.lineagePartId&&after.partId===before.partId&&after.targetInstanceId===before.targetInstanceId,'rewrite lineage identity');
      if(d.applicable===false)require(eq(partFacts(before),partFacts(after))&&d.consumedTokenIds.length===0,'negative rewrite preserves current part');
      else{require(d.applicable===true,'decided replacement');if(after.kind==='damage')require(before.kind==='damage'&&after.amount===Math.max(0,before.amount-1)&&d.consumedTokenIds.length===0&&supports(d,'REP01'),'Cover exact reduction');else require(before.kind==='damage'&&after.kind==='growth'&&after.amount===before.amount&&eq(d.consumedTokenIds,['WARD'])&&supports(d,'REP02'),'Mantle exact conversion');require(after.appliedRuleIds.includes(d.ruleId),'actual applied rule history');}
      current.set(d.lineagePartId,partFacts(after));}
     require(bag([...current.values()])===bag(w.bundle.parts.map(partFacts)),'rewrite final lineage coverage');
     for(const p of w.batch.outcomes){const gold=o.parts[p.part.lineagePartId];require(gold&&p.actualDamage===gold.actualDamage&&p.actualGrowth===gold.growthAdded,'resolved actual quantity');}
     require(eq(w.batch.consumedTokenIds,['WARD']),'replacement capacity spent once');require(w.batch.departures.length===1&&w.batch.departures[0].instanceId==='A1'&&w.batch.departures[0].lastKnownController==='Rowan'&&eq(w.batch.departures[0].causePartIds,['PA']),'simultaneous departure provenance');
     obj(expected,'A1').damage=2;obj(expected,'A1').zone='discard';obj(expected,'M1').growth=1;spend(expected,['WARD','RETURN','SCRIBE']);
    }else if(role==='RETURN'){
     require(w.kind==='trigger'&&w.causeSlotId===r.windows[2].slotId&&w.bundle.intentId===ret.ruleId,'return causal identity');require(bag(w.bundle.parts.map(partFacts))===bag(ret.consequenceParts.map(partFacts)),'queued return consumed');require(w.batch.objects.length===1&&w.batch.resources.length===0&&w.batch.departures.length===0&&w.batch.consumedTokenIds.length===0,'return effect domain');expected.objects.push({...o.returnObjectFacts,witnesses:[]});
    }else if(role==='SCRIBE'){
     require(w.kind==='trigger'&&w.causeSlotId===r.windows[2].slotId&&w.bundle.intentId===scribe.ruleId,'scribe causal identity');require(bag(w.bundle.parts.map(partFacts))===bag(scribe.consequenceParts.map(partFacts)),'queued scribe consumed');require(w.batch.objects.length===0&&w.batch.resources.length===1&&w.batch.departures.length===0&&w.batch.consumedTokenIds.length===0,'scribe effect domain');expected.resources.find(x=>x.player==='Ilex').ink=1;
    }
   }
   require(eq(boardFacts(expected),boardFacts(w.after)),'source-governed window delta '+role);previous=w.after;
  }
  require(eq(boardFacts(r.finalBoard),boardFacts(previous))&&r.finalBoard.revision===previous.revision,'actual final board');
 }catch(error){issues.push('malformed certificate: '+error.message);}
 return {ok:issues.length===0,issues,semantic_review_required:true,training_admission:false,scope:'Exact source IDs/quotes, source case state invariants, supplied rewrite/delta replay, capacities and reference chronology. NL rule grouping, condition entailment, missing-proposal semantics and actual callable/literal execution remain separately reviewed.'};
}
