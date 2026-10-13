/** Exact certificate checker for a supplied fictional ledger; no norm selection or policy generation. */
const canonical=v=>Array.isArray(v)?v.map(canonical):v&&typeof v==='object'?Object.fromEntries(Object.keys(v).sort().map(k=>[k,canonical(v[k])])):v;
const eq=(a,b)=>JSON.stringify(canonical(a))===JSON.stringify(canonical(b));
const setEq=(a,b)=>Array.isArray(a)&&Array.isArray(b)&&a.length===new Set(a).size&&b.length===new Set(b).size&&eq([...a].sort(),[...b].sort());
const key=k=>JSON.stringify([k.tripId,k.debtor,k.beneficiary,k.action]);
export function verifyCandidate({value:r,oracle,inputs,fixture}){
 const issues=[];const check=(ok,msg)=>{if(!ok)issues.push(msg);};
 try{
  const request=inputs.request??inputs;const manifest=JSON.parse(fixture['manifest.json']);
  check(r.caseId===oracle.caseId&&request.caseId===oracle.caseId&&manifest.caseId===oracle.caseId,'case identity');
  const docs=new Map(manifest.documents.map(d=>[d.id,{...d,text:fixture[d.path]}]));const clauseIds=new Set();
  for(const d of docs.values())for(const m of d.text.matchAll(/^\[([^\]]+)\]/gm))clauseIds.add(m[1]);
  function witnesses(v,path='report'){
   if(!v||typeof v!=='object')return;
   if('sourceId'in v&&'quote'in v){const d=docs.get(v.sourceId);check(d&&d.path===v.path&&typeof v.quote==='string'&&v.quote.length>0&&d.text.includes(v.quote),path+': witness');}
   for(const [k,x]of Object.entries(v))witnesses(x,path+'/'+k);
  }witnesses(r);
  check(setEq(r.sourceCoverage.map(x=>x.sourceId),[...docs.keys()]),'coverage documents');
  for(const d of docs.values()){const expected=[...d.text.matchAll(/^\[([^\]]+)\]/gm)].map(x=>x[1]);const row=r.sourceCoverage.find(x=>x.sourceId===d.id);check(row&&setEq(row.clauseIds,expected),'coverage '+d.id);}
  check(r.unresolved.length===0&&r.authority.unresolved.length===0&&r.compiled.unresolved.length===0,'unexpected unresolved source');
  const authority=new Map(r.authority.proofs.map(p=>[p.proofId,p]));check(authority.size===r.authority.proofs.length,'authority ID uniqueness');
  check(!r.authority.proofs.some(p=>['AL','BI'].includes(p.claimId)),'unsupported authority admitted');
  check(r.authority.rejected.some(x=>x.kind==='unsupported-cycle'&&setEq(x.claimIds,['AL','BI'])),'reciprocal authority rejection');
  check(['D01','O01','G01','G02'].every(id=>r.compiled.rejectedClauseIds.includes(id)),'rejected clause coverage');
  for(const p of r.authority.proofs){check(p.witnesses.length>0,'authority witnesses');for(const id of p.premiseProofIds)check(authority.has(id),'missing authority parent');if(p.holder==='F')check(eq(p.interval,oracle.effective_flood_interval),'board authority sunset');}
  const authorityActive=new Set(),authorityDone=new Set();function walkAuthority(id){if(authorityActive.has(id)){check(false,'cyclic authority proof');return;}if(authorityDone.has(id))return;authorityActive.add(id);for(const parent of authority.get(id)?.premiseProofIds??[])walkAuthority(parent);authorityActive.delete(id);authorityDone.add(id);}for(const id of authority.keys())walkAuthority(id);
  const norms=new Map(r.compiled.norms.map(n=>[n.normId,n]));check(norms.size===r.compiled.norms.length,'norm ID uniqueness');
  for(const n of r.compiled.norms){check(n.authorityProofIds.length>0&&n.authorityProofIds.every(id=>authority.has(id)),'norm authority references');if(n.issuer==='F'){check(['notice','passage'].includes(n.action),'board act scope');check(eq(n.effective,oracle.effective_flood_interval)&&setEq(n.reachScope,['Upper']),'board norm scope');}}
  check(r.compiled.norms.some(n=>n.effect==='rate'&&n.action==='toll'&&n.amount===12&&n.effective.startInclusive===20),'prospective rate');
  check(r.compiled.norms.some(n=>n.effect==='reserve'&&n.action==='translation'&&setEq(n.actorScope,['Birch'])&&n.effective.startInclusive===8),'reservation debtor scope');
  for(const n of r.compiled.norms)for(const target of n.targetNormIds)check(norms.has(target),'norm target reference');
  check(setEq(r.ledgers.map(l=>l.snapshot.id),request.snapshots.map(x=>x.id)),'snapshot set');
  const ledgerById=new Map(r.ledgers.map(l=>[l.snapshot.id,l]));
  const auditSeen=[];
  for(const l of r.ledgers){
   const sid=l.snapshot.id;const snap=request.snapshots.find(x=>x.id===sid);check(snap&&eq(l.snapshot,snap),'snapshot time');check(l.unresolved.length===0,'ledger unresolved');
   const expected=oracle.expected_entries.filter(e=>e.snapshot===sid),entries=new Map(l.entries.map(e=>[key(e.key),e]));
   check(setEq([...entries.keys()],expected.map(key))&&entries.size===l.entries.length,'business key coverage '+sid);
   const proofs=new Map(l.proofs.map(p=>[p.proofId,p])),defeats=new Map(l.defeats.map(d=>[d.decisionId,d]));
   check(proofs.size===l.proofs.length&&defeats.size===l.defeats.length,'proof/defeat ID uniqueness '+sid);
   function checkPremises(refs){for(const p of refs){if(p.kind==='proof')check(proofs.has(p.id),'proof premise domain');else if(p.kind==='authority')check(authority.has(p.id),'authority premise domain');else check(clauseIds.has(p.id),'source premise domain');if(sid==='S34')check(!['R01','R02'].includes(p.id),'future evidence leakage');}}
   for(const p of l.proofs){check(norms.has(p.normId),'proof norm domain');check(entries.has(key(p.key)),'proof business key');check(p.witnesses.length>0,'proof witness presence');checkPremises(p.premises);}
   for(const d of l.defeats){check(proofs.has(d.targetProofId)&&norms.has(d.defeatingNormId),'defeat reference domain');checkPremises(d.premises);}
   const visiting=new Set(),visited=new Set();function visit(id){if(visiting.has(id)){check(false,'cyclic justification');return;}if(visited.has(id))return;visiting.add(id);for(const q of proofs.get(id)?.premises??[])if(q.kind==='proof')visit(q.id);visiting.delete(id);visited.add(id);}for(const id of proofs.keys())visit(id);
   for(const gold of expected){const e=entries.get(key(gold));if(!e){check(false,'missing entry');continue;}
    check(e.status===gold.status,'status '+sid+key(gold));check(gold.amountAlternatives===null?e.amountAlternatives===null:setEq(e.amountAlternatives,gold.amountAlternatives),'amount '+sid+key(gold));check(e.permissionRequired===gold.permissionRequired,'permission '+sid+key(gold));
    check(e.supportingProofIds.length>0&&setEq(e.supportingProofIds,e.supportingProofIds),'entry support IDs');check(setEq(e.defeatingDecisionIds,e.defeatingDecisionIds),'entry defeat IDs');
    const owned=e.supportingProofIds.map(id=>proofs.get(id));check(owned.every(p=>p&&key(p.key)===key(e.key)),'entry proof ownership');
    const ds=e.defeatingDecisionIds.map(id=>defeats.get(id));check(ds.every(d=>d&&e.supportingProofIds.includes(d.targetProofId)),'entry defeat ownership');
    const outcomes=owned.map(p=>{if(!p)return null;const local=ds.filter(d=>d?.targetProofId===p.proofId);return p.supported===false||local.some(d=>d.defeated===true)?false:p.supported===true&&!local.some(d=>d.defeated===null)?true:null;});
    const derived=outcomes.includes(true)?'active':outcomes.includes(null)?'unresolved':owned.some(p=>p?.supported!==false)?'defeated':'unsupported';check(derived===e.status,'justification classification '+sid+key(e.key));
    const supportingClauses=new Set(owned.flatMap(p=>(p?.premises??[]).filter(x=>x.kind==='clause').map(x=>x.id)));
    check(gold.baseJustificationClauseIds.every(id=>supportingClauses.has(id)),'missing independent source support '+sid+key(e.key));
    if(e.key.action==='toll')check(owned.some(p=>p.amount===(e.key.tripId==='T1'?10:12)),'proof toll quantity');
    if(e.key.action==='passage')check(owned.every(p=>p.permission===true),'proof passage meaning');
    const audits=r.audits.filter(a=>a.snapshotId===sid&&key(a.key)===key(e.key));check(audits.length===1,'audit cardinality');
    if(audits.length===1){const a=audits[0];auditSeen.push(a);check(a.valid===true&&a.issues.length===0,'audit result');check(e.supportingProofIds.every(id=>a.visitedProofIds.includes(id))&&a.visitedProofIds.every(id=>proofs.has(id)),'audit proof coverage');check(a.authorityFindings.length>0&&a.authorityFindings.every(f=>f.permitted===true&&f.supportingProofIds.every(id=>authority.has(id))),'audit authority findings');}
   }
  }
  check(auditSeen.length===r.audits.length,'extraneous audits');
  const before=ledgerById.get('S34'),after=ledgerById.get('S35');const beforeEntries=new Map(before.entries.map(e=>[key(e.key),e])),afterEntries=new Map(after.entries.map(e=>[key(e.key),e]));
  check(setEq(r.deltas.map(d=>key(d.key)),[...beforeEntries.keys()]),'delta business coverage');
  for(const d of r.deltas){const old=beforeEntries.get(key(d.key)),next=afterEntries.get(key(d.key));check(d.beforeSnapshot==='S34'&&d.afterSnapshot==='S35'&&d.beforeStatus===old?.status&&d.afterStatus===next?.status,'delta snapshot/status');const changed=old?.status!==next?.status||!eq(old?.amountAlternatives,next?.amountAlternatives)||old?.permissionRequired!==next?.permissionRequired;check(d.changed===changed,'delta changed flag');
   check(d.invalidatedProofIds.every(id=>before.proofs.some(p=>p.proofId===id))&&d.newProofIds.every(id=>after.proofs.some(p=>p.proofId===id)),'delta proof references');
   if(changed)check(d.support&&setEq(d.support.newPremiseIds,['R01','R02'])&&d.support.minimality==='inclusion-minimal','changed premise support');
  }
 }catch(e){issues.push('malformed certificate: '+e.message);}
 return {ok:issues.length===0,issues,semantic_review_required:true,scope:'Exact entries, scope boundaries, proof references, alternative support and changed-premise identities; NL entailment, evidence interpretation and live contributions remain separate.'};
}
