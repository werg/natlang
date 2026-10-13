/** Exact row arithmetic, hash and text mechanics. Protocol meaning is supplied by NL stages. */
import {createHash} from 'node:crypto';
import type {ExecutablePlan, TrialData, NumericReceipt, ProtocolRecord, Witness, ExactCheck, RepairEdit} from '../types.js';
const hash=(text:string)=>createHash('sha256').update(text).digest('hex');
function sources(documents:ProtocolRecord[]):ExactCheck[]{
 const ids=new Set<string>(),paths=new Set<string>();
 return documents.map(d=>{const valid=!ids.has(d.id)&&!paths.has(d.path)&&hash(d.text)===d.sha256;ids.add(d.id);paths.add(d.path);return{id:`source:${d.id}`,valid,detail:valid?'Unique source identity and hash match.':'Source identity or hash mismatch.'};});
}
export function evaluate(data:TrialData,plan:ExecutablePlan):NumericReceipt{
 const unitIds=data.units.map(u=>u.unitId),readingIds=data.readings.map(r=>r.id);
 if(new Set(unitIds).size!==unitIds.length||new Set(readingIds).size!==readingIds.length)throw Error('Trial identity multiplicity.');
 if(new Set(plan.includedUnitIds).size!==plan.includedUnitIds.length||plan.includedUnitIds.some(id=>!unitIds.includes(id)))throw Error('Included unit identity is invalid.');
 if(!Number.isFinite(plan.window.start)||!Number.isFinite(plan.window.end)||plan.window.start>plan.window.end)throw Error('Invalid numeric interval.');
 for(const id of plan.includedUnitIds)if(!Number.isFinite(plan.anchorByUnit[id]))throw Error(`Missing finite anchor for ${id}.`);
 const selected=data.readings.filter(r=>{
  if(!unitIds.includes(r.unitId)||!Number.isFinite(r.time)||!['pass','fail'].includes(r.outcome))throw Error('Invalid typed reading.');
  if(!plan.includedUnitIds.includes(r.unitId))return false;
  const offset=r.time-plan.anchorByUnit[r.unitId];
  return (plan.window.includeStart?offset>=plan.window.start:offset>plan.window.start)&&
   (plan.window.includeEnd?offset<=plan.window.end:offset<plan.window.end);
 });
 const numerator=selected.filter(r=>r.outcome===plan.numeratorOutcome).length,denominator=selected.length;
 return{trialId:data.id,implementationId:plan.implementationId,selectedReadingIds:selected.map(r=>r.id),numerator,denominator,percent:denominator===0?null:100*numerator/denominator,sourceChecks:[{id:`numeric:${data.id}:${plan.implementationId}`,valid:true,detail:'Applied supplied identities, numeric anchors, interval and numerator outcome.'}]};
}
export function checkWitnesses(documents:ProtocolRecord[],witnesses:Witness[]):ExactCheck[]{
 return[...sources(documents),...witnesses.map((w,i)=>{const d=documents.find(d=>d.id===w.sourceId&&d.path===w.path);const valid=!!d&&w.quote.length>0&&d.text.includes(w.quote);return{id:`witness:${i}:${w.sourceId}`,valid,detail:valid?'Exact quotation matches source.':'Source identity or quotation mismatch.'};})];
}
function spans(documents:ProtocolRecord[],edits:RepairEdit[],allowedPaths:string[]){return edits.map(edit=>{
 const d=documents.find(d=>d.id===edit.sourceId&&d.path===edit.path),start=d?.text.indexOf(edit.oldQuote)??-1;
 return{edit,d,start,end:start+edit.oldQuote.length,valid:!!d&&allowedPaths.includes(edit.path)&&edit.oldQuote.length>0&&start>=0&&d.text.indexOf(edit.oldQuote,start+1)===-1};
});}
export function checkRepairs(documents:ProtocolRecord[],edits:RepairEdit[],allowedPaths:string[]):ExactCheck[]{
 const entries=spans(documents,edits,allowedPaths),seen=new Set<string>();
 return[...sources(documents),...entries.map(e=>{const overlap=entries.some(x=>x!==e&&x.edit.path===e.edit.path&&x.start<e.end&&e.start<x.end);const valid=e.valid&&!overlap&&!seen.has(e.edit.id);seen.add(e.edit.id);return{id:`edit:${e.edit.id}`,valid,detail:valid?'Unique authorized nonoverlapping original span.':'Exact edit check failed.'};})];
}
export function applyDefinitions(documents:ProtocolRecord[],edits:RepairEdit[],allowedPaths:string[]):ProtocolRecord[]{
 const bad=checkRepairs(documents,edits,allowedPaths).find(c=>!c.valid);if(bad)throw Error(`Exact definition application failed: ${bad.id}`);
 const entries=spans(documents,edits,allowedPaths);
 return documents.map(d=>{let text=d.text;for(const e of entries.filter(e=>e.edit.path===d.path).sort((a,b)=>b.start-a.start))text=text.slice(0,e.start)+e.edit.replacement+text.slice(e.end);return{...d,text,sha256:hash(text)};});
}
