#!/usr/bin/env node
/** Small, source-derived procurement research seed. Every resolver input belongs to a retrieved record. */
import { createHash } from 'node:crypto';
import { mkdirSync, openSync, writeFileSync, closeSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const metric={schema:'natlang.skill-research/1',kind:'research-classification'};
const sha=value=>createHash('sha256').update(typeof value==='string'?value:JSON.stringify(value)).digest('hex');
const labels=['eligible','ineligible'];

const ruleRecord=(standardCap=20000,safetyCap=60000,warehouseCap=30000,wording='standard')=>{
 const fields={effective:true,standardCap,safetyCap,warehouseCap,wording};
 const text=wording==='standard'
  ?`Fictional procurement rule P-22, effective for the filing date: only bids from suppliers with active registration on the filing date may use any route. a complete bid from an actively registered Northmere supplier qualifies under the standard route at or below ${standardCap} credits. A safety-critical item qualifies under the direct-award exception only with a signed waiver and a verified absence of equivalent suppliers, at or below ${safetyCap} credits. A warehouse contract qualifies when its depot is active and the amount is at or below ${warehouseCap} credits. If the bid is complete and all routes have been checked with none qualifying, classify it ineligible.`
  :`Fictional procurement directive P-23, current for this filing: every route requires a supplier whose registration is active on the filing date. Regular review is available to active Northmere vendors submitting completed bids whose total does not exceed ${standardCap} credits. The safety exception requires a critical item, a signed waiver, and a verified lack of equivalent suppliers; its ceiling is ${safetyCap} credits. A depot contract can use the warehouse route only when the depot is active and total value is no more than ${warehouseCap} credits. A completed bid that fails each examined route is ineligible.`;
 return{id:'rule-p22',type:'rule',title:`Procurement rule ${wording==='standard'?'P-22':'P-23'}`,fields,text};
};
const bidRecord=(id,amount,complete=true)=>({id:'bid-record',type:'bid',title:'Filed bid and intake sheet',fields:{bidId:id,supplierId:`vendor-${id}`,amount,complete},
 text:`Intake sheet for bid ${id} names supplier vendor-${id}; the filed amount is ${amount} credits. The application is ${complete?'complete':'incomplete'} and its totals were checked against the signed bid.`});
const supplierRecord=(id,region='Southmere',active=true)=>({id:'supplier-register',type:'supplier',title:'Supplier registry extract',fields:{supplierId:`vendor-${id}`,region,active},
 text:`Registry entry for vendor-${id} lists its registered business location as ${region}; registration status is ${active?'active':'inactive'} on the filing date.`});
const joinRecord=(id,bidFound=true,supplierFound=true)=>({id:'case-link',type:'join',title:'Intake identity reconciliation',fields:{bidId:id,supplierId:`vendor-${id}`,bidFound,supplierFound},
 text:`Identity reconciliation for bid ${id}: bid record found=${bidFound}; supplier record vendor-${id} found=${supplierFound}.`});
const routeRecord=(id,{safetyCritical=false,waiverSigned=false,noEquivalent=false,warehouseContract=false,depotActive=false,reviewComplete=true}={})=>({
 id:'route-review',type:'routes',title:'Alternative route review',fields:{bidId:id,safetyCritical,waiverSigned,noEquivalent,warehouseContract,depotActive,reviewComplete},
 text:`Alternative-route review for bid ${id}: item safety-critical=${safetyCritical}; signed safety waiver=${waiverSigned}; no-equivalent finding=${noEquivalent}; warehouse contract=${warehouseContract}; active depot=${depotActive}. Review is ${reviewComplete?'complete':'incomplete'}.`});

function readRecords(documents){
 if(!Array.isArray(documents))throw new Error('invalid host procurement dossier');
 const byType=new Map();
 for(const d of documents){if(!d||typeof d.type!=='string'||byType.has(d.type))throw new Error('invalid or duplicate host dossier record');byType.set(d.type,d.fields);}
 const rule=byType.get('rule'),bid=byType.get('bid'),supplier=byType.get('supplier'),join=byType.get('join'),routes=byType.get('routes');
 if(!rule||!bid||!supplier||!join)throw new Error('incomplete host procurement dossier');
 return {rule,bid,supplier,join,routes};
}

/** The oracle reads only facts stored in retrieved records; packet metadata is not an input. */
export function resolveProcurementDocuments(documents){
 const {rule,bid,supplier,join,routes}=readRecords(documents);
 if(rule.effective!==true||![rule.standardCap,rule.safetyCap,rule.warehouseCap].every(Number.isSafeInteger)||
    !Number.isSafeInteger(bid.amount)||bid.amount<0||typeof bid.complete!=='boolean'||typeof supplier.active!=='boolean'||
    typeof supplier.region!=='string'||typeof join.bidFound!=='boolean'||typeof join.supplierFound!=='boolean')
   throw new Error('malformed host procurement record facts');
 if(join.bidId!==bid.bidId||join.supplierId!==bid.supplierId||supplier.supplierId!==bid.supplierId)return null;
 if(join.bidFound!==true||join.supplierFound!==true||supplier.active!==true)return null;
 if(bid.complete!==true)return null;
 const standard=supplier.region==='Northmere'&&bid.amount<=rule.standardCap;
 if(standard)return 'eligible';
 if(!routes)return null;
 if(routes.bidId!==bid.bidId||typeof routes.reviewComplete!=='boolean'||typeof routes.safetyCritical!=='boolean'||typeof routes.waiverSigned!=='boolean'||typeof routes.noEquivalent!=='boolean'||typeof routes.warehouseContract!=='boolean'||typeof routes.depotActive!=='boolean')throw new Error('malformed host route review facts');
 if(routes.reviewComplete!==true)return null;
 const safety=routes.safetyCritical===true&&routes.waiverSigned===true&&routes.noEquivalent===true&&bid.amount<=rule.safetyCap;
 const warehouse=routes.warehouseContract===true&&routes.depotActive===true&&bid.amount<=rule.warehouseCap;
 if(standard||safety||warehouse)return 'eligible';
 return 'ineligible';
}

const scenarios=[
 {id:'northmere-standard-within-cap',group:'procurement-v3/standard-route',rulePath:'standard',make(){const id='B-104';return[ruleRecord(),bidRecord(id,18400),supplierRecord(id,'Northmere'),joinRecord(id)];},
  mutations:[d=>d.find(x=>x.type==='rule').fields.standardCap=10000,d=>d.find(x=>x.type==='bid').fields.amount=20001,d=>d.find(x=>x.type==='supplier').fields.region='Southmere',d=>d.find(x=>x.type==='join').fields.bidFound=false]},
 {id:'waiver-safety-critical',group:'procurement-v3/signed-safety-exception',rulePath:'safety',make(){const id='B-227';return[ruleRecord(),bidRecord(id,52000),supplierRecord(id,'Southmere'),joinRecord(id),routeRecord(id,{safetyCritical:true,waiverSigned:true,noEquivalent:true})];},
  mutations:[d=>d.find(x=>x.type==='rule').fields.safetyCap=40000,d=>d.find(x=>x.type==='bid').fields.amount=61000,d=>d.find(x=>x.type==='routes').fields.waiverSigned=false,d=>d.find(x=>x.type==='supplier').fields.active=false,d=>d.find(x=>x.type==='join').fields.supplierFound=false]},
 {id:'complete-bid-no-qualifying-route',group:'procurement-v3/all-routes-screened',rulePath:'all-routes',make(){const id='B-318';return[ruleRecord(19000,59000,29000,'alternative'),bidRecord(id,25000),supplierRecord(id,'Northmere'),joinRecord(id),routeRecord(id)];},
  mutations:[d=>d.find(x=>x.type==='rule').fields.standardCap=30000,d=>d.find(x=>x.type==='bid').fields.amount=18000,d=>{const r=d.find(x=>x.type==='routes').fields;r.warehouseContract=true;r.depotActive=true;},d=>d.find(x=>x.type==='supplier').fields.active=false,d=>d.find(x=>x.type==='routes').fields.reviewComplete=false,d=>d.find(x=>x.type==='join').fields.bidFound=false]},
 {id:'active-warehouse-depot',group:'procurement-v3/warehouse-route',rulePath:'warehouse',make(){const id='B-441';return[ruleRecord(21000,61000,30000,'alternative'),bidRecord(id,26000),supplierRecord(id,'Southmere'),joinRecord(id),routeRecord(id,{warehouseContract:true,depotActive:true})];},
  mutations:[d=>d.find(x=>x.type==='rule').fields.warehouseCap=20000,d=>d.find(x=>x.type==='bid').fields.amount=31000,d=>d.find(x=>x.type==='routes').fields.depotActive=false,d=>d.find(x=>x.type==='supplier').fields.active=false,d=>d.find(x=>x.type==='join').fields.bidFound=false]},
];

export function renderResearchRecords(docs){return docs.map(d=>{
 const f=d.fields;let text;
 if(d.type==='rule')text=ruleRecord(f.standardCap,f.safetyCap,f.warehouseCap,f.wording).text;
 else if(d.type==='bid')text=bidRecord(f.bidId,f.amount,f.complete).text;
 else if(d.type==='supplier')text=supplierRecord(f.supplierId.replace(/^vendor-/u,''),f.region,f.active).text;
 else if(d.type==='join')text=joinRecord(f.bidId,f.bidFound,f.supplierFound).text.replace(`vendor-${f.bidId}`,f.supplierId);
 else if(d.type==='routes')text=routeRecord(f.bidId,f).text;
 else throw new Error(`unknown record type ${d.type}`);
 return{...d,text};
});}
function serviceSource(documents){
 const data=JSON.stringify(documents);
 return `type ResearchHit = { id: string; title: string; kind: string };\nconst PRIVATE_DOCUMENTS = ${data};\nexport function search(query: string): ResearchHit[] { const words=query.toLowerCase().split(/[^\\p{L}\\p{N}]+/u).filter(word=>word.length>2); return PRIVATE_DOCUMENTS.map(document=>({document,score:words.filter(word=>(document.title+' '+document.text).toLowerCase().includes(word)).length})).filter(row=>row.score>0).sort((a,b)=>b.score-a.score||a.document.id.localeCompare(b.document.id)).slice(0,10).map(row=>({id:row.document.id,title:row.document.title,kind:row.document.type})); }\nexport function read(sourceId: string): string { const document=PRIVATE_DOCUMENTS.find(row=>row.id===sourceId); if(!document)throw new Error('Unknown research source ID'); return document.text; }`;
}

function evidenceFor(doc,rulePath){
 if(doc.type==='rule'){
  const clause=doc.fields.wording==='standard'?(rulePath==='standard'?/a complete bid[^.]+credits\./u:rulePath==='safety'?/A safety-critical item[^.]+credits\./u:rulePath==='warehouse'?/A warehouse contract[^.]+credits\./u:/a complete bid[\s\S]*?all routes have been checked with none qualifying, classify it ineligible\./u):(rulePath==='standard'?/regular review is available[^.]+credits\./u:rulePath==='safety'?/The safety exception[^.]+credits\./u:rulePath==='warehouse'?/A depot contract[^.]+credits\./u:/every route requires a supplier whose registration is active[\s\S]*?fails each examined route is ineligible\./u);
  const match=doc.text.match(clause);if(!match)throw new Error(`rule citation does not cover ${rulePath}`);return{sourceId:doc.id,text:match[0]};
 }
 return{sourceId:doc.id,text:doc.text};
}
function buildCase(spec){
 const documents=renderResearchRecords(spec.make()),label=resolveProcurementDocuments(documents);
 const pivots=[];
 for(const mutate of spec.mutations){const altered=structuredClone(documents);mutate(altered);const changed=renderResearchRecords(altered);const next=resolveProcurementDocuments(changed);
  const changedIndex=changed.findIndex((d,index)=>JSON.stringify(d.fields)!==JSON.stringify(documents[index].fields));
  const visibleTextChanged=changedIndex>=0&&changed[changedIndex].text!==documents[changedIndex].text;
  if(!visibleTextChanged)throw new Error(`mutation in ${spec.id} changes ${changedIndex<0?'no record':changed[changedIndex].type} resolver facts without changing rendered record text`);
  if(next!==label)pivots.push({sourceId:changed[changedIndex].id,visibleTextChanged,changedLabel:next});}
 const requiredEvidence=documents.map(doc=>evidenceFor(doc,spec.rulePath));
 if(pivots.length!==spec.mutations.length)throw new Error(`nonpivotal mutation in ${spec.id}: ${pivots.length}/${spec.mutations.length}`);
 const packet={disclaimer:'Fictional project-generated procurement exercise; these rules have no real-world legal effect.',question:`Using retrieved records, classify procurement case ${sha(spec.id).slice(0,10)} under the effective fictional rule. Use an allowed label only when source records determine it; otherwise return null and unresolved=true.`,allowedLabels:labels,catalog:documents.map(({id,title,type})=>({id,title,kind:type}))};
 const expected={kind:'research-classification',label,unresolved:label===null,requiredEvidence,documents:documents.map(({id,text})=>({id,text}))};
 return{packet,expected,documents,pivots};
}

const target={kind:'improvement-case',entry:'solve.nl',source:{schema:'natlang.research-landscape/3',id:'fictional-procurement-dossiers-v3'},files:{'solve.nl':`---\nargs: { packet: string }\nreturns: string\n---\nThe packet names a fictional procurement case, allowed classification IDs, and a source catalog. Search and read records through research.search and research.read; do not infer facts from titles. Join the bid, supplier, and route records by their identifiers, then apply the rule record’s stated thresholds and conditions. Return JSON only: {"label":"eligible or ineligible","unresolved":false,"citations":[{"sourceId":"...","evidence":"contiguous text from that source"}]}. If the records do not establish a valid join or a completed review, return {"label":null,"unresolved":true,"citations":[...]}. Cite the rule and every case record needed for your decision.\n`}};

export function buildResearchLandscapeV3(){
 const built=scenarios.map(spec=>({spec,...buildCase(spec)}));
 return[{version:'natlang.skill-episode/1',id:'research-landscape-procurement-v3',family:'research-landscape-v3:procurement-eligibility',split:'train',
  source_groups:built.map(x=>x.spec.group),license:'project-generated',target,library:{kind:'empty',skills:{}},
  support:{cases:built.slice(0,2).map(({spec,packet,expected,documents})=>({id:`${spec.id}-support`,group:spec.group,args:[JSON.stringify(packet)],expected,services:{research:serviceSource(documents)}}))},
  query:{cases:built.slice(2).map(({spec,packet,expected,documents})=>({id:`${spec.id}-query`,group:spec.group,args:[JSON.stringify(packet)],expected,services:{research:serviceSource(documents)}}))},
  operations:['create','revise'],limits:{maxSteps:6},provenance:{generator:'natlang.research-landscape/3',metric,controlled_synthetic_seed:true,
   source_decisions:'resolver consumes only fields on retrieved records; each required evidence source has a validated single-record mutation that changes the host result',
   split_holdout:'The two route templates in query are absent from support; do not claim broad research-benchmark hardness from four fictional cases.'}}];
}

export function auditResearchLandscapeV3(){
 return scenarios.map(spec=>{const c=buildCase(spec);return{id:spec.id,group:spec.group,documents:c.documents.length,label:c.expected.label,
  requiredSources:[...new Set(c.expected.requiredEvidence.map(x=>x.sourceId))],mutations:c.pivots.map(x=>({sourceId:x.sourceId,visibleTextChanged:x.visibleTextChanged,changesResult:x.changedLabel!==c.expected.label,changedLabel:x.changedLabel}))};});
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const at=process.argv.indexOf('--out'),arg=at>=0?process.argv[at+1]:undefined;
 if(!arg||arg.startsWith('--'))throw Error('Usage: build-research-landscape-v3.mjs --out DIR');
 const episode=buildResearchLandscapeV3()[0],out=resolve(arg);mkdirSync(out,{recursive:true});
 const body=JSON.stringify(episode)+'\n',fd=openSync(join(out,'research-landscape-v3.jsonl'),'wx');try{writeFileSync(fd,body)}finally{closeSync(fd)}
 const audit=auditResearchLandscapeV3(),manifest={schema:'natlang.research-landscape/3',episodes:1,uniqueScenarios:audit.length,supportCases:episode.support.cases.length,queryCases:episode.query.cases.length,audit,sha256:sha(body),model_calls:0,
  note:'Small controlled fictional procurement seed. Host output is derived from record-owned facts; every reference source passed an individual fact mutation. This is not a broad or empirically measured research benchmark.'};
 const mf=openSync(join(out,'manifest.json'),'wx');try{writeFileSync(mf,JSON.stringify(manifest,null,2)+'\n')}finally{closeSync(mf)}
 process.stdout.write(JSON.stringify(manifest)+'\n');
}
