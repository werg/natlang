/** Exact typed progress/provenance controls; does not execute or grade NL meaning. */
import fs from 'node:fs';import path from 'node:path';import{createRequire}from'node:module';import{pathToFileURL}from'node:url';
const repo=path.resolve(import.meta.dirname,'../../../../..'),req=createRequire(path.join(repo,'ts-host/package.json')),ts=req('typescript');
const{TypeEnv,parseType}=await import(pathToFileURL(path.join(repo,'ts-host/dist/native/types.js'))),{coerce}=await import(pathToFileURL(path.join(repo,'ts-host/dist/native/values.js')));
const source=fs.readFileSync(path.join(repo,'applications/protocol-meaning-debugger/types.ts'),'utf8'),ast=ts.createSourceFile('types.ts',source,ts.ScriptTarget.Latest,true),defs={};for(const n of ast.statements)if(ts.isTypeAliasDeclaration(n))defs[n.name.text]=parseType(n.type.getText(ast));const env=new TypeEnv(defs);
const input=JSON.parse(fs.readFileSync(path.join(repo,'training/semantic_applications/sol/protocol-meaning-debugger/field-signal/input.json'),'utf8')),documents=input.documents;
const clause=(id)=>{for(const d of documents){const line=d.text.split('\n').find(x=>x.startsWith('['+id+']'));if(line)return{d,line};}throw Error('original clause absent '+id);};
const unit=(id)=>{const{d,line}=clause(id);return{unitId:'UNIT_'+id,sourceId:d.id,clauseId:id,exactQuote:line,focus:'Original permission or population definition',supportingClauseIds:[]};};
const reading=(id,kind,ownerId)=>{const{d,line}=clause(id);return{id,kind,ownerId,meaning:line,witnesses:[{sourceId:d.id,path:d.path,quote:line}]};};
const accepted=reading('C_POP','population','TARGET'),u1=unit('C_POP'),u2=unit('P_SCOPE');
const row={unitId:u1.unitId,sourceId:u1.sourceId,clauseId:u1.clauseId,status:'accepted',reading:accepted,failure:null,unresolved:[]};
const before={documents,pendingUnits:[u2],completed:[row],definitions:[accepted],unresolved:[]};
const eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b);const checks=[];
function verify(after,mode){const issues=[];try{coerce(after,parseType('DefinitionFrontier'),env);}catch(e){return{ok:false,issues:['type: '+e.message]};}
if(!eq(after.documents,before.documents))issues.push('immutable original documents');if(!eq(after.completed.slice(0,1),before.completed)||!eq(after.definitions.slice(0,1),before.definitions))issues.push('prior accepted progress preserved');if(after.pendingUnits.length!==0||after.completed.length!==2)issues.push('one original unit consumed');const r=after.completed[1];if(!r||r.unitId!==u2.unitId||r.sourceId!==u2.sourceId||r.clauseId!==u2.clauseId)issues.push('completed original identity');if(r?.status!==mode)issues.push('declared outcome');if(mode==='accepted'){if(r?.failure!==null||!r?.reading||after.definitions.length!==2||!eq(r.reading,after.definitions[1]))issues.push('actual accepted row');}else{if(r?.reading!==null||after.definitions.length!==1||r?.unresolved?.length===0||after.unresolved.length===0)issues.push('failed/unknown preserves progress and reason');if(mode==='failed'&&(typeof r?.failure!=='string'||r.failure.length===0))issues.push('actual failure');if(mode==='unresolved'&&r?.failure!==null)issues.push('semantic uncertainty differs from failure');}
for(const d of after.definitions){for(const w of d.witnesses){const original=documents.find(x=>x.id===w.sourceId);if(!original||original.path!==w.path||!w.quote||!original.text.includes(w.quote))issues.push('original quote identity');}if(d.witnesses.length===0)issues.push('nonempty source support');}return{ok:issues.length===0,issues};}
function state(mode){const r=mode==='accepted'?reading('P_SCOPE','probe','CONTROL'):null,reason=mode==='failed'?'Typed child return rejected at kind; repair only this original unit.':'Original role remains unresolved.';return{documents:structuredClone(documents),pendingUnits:[],completed:[structuredClone(row),{unitId:u2.unitId,sourceId:u2.sourceId,clauseId:u2.clauseId,status:mode,reading:r,failure:mode==='failed'?reason:null,unresolved:mode==='accepted'?[]:[reason]}],definitions:r?[structuredClone(accepted),r]:[structuredClone(accepted)],unresolved:mode==='accepted'?[]:[u2.unitId+': '+reason]};}
function check(name,mode,expected,mutate){const value=state(mode);mutate?.(value);const r=verify(value,mode);checks.push({name,expected,actual:r.ok,passed:r.ok===expected,issues:r.issues});}
check('accepted source probe role','accepted',true);check('semantic uncertainty preserves prior reading','unresolved',true);check('typed failure preserves prior reading','failed',true);
check('free semantic prose is not exact gold','accepted',true,v=>{v.definitions[1].meaning='Only source-permitted interventions, with their restrictions.';v.completed[1].reading=v.definitions[1];});
check('actual one-sentence error wording may vary','failed',true,v=>{v.completed[1].failure='Use the declared DefinitionMeaning kind union for this unit.';});
check('all-or-nothing erasure','failed',false,v=>{v.definitions=[];v.completed[0].reading=null;});
check('earlier accepted result overwritten','failed',false,v=>v.completed[0].reading.meaning='invented replacement');
check('prior source documents replaced','failed',false,v=>v.documents[0].text='replacement');
check('pending original unit retained','failed',false,v=>v.pendingUnits=[u2]);
check('failed unit omitted from coverage','failed',false,v=>v.completed.pop());
check('failed result has fabricated semantic reading','failed',false,v=>v.completed[1].reading=accepted);
check('unknown result falsely claimed transport failure','unresolved',false,v=>v.completed[1].failure='transport');
check('failed result no actual error','failed',false,v=>v.completed[1].failure=null);
check('unknown reason dropped','unresolved',false,v=>{v.completed[1].unresolved=[];v.unresolved=[];});
check('wrong original completed identity','failed',false,v=>v.completed[1].clauseId='C_POP');
check('invalid probe role fails typed boundary','accepted',false,v=>{v.definitions[1].kind='authorization';v.completed[1].reading=v.definitions[1];});
check('invented exact support quotation','accepted',false,v=>{v.definitions[1].witnesses[0].quote='Not in the source';v.completed[1].reading=v.definitions[1];});
check('source path reconstructed from label','accepted',false,v=>{v.definitions[1].witnesses[0].path='input';v.completed[1].reading=v.definitions[1];});
console.log(JSON.stringify({scope:'Private authored progress/type/provenance certificates only; no NL extraction, execution or entailment proof.',passed:checks.every(x=>x.passed),positive:checks.filter(x=>x.expected).length,negative:checks.filter(x=>!x.expected).length,checks},null,2));if(checks.some(x=>!x.passed))process.exitCode=1;
