/** Exact type, lexical identity, quote provenance and one-item progress; no semantic interpreter. */
import fs from 'node:fs';import path from 'node:path';import{createRequire}from'node:module';import{pathToFileURL}from'node:url';
const repo=path.resolve(import.meta.dirname,'../../../../..'),require=createRequire(path.join(repo,'ts-host/package.json')),ts=require('typescript');
const{TypeEnv,parseType}=await import(pathToFileURL(path.join(repo,'ts-host/dist/native/types.js'))),{coerce}=await import(pathToFileURL(path.join(repo,'ts-host/dist/native/values.js')));
const ast=ts.createSourceFile('types.ts',fs.readFileSync(path.join(repo,'applications/cave-mission/types.ts'),'utf8'),ts.ScriptTarget.Latest,true),names={};for(const n of ast.statements)if(ts.isTypeAliasDeclaration(n))names[n.name.text]=parseType(n.type.getText(ast));const env=new TypeEnv(names);
const authored=JSON.parse(fs.readFileSync(new URL('progress-witness.json',import.meta.url),'utf8'));
const canonical=v=>Array.isArray(v)?v.map(canonical):v&&typeof v==='object'?Object.fromEntries(Object.keys(v).sort().map(k=>[k,canonical(v[k])])):v,eq=(a,b)=>JSON.stringify(canonical(a))===JSON.stringify(canonical(b));
const word=c=>c!==undefined&&/[\p{L}\p{N}_]/u.test(c);function token(text,name){if(!name)return false;for(let i=text.indexOf(name);i>=0;i=text.indexOf(name,i+1))if(!word(text[i-1])&&!word(text[i+name.length]))return true;return false;}
function verify(before,after,trace=null){const issues=[];try{coerce(before,parseType('FrameReadFrontier'),env);coerce(after,parseType('FrameReadFrontier'),env);
 if(!eq(before.clauses,after.clauses))issues.push('immutable original clauses');
 if(!eq(before.completed,after.completed.slice(0,before.completed.length)))issues.push('prior actual returns preserved');
 if(after.completed.length!==before.completed.length+1||!eq(after.pendingUnits,before.pendingUnits.slice(1)))issues.push('one original unit consumed');
 const unit=before.pendingUnits[0],row=after.completed.at(-1),by=new Map(before.clauses.map(c=>[c.id,c]));
 if(!unit.sourceFrame)issues.push('original frame scope required');
 if(unit.kind==='landmark'&&!eq(unit,row.unit))issues.push('original landmark unit retained');
 if(unit.kind==='frame'){
  if(!trace||!eq(row.unit,trace.actual_support_selection)||!eq({...row.unit,supportingClauseIds:unit.supportingClauseIds},unit)||!unit.supportingClauseIds.every(id=>row.unit.supportingClauseIds.includes(id))||!row.unit.supportingClauseIds.every(id=>by.has(id)))issues.push('actual frame support selection retained');
  if(!eq(row.convention,trace?.actual_child_return))issues.push('actual convention child return retained');
 }
 if(unit.kind==='landmark'&&!unit.primaryClauseIds.some(id=>token(by.get(id)?.text??'',unit.originalLocalName)))issues.push('complete case-preserved original name token');
 if(row.status==='accepted'&&unit.kind==='frame'){
  if(row.failure!==null||row.landmark!==null||!row.convention?.alternatives.length||row.landmarks.length)issues.push('accepted actual convention');
  const projected=(row.convention?.alternatives??[]).map((a,i)=>{const ws=[];for(const e of a.evidence){const c=by.get(e.clauseId);if(!c||!e.exactQuote||!c.text.includes(e.exactQuote)||![...row.unit.primaryClauseIds,...row.unit.supportingClauseIds].includes(e.clauseId))issues.push('actual frame supplied quote');else ws.push({...c.witnesses[0],quote:e.exactQuote});}if(!a.evidence.length||a.sourceFrame!==unit.sourceFrame)issues.push('frame source scope/support');return{...Object.fromEntries(['sourceFrame','origin','uVector','vVector'].map(k=>[k,a[k]])),witnesses:ws};});
  if(!eq(row.alternatives.map(({alternativeId,...v})=>v),projected)||new Set(row.alternatives.map(v=>v.alternativeId)).size!==projected.length)issues.push('projection of actual returned frame alternatives');
 }else if(row.status==='accepted'){
  if(row.failure!==null||!row.landmark?.meaning||row.convention!==null)issues.push('accepted actual typed landmark return');
  const actual=row.landmark?.meaning,ev=row.landmark?.evidence??[];
  if(actual?.localId!==unit.originalLocalName||actual?.frame!==unit.sourceFrame)issues.push('original name/frame binding');
  if(!ev.length)issues.push('nonempty original supporting quotation');
  const ws=[];for(const e of ev){const c=by.get(e.clauseId);if(!c||!e.exactQuote||!c.text.includes(e.exactQuote)||![...unit.primaryClauseIds,...unit.supportingClauseIds].includes(e.clauseId))issues.push('actual supplied clause quote');else ws.push({...c.witnesses[0],quote:e.exactQuote});}
  if(!eq(row.landmarks,[{...actual,witnesses:ws}])||row.alternatives.length)issues.push('projection of actual child fields and parent-bound witnesses');
 }else{
  if(row.alternatives.length||row.landmarks.length)issues.push('no unsupported projection');
  if(row.status==='failed'&&!row.failure)issues.push('actual failure reason');
  if(row.status==='unresolved'&&!row.unresolved.length)issues.push('actual unresolved premise');
 }
}catch(e){issues.push('type or shape: '+e.message);}return{ok:!issues.length,issues};}
const checks=[];function check(name,expected,mutation){const before=structuredClone(authored.before),after=structuredClone(authored.after);mutation?.(before,after);const r=verify(before,after);checks.push({name,expected,actual:r.ok,passed:r.ok===expected,issues:r.issues});}
check('actual one-item typed return',true);
check('free description preserved through actual child projection',true,(_,a)=>{a.completed.at(-1).landmark.meaning.description='An imitation marking, with no sample';a.completed.at(-1).landmarks[0].description='An imitation marking, with no sample';});
check('exact supporting quote subset',true,(_,a)=>{const r=a.completed.at(-1);r.landmark.evidence[0].exactQuote='X';r.landmarks[0].witnesses[0].quote=r.landmark.evidence[0].exactQuote;});
check('failed item keeps prior two alternatives',true,(_,a)=>{Object.assign(a.completed.at(-1),{status:'failed',landmark:null,landmarks:[],failure:'Actual item return failed provenance validation'});});
check('unresolved item preserves typed null judgment',true,(_,a)=>{Object.assign(a.completed.at(-1),{status:'unresolved',landmark:{meaning:null,evidence:[],unresolved:['Original position is unresolved']},landmarks:[],unresolved:['Original position is unresolved']});});
check('invented local identifier rejected',false,(_,a)=>{a.completed.at(-1).landmark.meaning.localId='landmark-amber-X';a.completed.at(-1).landmarks[0].localId='landmark-amber-X';});
for(const name of ['amber-X','x','X_missing'])check('source token '+name,false,(b,a)=>{b.pendingUnits[0].originalLocalName=name;a.completed.at(-1).unit.originalLocalName=name;a.completed.at(-1).landmark.meaning.localId=name;a.completed.at(-1).landmarks[0].localId=name;});
check('missing original name',false,(b,a)=>{delete b.pendingUnits[0].originalLocalName;delete a.completed.at(-1).unit.originalLocalName;});
check('prior actual reflection changed',false,(_,a)=>a.completed[0].alternatives[0].uVector.x=-1);
check('prior raw child discarded',false,(_,a)=>a.completed[0].convention=null);
check('completed unit replayed',false,(_,a)=>a.completed.push(structuredClone(a.completed[0])));
check('pending unit not consumed',false,(b,a)=>a.pendingUnits=b.pendingUnits);
check('actual child missing',false,(_,a)=>a.completed.at(-1).landmark=null);
check('projection changes actual coordinate',false,(_,a)=>a.completed.at(-1).landmarks[0].position.x=2);
check('empty supporting evidence',false,(_,a)=>a.completed.at(-1).landmark.evidence=[]);
check('absent original quotation',false,(_,a)=>a.completed.at(-1).landmark.evidence[0].exactQuote='An invented source quotation');
check('fabricated witness path',false,(_,a)=>a.completed.at(-1).landmarks[0].witnesses[0].path='input');
check('failed item projects accepted rows',false,(_,a)=>{a.completed.at(-1).status='failed';a.completed.at(-1).failure='Actual type error';});
function frameCheck(name,expected,mutation){const t=structuredClone(authored.frame_transition);mutation?.(t.before,t.after);const r=verify(t.before,t.after,t);checks.push({name,expected,actual:r.ok,passed:r.ok===expected,issues:r.issues});}
frameCheck('actual frame support refinement and two returned alternatives',true);
frameCheck('frame generated IDs alpha rename',true,(_,a)=>a.completed[0].alternatives.forEach((v,i)=>v.alternativeId='renamed-'+i));
frameCheck('frame actual convention dropped',false,(_,a)=>a.completed[0].convention=null);
frameCheck('frame actual convention altered',false,(_,a)=>a.completed[0].convention.alternatives[0].origin.y=0);
frameCheck('frame supporting anchor lost',false,(_,a)=>a.completed[0].unit.supportingClauseIds=a.completed[0].unit.supportingClauseIds.filter(id=>id!=='F01'));
frameCheck('frame projection re-solved',false,(_,a)=>a.completed[0].alternatives[0].uVector.x=-1);
frameCheck('frame scope missing',false,(b,a)=>{b.pendingUnits[0].sourceFrame=null;a.completed[0].unit.sourceFrame=null;});
console.log(JSON.stringify({scope:authored.role,passed:checks.every(c=>c.passed),positive:checks.filter(c=>c.expected).length,negative:checks.filter(c=>!c.expected).length,checks},null,2));if(checks.some(c=>!c.passed))process.exitCode=1;
