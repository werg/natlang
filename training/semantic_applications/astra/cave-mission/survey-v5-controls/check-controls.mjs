/** Exact typed item-witness controls. NL extraction and entailment are not executed here. */
import fs from 'node:fs';import path from'node:path';import{createRequire}from'node:module';import{pathToFileURL}from'node:url';
const repo=path.resolve(import.meta.dirname,'../../../../..');const require=createRequire(path.join(repo,'ts-host/package.json'));const ts=require('typescript');
const{TypeEnv,parseType}=await import(pathToFileURL(path.join(repo,'ts-host/dist/native/types.js')));const{coerce}=await import(pathToFileURL(path.join(repo,'ts-host/dist/native/values.js')));
const read=p=>JSON.parse(fs.readFileSync(p,'utf8')),fixture=read(new URL('item-witnesses.json',import.meta.url));
const source=fs.readFileSync(path.join(repo,'applications/cave-mission/types.ts'),'utf8');const ast=ts.createSourceFile('types.ts',source,ts.ScriptTarget.Latest,true),names={};for(const n of ast.statements)if(ts.isTypeAliasDeclaration(n))names[n.name.text]=parseType(n.type.getText(ast));const env=new TypeEnv(names);
const root=path.join(repo,fixture.fixture_root),manifest=read(path.join(root,'manifest.json')),clauses=new Map();for(const d of manifest.documents)for(const line of fs.readFileSync(path.join(root,d.path),'utf8').split('\n')){const m=line.match(/^\[([^\]]+)\]/);if(m)clauses.set(m[1],line);}
const canonical=v=>Array.isArray(v)?v.map(canonical):v&&typeof v==='object'?Object.fromEntries(Object.keys(v).sort().map(k=>[k,canonical(v[k])])):v;const eq=(a,b)=>JSON.stringify(canonical(a))===JSON.stringify(canonical(b));const bag=a=>a.map(x=>JSON.stringify(canonical(x))).sort();const pointer=(v,p)=>p.split('/').slice(1).reduce((a,k)=>a?.[k.replace(/~1/g,'/').replace(/~0/g,'~')],v);
function verify(c,value){const issues=[];try{coerce(value,parseType(c.type),env);function quotes(v){if(!v||typeof v!=='object')return;if(Array.isArray(v.evidence)){if(v.evidence.length===0)issues.push('nonempty item evidence');for(const q of v.evidence){const text=clauses.get(q.clauseId);if(!text||q.exactQuote.length===0||!text.includes(q.exactQuote))issues.push('original clause quote');}}for(const x of Object.values(v))if(Array.isArray(x))x.forEach(quotes);else if(x&&typeof x==='object')quotes(x);}quotes(value);for(const[p,e]of Object.entries(c.exact_invariants))if(!eq(pointer(value,p),e))issues.push('declared invariant '+p);for(const p of c.unordered_projections??[]){const rows=pointer(value,p.path);const project=x=>Object.fromEntries(p.fields.map(k=>[k,x[k]]));if(!eq(bag(rows.map(project)),bag(p.expected)))issues.push('declared unordered projection '+p.path);}}catch(e){issues.push('type or shape: '+e.message);}return{ok:issues.length===0,issues};}
const checks=[];function check(id,name,expected,mutate){const c=fixture.cases.find(x=>x.id===id),value=structuredClone(c.value);mutate?.(value);const result=verify(c,value);checks.push({id,name,expected,actual:result.ok,passed:result.ok===expected,issues:result.issues});}
for(const c of fixture.cases)check(c.id,'authored typed item witness',true);
check('closed-frame-alternatives','unordered convention and evidence collections',true,v=>{v.alternatives.reverse();for(const a of v.alternatives)a.evidence.reverse();});
check('present-forward-gate','provenance subset quotation',true,v=>v.evidence[0].exactQuote='neither permits a return from a landing toward J.');
check('present-forward-gate','reverse prohibition is not physical absence',false,v=>v.assertion.physicalExistence='absent');
check('absent-dead-end-chimney','nonexistent route gains invented action cost',false,v=>v.assertion.cost=2);
check('local-origin','local origin replaced by global point',false,v=>v.meaning.position={x:0,y:1});
check('closed-frame-alternatives','missing supported reflection',false,v=>v.alternatives.pop());
check('closed-frame-alternatives','complete ambiguity becomes blocking unknown',false,v=>v.unresolved.push('choose one reflection'));
check('echo-scope','sensor cost invented',false,v=>v.meaning.cost=0);
check('glow-scope','source sensor identity replaced',false,v=>v.meaning.sensorId='ECHO');
check('full-carrier-tunnel-scope','missing item evidence',false,v=>v.evidence=[]);
check('present-forward-gate','wrong quote identity',false,v=>v.evidence[0].clauseId='NO_CLAUSE');
check('global-junction','invented supporting quote',false,v=>v.evidence[0].exactQuote='Not in source');
console.log(JSON.stringify({scope:'Authored per-item type/provenance/declared invariant controls; no NL execution or semantic-entailment proof',passed:checks.every(x=>x.passed),positive:checks.filter(x=>x.expected).length,negative:checks.filter(x=>!x.expected).length,checks},null,2));if(checks.some(x=>!x.passed))process.exitCode=1;
