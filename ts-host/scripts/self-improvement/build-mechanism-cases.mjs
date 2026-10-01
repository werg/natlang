/** Promote generated-data failure mechanisms into declared source fixtures; provenance says which logic was synthesized. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';import {join} from 'node:path';import {createHash} from 'node:crypto';
const [index,output]=process.argv.slice(2);await mkdir(output,{recursive:true});
const incidents=(await readFile(join(index,'incidents.jsonl'),'utf8')).trim().split('\n').map(JSON.parse).filter(row=>row.split==='train'&&['program-improvement','decision-repair'].includes(row.route));
const definitions=[
 ['zero-fallback','number | null','number','return value||7;',[0,null,2,0,8,0,-3],v=>v??7,'Return 7 only for null, retaining zero and other numeric values.','undefined'],
 ['empty-reduction','number[]','number','return value.reduce((a,b)=>a+b);',[[],[1],[2,3],[],[-2,5],[],[-9]],v=>v.reduce((a,b)=>a+b,0),'Sum the numeric list; an empty list has sum zero.','reduce'],
 ['index-boundary','string[]','string','return value[value.length]??"";',[['a'],['b','c'],[],['d','e','f'],['g'],['h','i'],['j']],v=>v.at(-1)??'','Return the last string or an empty string for an empty list.','undefined'],
 ['comma-join','string[]','string','return value.join("");',[['a','b'],[],['c'],['d','e'],['f','g','h'],['i','j'],['k']],v=>v.join(','),'Join all strings with a comma separator.','join'],
 ['filtered-count','number[]','number','return value.length;',[[1,-1,0],[],[-2,-3],[4,5],[0,0],[-1,2,3],[9]],v=>v.filter(x=>x>0).length,'Count strictly positive numbers, excluding zero and negatives.','undefined'],
 ['string-suffix','string','boolean','return value.startsWith(".ts");',['a.ts','ts.a','.ts','b.js','c.ts','x','nested/file.ts'],v=>v.endsWith('.ts'),'Return whether the string ends with .ts.','missing'],
 ['compile-interface','number','number','const output:number="invalid";return output;',[0,1,2,3,4,7,9],v=>v*v,'Repair compilation and return the square of the input.','type'],
 ['helper-contract','number','number','return double(value);',[0,1,2,3,4,5,6],v=>v*2,'Return twice the input; repair helper behavior without changing the external signature.','type'],
 ['ascending-order','number[]','number[]','return [...value].sort((a,b)=>b-a);',[[3,1],[2,2,0],[],[-1,4],[5,2,8],[0,-4],[9,7]],v=>[...v].sort((a,b)=>a-b),'Return a new ascending numeric ordering, preserving duplicates and input values.','sort'],
 ['record-sum','{amount:number}[]','number','return 0;',[[{amount:2},{amount:3}],[],[{amount:0}],[{amount:-2},{amount:4}],[{amount:8}],[{amount:1},{amount:1}],[{amount:-3}]],v=>v.reduce((s,r)=>s+r.amount,0),'Sum all record amounts, including zero for an empty list.','undefined'],
 ['reject-regression','number','number','return value-1;',[0,1,2,5,7,9,12],v=>v-1,'Retain the correct decrement behavior. Reject changes that regress independently scored cases.','rejected'],
 ['helper-simplify','number','number','const first=value+1;const second=first-1;return second+1;',[0,1,2,3,6,8,10],v=>v+1,'Simplify the implementation while preserving increment behavior. Prefer smaller passing source.','type']
];
const rows=definitions.map(([family,type,returns,body,inputs,gold,goal,mechanism])=>{
 const id='mechanism-'+family,incident=incidents.find(row=>row.reason.toLowerCase().includes(mechanism))??incidents[0];
 const files={'main.ts':`${family==='helper-contract'?'import double from "./double";\n':''}export function solve(value:${type}):${returns}{${body}}`,...(family==='helper-contract'?{'double.ts':'export default function double(value:number):number{return value+2;}'}:{})};
 return {version:'natlang.improvement-case/1',id,family,files,contract:{entry:'main.ts',exportName:'solve',programId:id},control:family==='reject-regression',cases:inputs.map((v,i)=>({id:id+'-'+i,group:id+'-'+i,split:i<3?'train':i<5?'validation':'test',args:[v],expected:gold(v)})),policy:{maxExperiments:3,maxPopulation:3,mode:'structural',strategy:'adaptive',objective:family==='helper-simplify'?'source-size':'quality',goal,allowedFiles:Object.keys(files)},incidents:[{...incident,cluster:id,provenance:'promoted-failure-mechanism',originalCluster:incident.cluster}],sourceGroups:[id],provenance:{kind:'promoted-failure-mechanism',historicalSource:false,incident:incident.id,sourceRevision:createHash('sha256').update(JSON.stringify(files)).digest('hex'),independentOracle:'declared finite fixture'}};
});
await writeFile(join(output,'cases.jsonl'),rows.map(JSON.stringify).join('\n')+'\n');await writeFile(join(output,'teacher-cases.jsonl'),rows.slice(2).map(JSON.stringify).join('\n')+'\n');console.log(JSON.stringify({cases:rows.length,promoted:true}));
