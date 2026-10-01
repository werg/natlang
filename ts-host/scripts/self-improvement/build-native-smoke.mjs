/** Synthetic supplements with independently checked source defects; never counted as historical incidents. */
import {mkdir,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
const output=process.argv[2];if(!output)throw Error('usage: build-native-smoke.mjs OUTPUT_DIR');
const definitions=[
 {family:'numeric-rounding',body:'let count=0;for(let n=9007199254740992;n<9007199254740994;n++){count++;}return count;',type:'number',inputs:[0,2,5,8,13,21,34],expected:x=>x+1,goal:'Return the input incremented by one. Repair the observed stalled numeric loop.'},
 {family:'branch-repair',body:'return value<0 ? -value : -value;',type:'number',inputs:[-9,-2,0,3,11,-20,31],expected:Math.abs,goal:'Return the absolute value, including positive and negative inputs.'},
 {family:'aggregation',body:'return value.length;',type:'number[]',inputs:[[1,2],[5,0],[2,4,6],[],[-1,3],[7,8],[-3,-2]],expected:x=>x.reduce((a,b)=>a+b,0),goal:'Return the numeric sum, including empty arrays and negative values.'},
 {family:'normalization',body:'return value;',type:'string',returns:'string',inputs:[' A ','b ',' C','hello ',' WORLD',' Again ',' done '],expected:x=>x.trim().toLowerCase(),goal:'Trim leading/trailing whitespace and lowercase the input.'},
 {family:'compile-repair',body:'const result: number="wrong";return result;',type:'number',inputs:[1,2,3,7,9,13,21],expected:x=>x*2,goal:'Return twice the input while fixing the observed compilation error.'},
 {family:'retained-baseline',body:'return value+1;',type:'number',inputs:[0,1,2,8,12,17,29],expected:x=>x+1,goal:'Return the input incremented by one. Inspect evidence and retain correct source rather than inventing an improvement.',control:true}
];
const rows=[];
for(const definition of definitions){
 const files={'main.ts':`export function solve(value: ${definition.type}): ${definition.returns??'number'} {${definition.body}}`};
 const id='native-'+definition.family,source=createHash('sha256').update(JSON.stringify(files)).digest('hex');
 rows.push({version:'natlang.improvement-case/1',id,family:definition.family,control:definition.control??false,files,contract:{entry:'main.ts',exportName:'solve',programId:id},
 cases:definition.inputs.map((value,index)=>({id:id+'-'+index,group:id+'-'+index,split:index<3?'train':index<5?'validation':'test',args:[value],expected:definition.expected(value)})),
 policy:{maxExperiments:3,maxPopulation:3,mode:'structural',strategy:'adaptive',goal:definition.goal,allowedFiles:['main.ts']},sourceGroups:[id],
 incidents:[{id:'synthetic:'+source,cluster:id,reason:definition.goal,provenance:'synthetic-native-source-fixture'}],provenance:{kind:'synthetic-supplement',historicalIncident:false}});
}
await mkdir(output,{recursive:true});await writeFile(join(output,'cases.jsonl'),rows.map(JSON.stringify).join('\n')+'\n');console.log(JSON.stringify({cases:rows.length,synthetic:true}));
