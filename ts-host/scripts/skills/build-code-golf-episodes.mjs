#!/usr/bin/env node
/** Original Python code-golf tasks: correctness is checked before source-byte savings. */
import {mkdir,writeFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {validateEpisode} from '../../dist/skills/episode.js';
const hash=value=>createHash('sha256').update(typeof value==='string'?value:JSON.stringify(value)).digest('hex');
const r=(key,max)=>parseInt(hash(key).slice(0,8),16)%max;
const moduleText=(lines)=>'def solve(x):\n'+lines.map(line=>'    '+line).join('\n')+'\n';
const templates={
 sequences:[
  {id:'stable-unique',description:'Return the integers in first-occurrence order, removing duplicates.',source:()=>moduleText(['result = []','for item in x:','    if item not in result:','        result.append(item)','return result']),input:(key)=>Array.from({length:r(key,14)},(_,i)=>r(key+'/'+i,9)-4),gold:x=>[...new Set(x)]},
  {id:'rotate',description:c=>`Rotate an integer list left by ${c} positions; return [] for an empty list.`,source:c=>moduleText(['if len(x) == 0:','    return []',`amount = ${c} % len(x)`,'left = x[amount:]','right = x[:amount]','result = left + right','return result']),input:key=>Array.from({length:r(key,14)},(_,i)=>r(key+'/'+i,19)-9),gold:(x,c)=>x.length?[...x.slice(c%x.length),...x.slice(0,c%x.length)]:[]},
  {id:'window-sums',description:c=>`Return sums of consecutive length-${c} windows. Shorter lists return [].`,source:c=>moduleText(['result = []',`for start in range(len(x) - ${c} + 1):`,'    total = 0',`    for offset in range(${c}):`,'        total += x[start + offset]','    result.append(total)','return result']),input:key=>Array.from({length:r(key,16)},(_,i)=>r(key+'/'+i,21)-10),gold:(x,c)=>Array.from({length:Math.max(0,x.length-c+1)},(_,i)=>x.slice(i,i+c).reduce((a,b)=>a+b,0))},
 ],
 filtering:[
  {id:'above-threshold',description:c=>`Keep all integers strictly greater than ${c}, preserving order.`,source:c=>moduleText(['result = []','for value in x:',`    if value > ${c}:`,'        result.append(value)','return result']),input:key=>Array.from({length:r(key,17)},(_,i)=>r(key+'/'+i,17)-8),gold:(x,c)=>x.filter(v=>v>c)},
  {id:'bounded-range',description:c=>`Keep integers inclusively between -${c} and ${c}, preserving duplicates and order.`,source:c=>moduleText(['result = []','for value in x:',`    if value >= -${c} and value <= ${c}:`,'        result.append(value)','return result']),input:key=>Array.from({length:r(key,17)},(_,i)=>r(key+'/'+i,23)-11),gold:(x,c)=>x.filter(v=>v>=-c&&v<=c)},
  {id:'divisibility',description:c=>`Keep integers divisible by ${c}. Zero and negative multiples are included.`,source:c=>moduleText(['result = []','for value in x:',`    remainder = value % ${c}`,'    if remainder == 0:','        result.append(value)','return result']),input:key=>Array.from({length:r(key,19)},(_,i)=>r(key+'/'+i,31)-15),gold:(x,c)=>x.filter(v=>v%c===0)},
 ],
 aggregates:[
  {id:'count-above',description:c=>`Count integers strictly above ${c}.`,source:c=>moduleText(['count = 0','for value in x:',`    if value > ${c}:`,'        count = count + 1','return count']),input:key=>Array.from({length:r(key,21)},(_,i)=>r(key+'/'+i,19)-9),gold:(x,c)=>x.filter(v=>v>c).length},
  {id:'sum-positive',description:'Sum positive integers; zero and negatives contribute nothing.',source:()=>moduleText(['total = 0','for value in x:','    if value > 0:','        total = total + value','return total']),input:key=>Array.from({length:r(key,21)},(_,i)=>r(key+'/'+i,19)-9),gold:x=>x.reduce((s,v)=>s+Math.max(0,v),0)},
  {id:'sum-squares',description:'Return the sum of squares of all input integers, including negatives.',source:()=>moduleText(['total = 0','for value in x:','    square = value * value','    total = total + square','return total']),input:key=>Array.from({length:r(key,21)},(_,i)=>r(key+'/'+i,19)-9),gold:x=>x.reduce((s,v)=>s+v*v,0)},
 ],
 strings:[
  {id:'collapse-space',description:'Collapse ASCII spaces/tabs/newlines into single spaces, removing leading and trailing whitespace; preserve other characters.',source:()=>moduleText(['words = x.split()','result = ""','for index in range(len(words)):','    if index > 0:','        result += " "','    result += words[index]','return result']),input:key=>['',' a  b ','\twind\nrain  ','café   moon','🌙  雨','already neat'][r(key,6)],gold:x=>x.trim().split(/\s+/).filter(Boolean).join(' ')},
  {id:'reverse-words',description:'Reverse whitespace-separated word order and join with one ASCII space, preserving letters inside each word.',source:()=>moduleText(['words = x.split()','result = []','for index in range(len(words) - 1, -1, -1):','    result.append(words[index])','return " ".join(result)']),input:key=>['','blue green gold','  one\ttwo\nthree ','café moon 雨','🌙  cloud'][r(key,5)],gold:x=>x.trim().split(/\s+/).filter(Boolean).reverse().join(' ')},
  {id:'word-lengths',description:'Return Unicode code-point lengths of whitespace-separated words; an empty input produces [].',source:()=>moduleText(['words = x.split()','result = []','for word in words:','    size = len(word)','    result.append(size)','return result']),input:key=>['','ab cdef','café  雨','🌙 cloud','  a\nbb\tccc '][r(key,5)],gold:x=>x.trim().split(/\s+/).filter(Boolean).map(word=>[...word].length)},
 ],
 ordering:[
  {id:'ascending',description:'Sort input integers ascending, retaining duplicates.',source:()=>moduleText(['result = list(x)','for i in range(len(result)):','    for j in range(i + 1, len(result)):','        if result[j] < result[i]:','            temporary = result[i]','            result[i] = result[j]','            result[j] = temporary','return result']),input:key=>Array.from({length:r(key,16)},(_,i)=>r(key+'/'+i,17)-8),gold:x=>[...x].sort((a,b)=>a-b)},
  {id:'descending',description:'Sort input integers descending, retaining duplicates.',source:()=>moduleText(['result = sorted(x)','result.reverse()','return result']),input:key=>Array.from({length:r(key,16)},(_,i)=>r(key+'/'+i,17)-8),gold:x=>[...x].sort((a,b)=>b-a)},
  {id:'stable-argsort',description:'Return indices sorted by the input integer values, breaking ties by original index.',source:()=>moduleText(['pairs = []','for index in range(len(x)):','    pairs.append((x[index], index))','pairs.sort()','result = []','for pair in pairs:','    result.append(pair[1])','return result']),input:key=>Array.from({length:r(key,16)},(_,i)=>r(key+'/'+i,9)-4),gold:x=>x.map((v,i)=>[v,i]).sort((a,b)=>a[0]-b[0]||a[1]-b[1]).map(p=>p[1])},
 ],
 transformations:[
  {id:'flatten-one',description:'Flatten exactly one level of a list of integer lists, preserving order.',source:()=>moduleText(['result = []','for row in x:','    for value in row:','        result.append(value)','return result']),input:key=>Array.from({length:r(key,6)},(_,i)=>Array.from({length:r(key+'/len'+i,6)},(_,j)=>r(key+'/'+i+'/'+j,17)-8)),gold:x=>x.flat()},
  {id:'neighbor-differences',description:'Return next-minus-current differences for adjacent integers; lists of length zero or one produce [].',source:()=>moduleText(['result = []','for index in range(len(x) - 1):','    difference = x[index + 1] - x[index]','    result.append(difference)','return result']),input:key=>Array.from({length:r(key,18)},(_,i)=>r(key+'/'+i,17)-8),gold:x=>x.slice(1).map((v,i)=>v-x[i])},
  {id:'run-lengths',description:'Encode consecutive equal integers as [value,count] pairs. Return [] for an empty list.',source:()=>moduleText(['result = []','for value in x:','    if result and result[-1][0] == value:','        result[-1][1] += 1','    else:','        result.append([value, 1])','return result']),input:key=>Array.from({length:r(key,25)},(_,i)=>r(key+'/'+Math.floor(i/3),5)-2),gold:x=>{const out=[];for(const v of x){if(out.length&&out.at(-1)[0]===v)out.at(-1)[1]++;else out.push([v,1]);}return out;}},
 ],
};
const skill=(description)=>({'SKILL.md':`---\nname: python-golf\ndescription: ${JSON.stringify(description)}\n---\nPreserve the exact function contract and edge cases. Replace verbose accumulation loops with builtins, comprehensions and slicing where equivalent. Count UTF-8 source bytes, not claimed token counts. Never hardcode observed input/output pairs. Test empty inputs, duplicates, negative numbers and Unicode. Correctness precedes brevity.\n`});
export function createCodeGolfEpisodes({replicas=3,variants=['empty','metadata','distractor']}={}){
 if(!Number.isSafeInteger(replicas)||replicas<1||replicas>100||!variants.length||new Set(variants).size!==variants.length||variants.some(v=>!['empty','metadata','distractor'].includes(v)))throw Error('invalid allocation');
 const episodes=[];
 for(const [family,list] of Object.entries(templates))for(let replica=0;replica<replicas;replica++)for(const variant of variants){
  const id=`code-golf-v1-${family}-r${replica}-${variant}`;
  const cases=list.flatMap(template=>Array.from({length:2},(_,i)=>{
   const key=id+'/'+template.id+'/'+i,c=1+r(key,6),source=template.source(c),description=typeof template.description==='function'?template.description(c):template.description;
   const group='code-golf/v1/'+family+'/'+template.id,taskId='golf-'+hash(key).slice(0,20);
   const tests=Array.from({length:16},(_,n)=>{const input=n===0?template.input('edge-empty-search/'+template.id+'/'+n):template.input(key+'/input/'+n);return {id:taskId+'-'+n,group,input,expected:template.gold(input,c)};});
   const publicExample=tests[1].input;
   return {id:taskId,group,args:[JSON.stringify({description,reference_source:source,public_example_input:publicExample,contract:'Return a Python module defining solve(x); match the specified behavior over the declared input domain. Shorten UTF-8 source bytes only after preserving correctness.'})],
    expected:{schema:'natlang.skill-code-objective/1',id:taskId,revision:'1',description,functionName:'solve',cases:tests,sizeObjective:{referenceSource:source,bestKnownLowerBoundBytes:1}}};
  }));
  const episode={version:'natlang.skill-episode/1',id,family:'code-golf-'+family,split:'train',license:'project-generated',source_groups:[...new Set(cases.map(row=>row.group))],
   target:{kind:'improvement-case',entry:'solve.nl',source:{schema:'natlang.code-golf-target/1',id:family},files:{'solve.nl':'---\nargs: { problem: string }\nreturns: string\n---\nRead the problem JSON. Produce a short correct Python module defining solve(x), equivalent to reference_source on the described domain. Return raw Python source. Use helpful bound skills. All reference checks must pass before byte savings earn credit. Do not use test-specific lookups or make claims about measured runtime; this task scores source bytes.\n'}},
   library:variant==='empty'?{kind:'empty',skills:{}}:variant==='metadata'?{kind:'existing',skills:{'python-golf':skill('Unclassified notes; usually unnecessary.')}}:{kind:'existing',skills:{'formatting-notes':{'SKILL.md':'---\nname: formatting-notes\ndescription: Use for styling a document.\n---\nUse clear headings and spacious margins.\n'}}},
   support:{cases:cases.slice(0,4)},query:{cases:cases.slice(4)},operations:variant==='metadata'?['revise','select','test']:['create','revise','select','test'],limits:{maxSteps:6},
   provenance:{generator:'natlang.code-golf-corpus/1',metric:{schema:'natlang.skill-code-objective/1',kind:'python-source-bytes'},library_variant:variant,...(variant==='metadata'?{selection_design:'metadata-tuning'}:{}),size_lower_bound_semantics:'one byte is a conservative bound, not a claimed achievable optimum'}};
  const errors=validateEpisode(episode);if(errors.length)throw Error(JSON.stringify(errors));episodes.push(episode);
 }
 return episodes;
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const options={};for(let i=2;i<process.argv.length;i+=2){const key=process.argv[i].replace(/^--/,'');if(!['out','replicas','variants'].includes(key)||process.argv[i+1]===undefined)throw Error('invalid option');options[key]=process.argv[i+1];}
 if(!options.out)throw Error('--out required');const out=resolve(options.out),episodes=createCodeGolfEpisodes({replicas:Number(options.replicas??3),variants:(options.variants??'empty,metadata,distractor').split(',')});
 await mkdir(out,{recursive:true});const body=episodes.map(JSON.stringify).join('\n')+'\n';await writeFile(join(out,'code-golf-episodes.jsonl'),body,{flag:'wx'});
 const manifest={schema:'natlang.code-golf-corpus/1',families:Object.keys(templates),program_templates:Object.values(templates).flat().length,episodes:episodes.length,cases:episodes.reduce((n,e)=>n+e.support.cases.length+e.query.cases.length,0),sha256:hash(body),provider_calls:0,publication:'Task candidates; exact reference validation and model collection still required'};
 await writeFile(join(out,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(manifest));
}
