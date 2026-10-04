#!/usr/bin/env node
/** Original JS/Python-to-specified-language translations, with exact bounded behavior checks. */
import {createHash} from 'node:crypto';
import {mkdir,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {validateEpisode} from '../../dist/skills/episode.js';
import {evaluateExpression,scoreSpecifiedTranslation} from '../../dist/skills/translation-objective.js';
const hash=x=>createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const v=name=>({variable:name}),c=constant=>({constant}),op=(name,...args)=>({op:name,args});
const families={
 arithmetic:[{id:'affine',tree:op('add',op('multiply',c(3),v('x')),v('y')),js:'3*x+y',py:'3*x+y'},{id:'difference',tree:op('subtract',v('x'),op('multiply',c(2),v('y'))),js:'x-2*y',py:'x-2*y'},{id:'shifted-product',tree:op('multiply',op('add',v('x'),c(2)),op('subtract',v('y'),c(1))),js:'(x+2)*(y-1)',py:'(x+2)*(y-1)'}],
 bounds:[{id:'lower-bound',tree:op('maximum',v('x'),c(0)),js:'Math.max(x,0)',py:'max(x,0)'},{id:'upper-bound',tree:op('minimum',v('y'),c(3)),js:'Math.min(y,3)',py:'min(y,3)'},{id:'clamp',tree:op('minimum',c(4),op('maximum',c(-2),v('x'))),js:'Math.min(4,Math.max(-2,x))',py:'min(4,max(-2,x))'}],
 distance:[{id:'absolute',tree:op('absolute',v('x')),js:'Math.abs(x)',py:'abs(x)'},{id:'negated-difference',tree:op('negate',op('subtract',v('x'),v('y'))),js:'-(x-y)',py:'-(x-y)'},{id:'manhattan',tree:op('add',op('absolute',v('x')),op('absolute',v('y'))),js:'Math.abs(x)+Math.abs(y)',py:'abs(x)+abs(y)'}],
};
export function createTranslationEpisodes({replicas=3,variants=['empty','metadata','distractor']}={}){
 if(!Number.isSafeInteger(replicas)||replicas<1||replicas>100||!variants.length||new Set(variants).size!==variants.length||variants.some(v=>!['empty','metadata','distractor'].includes(v)))throw Error('invalid allocation');
 const episodes=[];
 for(const [family,templates] of Object.entries(families))for(const source of ['javascript','python'])for(let replica=0;replica<replicas;replica++)for(const variant of variants){
  const id=`translation-v1-${source}-${family}-r${replica}-${variant}`;
  const cases=templates.map(template=>{
   const key=id+'/'+template.id,group=`translation/v1/${family}/${template.id}`;
   const mapping=Object.fromEntries(['add','subtract','multiply','minimum','maximum','negate','absolute'].map((name,i)=>['op'+hash(key+'/'+i).slice(0,6),name]));
   const language={operators:mapping},names=Object.fromEntries(Object.entries(mapping).map(([token,meaning])=>[meaning,token]));
   const translate=node=>'op' in node?{op:names[node.op],args:node.args.map(translate)}:node;
   const reference=translate(template.tree),canonical={operators:Object.fromEntries(Object.keys(names).map(name=>[name,name]))};
   const tests=Array.from({length:49},(_,i)=>({input:{x:Math.floor(i/7)-3,y:i%7-3},expected:evaluateExpression(template.tree,{x:Math.floor(i/7)-3,y:i%7-3},canonical)}));
   const task={schema:'natlang.skill-translation/1',id:'translation-'+hash(key).slice(0,20),revision:'1',language,tests};
   if(scoreSpecifiedTranslation(task,reference).quality!==1)throw Error('reference does not preserve behavior');
   const sourceProgram=source==='javascript'?`function solve(x,y){return ${template.js};}`:`def solve(x,y):\n    return ${template.py}\n`;
   return {id:task.id,group,args:[JSON.stringify({source_language:source,source_program:sourceProgram,target_language:language,grammar:{variable:'{variable:"x"} or {variable:"y"}',constant:'{constant:integer} with magnitude at most 1000',operator:'{op:token,args:[expressions]}',arity:'negate/absolute have one argument; other operators two. Subtract uses first minus second.',allocation:'At most 128 expression nodes and nesting depth 16 (root depth 0).',domain:'x and y are integers -3 through 3, inclusive; exact safe-integer arithmetic.'}})],expected:task};
  });
  const body='Read the supplied language specification, map operations by their meaning, preserve variable binding and argument order. Translate syntax structurally; never guess semantics from unfamiliar token spelling. Check zero, negative and boundary inputs. Do not hardcode input/output tables.';
  const library=variant==='empty'?{kind:'empty',skills:{}}:variant==='metadata'?{kind:'existing',skills:{'language-translation':{'SKILL.md':`---\nname: language-translation\ndescription: Unclassified notes, rarely useful.\n---\n${body}\n`}}}:{kind:'existing',skills:{'formatting':{'SKILL.md':'---\nname: formatting\ndescription: Use for prose styling.\n---\nChoose spacious headings.\n'}}};
  const episode={version:'natlang.skill-episode/1',id,family:`translation-${source}-${family}`,split:'train',license:'project-generated',source_groups:cases.map(r=>r.group),target:{kind:'improvement-case',entry:'solve.nl',source:{schema:'natlang.specified-translation-target/1',id:family},files:{'solve.nl':'---\nargs: { problem: string }\nreturns: string\n---\nTranslate the supplied source program into the fully specified target expression language. Return only a JSON expression as a string. Preserve exact behavior across the stated input domain. Use relevant bound skills.\n'}},library,support:{cases:cases.slice(0,2)},query:{cases:cases.slice(2)},operations:variant==='metadata'?['revise','select','test']:['create','revise','select','test'],limits:{maxSteps:6},provenance:{generator:'natlang.translation-corpus/1',metric:{schema:'natlang.skill-translation/1',kind:'specified-expression'},library_variant:variant,...(variant==='metadata'?{selection_design:'metadata-tuning'}:{}),scope:'bounded original expression-language tasks; not unrestricted whole-program translation'}};
  const errors=validateEpisode(episode);if(errors.length)throw Error(JSON.stringify(errors));episodes.push(episode);
 }
 return episodes;
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const [outArg]=process.argv.slice(2);if(!outArg)throw Error('usage: build-translation-episodes.mjs OUTPUT');
 const out=resolve(outArg);await mkdir(out,{recursive:true});const episodes=createTranslationEpisodes(),body=episodes.map(JSON.stringify).join('\n')+'\n';await writeFile(join(out,'translation-episodes.jsonl'),body,{flag:'wx'});
 const manifest={schema:'natlang.translation-corpus/1',episodes:episodes.length,cases:episodes.reduce((n,e)=>n+e.support.cases.length+e.query.cases.length,0),families:[...new Set(episodes.map(e=>e.family))],sha256:hash(body),provider_calls:0,publication:'Prepared task candidates only; no training admission'};await writeFile(join(out,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(manifest));
}
