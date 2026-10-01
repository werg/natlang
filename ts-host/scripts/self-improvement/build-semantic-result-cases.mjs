/** General semantic-result tasks grounded in observed redundant inspection/delegation. No invented replay. */
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {loadVirtualNatlang} from '../../dist/runtime/virtual-project.js';
const [output]=process.argv.slice(2);
if(!output)throw Error('usage: build-semantic-result-cases.mjs OUTPUT');
const source='runs/luna-bonsai-prepared-optimizer-20261001/exchanges.ndjson';
const hash=createHash('sha256').update(await readFile(new URL('../../../'+source,import.meta.url))).digest('hex');
const examples=[
 ['recommendation','boolean','Does text recommend the product overall? Judge meaning, including contrast, negation and sarcasm.',[
  ['It looked disappointing. After six months I would happily buy it again.',true],
  ['A beautiful box for an unusable machine. Avoid buying it.',false],
  ['The noise bothers me, but its usefulness wins and I recommend it.',true],
  ['I recommend its predecessor. This replacement is a mistake.',false],
  ['Not bad at all; I suggested it to all my friends.',true],
  ['What a delight: another broken device. I returned it.',false]]],
 ['ticket-intent',"'refund' | 'repair'",'Classify the requested resolution in text as refund or repair. Use the actual request, not just mention of money or fixing.',[
  ['Please send my money back; I do not want a replacement.', 'refund'],
  ['I would like the faulty switch fixed, even if it costs extra.', 'repair'],
  ['I do not want a refund. Can you mend this instead?', 'repair'],
  ['Repair was offered, but I would rather reverse the purchase.', 'refund'],
  ['Keep the payment; just restore it to working order.', 'repair'],
  ['There is no point fixing it. Return what I paid.', 'refund']]],
 ['paired-verdicts','boolean[]','text contains two numbered product reviews. Judge each overall recommendation independently by meaning and return two booleans in input order.',[
  ['1. A few flaws, but I recommend it. 2. Lovely packaging; terrible product, avoid it.',[true,false]],
  ['1. I sent it back and cannot recommend it. 2. I bought a second for my brother.',[false,true]],
  ['1. Not perfect, yet I would choose it again. 2. No regrets; worth buying.',[true,true]],
  ['1. Never again. 2. An expensive paperweight; returned.',[false,false]],
  ['1. This version disappoints; buy the old one. 2. No flashy extras, just reliable performance; recommended.',[false,true]],
  ['1. Awkward setup, but I love using it. 2. Five stars for the shop accepting my return; avoid the device.',[true,false]]]]
];
const rows=examples.map(([family,returns,criterion,examples])=>{
 const id='semantic-result-'+family;
 const files={'judge.nl':`---\nargs: {text: string}\nreturns: "${returns}"\n---\n${criterion} Delegate the judgment to another nl function, inspect its answer with eval, stage the answer in another eval, and then reply done.\n`};
 loadVirtualNatlang(files,'judge.nl');
 return {version:'natlang.improvement-case/1',id,family:'general-semantic-result',files,contract:{entry:'judge.nl',exportName:'default',programId:id},cases:examples.map(([text,expected],index)=>({id:id+':'+index,group:id+':'+index,split:index<2?'train':index<4?'validation':'test',args:[text],expected})),policy:{maxExperiments:3,maxPopulation:3,mode:'structural',strategy:'adaptive',objective:'model-calls',goal:criterion+' Simplify redundant delegation and completion while preserving the declared result. The current call can judge visible inputs and return its semantic answer directly.',allowedFiles:['judge.nl']},budget:{maxModelCalls:120,maxRollouts:30,maxProposals:3,maxElapsedMs:1200000},incidents:[{id:'observed-semantic-plumbing-'+family,cluster:'redundant-semantic-plumbing',split:'train',route:'structural-repair',reason:'Actual optimizer repeatedly inspected prepared context and used tools to return a semantic hypothesis. This reconstruction exercises the shared language affordance on ordinary tasks.',source:{file:source,hash}}],sourceGroups:['semantic-result-'+family],provenance:{kind:'reconstructed-general-language-overhead',startingInstructions:'Explicitly reconstructed overhead, not an original student failure.',oracle:'Fresh independently declared semantic labels; held-out examples unopened.',positiveSFT:false,disposition:'requires-target-reproduction-native-collection-and-independent-replay'}};
});
await mkdir(output,{recursive:true});await writeFile(join(output,'cases.jsonl'),rows.map(JSON.stringify).join('\n')+'\n');
await writeFile(join(output,'manifest.json'),JSON.stringify({cases:rows.length,source,hash,positiveSFT:false,disposition:'All source contracts load; collection/reproduction/replay still required.'},null,2)+'\n');
console.log(JSON.stringify({cases:rows.length,positiveSFT:false}));
