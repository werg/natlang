/** A small, diverse native benchmark. Independent gold; reconstruction is never called an observed student failure. */
import {mkdir,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {loadVirtualNatlang} from '../../dist/runtime/virtual-project.js';
export function followupCases(){
 const rows=[];
 const add=(id,family,args,returns,types,bad,goal,examples,extra={})=>{
  const header=`---\n${extra.reducer?'kind: directory-reducer\n':''}args: ${args}\nreturns: ${returns}\n---\n`;
  const files={'solve.nl':header+bad+'\n',...(types?{'types.ts':types}:{} )};
  const reference={...files,'solve.nl':header+goal+' Judge the visible inputs here by meaning. Use exact code for aggregation and file bookkeeping. Return the declared semantic value directly when no computation or effect is needed. Do not delegate a small visible task or use keyword shortcuts.\n'};
  loadVirtualNatlang(files,'solve.nl');loadVirtualNatlang(reference,'solve.nl');
  rows.push({version:'natlang.improvement-case/1',id:'followup-'+id,family,files,reference,contract:{entry:'solve.nl',exportName:'default',programId:'followup-'+id},cases:examples.map((e,i)=>({id:id+'-'+i,group:id+'-'+i,split:i===0?'train':i===1?'validation':'test',args:e.args,expected:e.expected,...(e.services?{services:e.services}:{}),...(e.folder?{folder:e.folder}:{}),...(e.expectedFiles?{expectedFiles:e.expectedFiles}:{})})),policy:{maxExperiments:2,maxPopulation:3,strategy:'adaptive',mode:'structural',objective:'model-calls',goal,allowedFiles:['solve.nl','types.ts','helpers.ts']},budget:{maxModelCalls:120,maxRollouts:24,maxProposals:2,maxElapsedMs:2400000},incidents:[{id:'followup-'+id,cluster:id,split:'train',route:'structural-repair',reason:family==='classification'&&id==='recommendation'?'Observed per-item recommendation delegation overhead; fresh examples.':'Reconstructed candidate failure mechanism; requires actual current-student reproduction.',source:{file:'docs/LUNA_BONSAI_IMPROVEMENT_RESULTS.md'}}],sourceGroups:['followup-'+id],provenance:{kind:'failure-mechanism-reconstruction',positiveSFT:false,oracle:'Independently declared labels and exact output effects',referenceHiddenFromOptimizer:true,reproductionRequired:true}});
 };
 const texts=(items)=>items.map(([text,expected])=>({args:[text],expected}));
 add('recommendation','classification','{reviews: "string[]"}','number','', 'Judge each review using a separate nl call, then count the recommending reviews in code.', 'Count reviews that recommend the product overall, considering contrast, sarcasm and negation. Count the semantic verdicts exactly.',[
  {args:[['I expected to return it; now I recommend it to everyone.','Lovely box, unusable product. Avoid it.']],expected:1},
  {args:[['No regrets; worth buying.','A few flaws cannot outweigh its usefulness. Recommended.']],expected:2},
  {args:[['I recommend the old model. This one disappoints.','What a triumph: another broken machine. Returned.']],expected:0}]);
 add('priority','classification','{text: string}', '"\'urgent\' | \'routine\'"','', 'If the word urgent appears anywhere, return urgent; otherwise return routine.', 'Classify whether the writer requests immediate attention or is describing a routine matter. Mentions, quotations and explicit denials of urgency are not immediate requests.',texts([
  ['This is not urgent; please handle it next month.','routine'],['Ignore the old urgent flag. There is no rush now.','routine'],['Production has stopped. Please intervene immediately.','urgent']]));
 add('resolution','classification','{text: string}', '"\'refund\' | \'repair\'"','', 'If refund is mentioned, return refund. Otherwise return repair.', 'Return the resolution actually requested: refund for reversing payment, repair for restoring the item. Respect negation and the final request.',texts([
  ['I do not want a refund. Please fix the faulty switch.','repair'],['A refund was suggested, but I would prefer you mend it.','repair'],['Please return my payment; I do not want this fixed.','refund']]));
 add('assignment','extraction','{text: string}','Assignment','export type Assignment={owner:string;place:string};\n', 'Return the first named owner and first place in text, even if a later correction appears.', 'Extract the final confirmed owner and meeting place. Discard superseded proposals. Return {owner,place}.',texts([
  ['Initially Alice at Park; confirmed correction: Bruno at Library.',{owner:'Bruno',place:'Library'}],['Draft: Clara at Hall. Final: Diego at Studio.',{owner:'Diego',place:'Studio'}],['Evan at School was cancelled. Faye at Office is confirmed.',{owner:'Faye',place:'Office'}]]));
 add('quantity','extraction','{text: string}','Order','export type Order={item:string;quantity:number};\n', 'Extract the first quantity and first item mentioned.', 'Extract the final committed order as {item,quantity}; ignore rejected estimates and cancelled orders.',texts([
  ['We considered five chairs. The final order is two chairs.',{item:'chairs',quantity:2}],['A plan for nine lamps was rejected. Order three lamps.',{item:'lamps',quantity:3}],['Six desks was an estimate. We committed to four desks.',{item:'desks',quantity:4}]]));
 add('confirmed-events','extraction','{text: string}','"string[]"','', 'Return every mentioned event name, including cancelled events.', 'Return only currently confirmed event names in alphabetical order, excluding cancelled or merely suggested events.',texts([
  ['Confirmed: Dinner. Cancelled: Concert. Suggested but unconfirmed: Workshop.',['Dinner']],['Lunch is confirmed. Meeting was cancelled. Picnic is only a suggestion.',['Lunch']],['Cancelled: Lecture. Confirmed: Brunch and Seminar.',['Brunch','Seminar']]]));
 const paged=(pages)=>`export function pages():number{return ${pages.length};} export function page(index:number):{id:string;text:string;amount:number}[]{const data=${JSON.stringify(pages)};return data[index]??[];}`;
 for(const [id,goal,returns,sets]of [
  ['approved-sum','Read all store pages (zero based; store.pages() gives their finite count). Sum amounts only for claims that are finally approved, not rejected or pending. Classify each by meaning and sum exactly.','number',[[[false,7],[true,4]],[[false,9],[true,6]],[[true,2],[true,3]]]],
  ['urgent-ids','Read every store page using its finite page count. Return IDs requiring immediate action in input order, respecting negation and quotations.','"string[]"',[[[false,1],[true,2]],[[false,3],[true,4]],[[true,5],[false,6]]]],
  ['recommendation-flags','Read every store page using its finite page count. Return one recommendation boolean per row in input order, interpreting the whole review by meaning.','"boolean[]"',[[[false,1],[true,2]],[[false,3],[true,4]],[[true,5],[false,6]]]]
 ]){
  const examples=sets.map((set,i)=>{const data=set.map(([yes,amount],j)=>[{id:'r'+j,text:id==='approved-sum'?(yes?'Approved after final review.':'Approval was discussed, but the final decision is rejection.'):id==='urgent-ids'?(yes?'Please intervene immediately; the service is down.':'This is not urgent; take your time.'):(yes?'I doubted it, but now I recommend it.':'I recommend avoiding this unreliable model.'),amount}]);return {args:[],services:{store:paged(data)},expected:id==='approved-sum'?set.filter(([yes])=>yes).reduce((sum,[,amount])=>sum+amount,0):id==='urgent-ids'?set.flatMap(([yes],j)=>yes?['r'+j]:[]):set.map(([yes])=>yes)};});
  add(id,'pagination','{}',returns,'','Read store.page(0) only; do not request later pages. '+goal.replace('Read all store pages (zero based; store.pages() gives their finite count). ','').replace('Read every store page using its finite page count. ',''),goal,examples);
 }
 for(const [id,criterion,key,sets]of [
  ['refund-files','the customer actually requests reversing payment, not a repair','refunds',[['I do not want a refund; repair it.','Return my money; no repair please.'],['Please mend it rather than refund me.','Reverse the payment.'],['A refund was offered, but fix it instead.','I want my payment returned.']]],
  ['urgent-files','the writer requests immediate action, respecting denials of urgency','urgent',[['The urgent flag is obsolete; no rush.','Production is down; intervene now.'],['Not urgent; next month is fine.','Please help immediately.'],['This can wait.','We need action right now.']]],
  ['keep-files','the reviewer recommends the product overall, despite mixed details','keep',[['Lovely box, terrible product. Avoid it.','Bad setup, but now I recommend it.'],['I recommend the old one. This replacement disappoints.','I would happily buy this again.'],['Another expensive paperweight. Returned.','Not perfect; still worth buying.']]]
 ]){
  const goal=`Read all notes/*.txt and classify whether ${criterion}. Write report.json with exactly {"${key}":[sorted matching paths]}; preserve original files. Serialize that object with JSON.stringify, without extra whitespace. Return "done".`;
  const examples=sets.map(([a,b],index)=>{
   const vary=id!=='refund-files';
   const folder=index===1&&vary?{'notes/a.txt':b,'notes/b.txt':a}:{'notes/a.txt':a,'notes/b.txt':b};
   const matching=index===1&&vary?['notes/a.txt']:['notes/b.txt'];
   if(index===2&&vary){folder['notes/c.txt']=id==='urgent-files'?'Production has stopped. Please intervene immediately.':'Despite a fiddly setup, I recommend this and would buy it again.';matching.push('notes/c.txt');}
   return {args:[],folder,expected:'done',expectedFiles:{...folder,'report.json':JSON.stringify({[key]:matching})}};
  });
  add(id,'directory-reducer','{}','string','', 'Return "done" after inspecting the first note. Do not write a report.',goal,examples,{reducer:true});
 }
 return rows;
}
if(process.argv[1]===new URL(import.meta.url).pathname){const output=process.argv[2];if(!output)throw Error('usage: build-followup-cases.mjs OUTPUT');const rows=followupCases();await mkdir(output,{recursive:true});await writeFile(join(output,'cases.jsonl'),rows.map(JSON.stringify).join('\n')+'\n');await writeFile(join(output,'manifest.json'),JSON.stringify({cases:rows.length,families:[...new Set(rows.map(row=>row.family))],positiveSFT:false,reconstruction:'Independent labels and hidden reference programs; actual baseline reproduction recorded by the study.'},null,2)+'\n');console.log(JSON.stringify({cases:rows.length}));}
