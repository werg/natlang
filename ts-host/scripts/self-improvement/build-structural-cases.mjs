/** Turn measured development inefficiencies into executable cases, including failed optimizer edits. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import ts from 'typescript';
import {isDeepStrictEqual} from 'node:util';
import {Folder,OperationJournal} from '../../dist/index.js';
import {fingerprint} from '../../dist/adaptation/identity.js';
const [study,output]=process.argv.slice(2);
if(!output)throw Error('usage: build-structural-cases.mjs STUDY OUTPUT');
const protocol=JSON.parse(await readFile(join(study,'protocol.json'),'utf8'));
if(protocol.contract.entry!=='recommending_reviews.nl')throw Error('This curriculum builder requires the measured semantic-review fixture.');
const native=JSON.parse(await readFile(join(study,'native.json'),'utf8'));
const journal=new OperationJournal(join(study,'native-journal'));
const observations=(await readFile(join(study,'exchanges.ndjson'),'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
const training=protocol.cases.filter(row=>row.split==='train');
let files={...protocol.files};const snapshots=[{files:{...files},mechanism:'original-delegation'}];
// Parse literal writes. Never execute saved model code while reconstructing training tasks.
for(const exchange of observations.filter(row=>row.command==='native'&&row.role==='optimizer')){
 for(const [tool,args]of exchange.turn.calls??[]){
  if(tool!=='eval'||typeof args.code!=='string')continue;
  let changed=false;
  const visit=node=>{
   if(ts.isCallExpression(node)&&ts.isPropertyAccessExpression(node.expression)&&node.expression.name.text==='writeText'){
    const fileCall=node.expression.expression;
    if(ts.isCallExpression(fileCall)&&ts.isPropertyAccessExpression(fileCall.expression)&&fileCall.expression.name.text==='file'){
     const [path]=fileCall.arguments,[text]=node.arguments;
     if(path&&text&&ts.isStringLiteralLike(path)&&ts.isStringLiteralLike(text)&&protocol.policy.allowedFiles.includes(path.text)){
      files[path.text]=text.text;changed=true;
     }
    }
   }
   ts.forEachChild(node,visit);
  };
  visit(ts.createSourceFile('saved.ts',args.code,ts.ScriptTarget.Latest,true));
  if(changed)snapshots.push({files:{...files},mechanism:'observed-editor-trial'});
 }
}
const rows=[],seen=new Set();
for(const snapshot of snapshots){
 const source=Folder.fromFiles(snapshot.files).snapshot().digest;if(seen.has(source))continue;seen.add(source);
 const reference=fingerprint({source,suite:native.baseline.suiteVersion,split:'train',ids:training.map(row=>row.id),seed:protocol.policy.seed??0});
 const measured=training.map(row=>({row,result:journal.read(reference+':'+row.id)?.value}));
 // Validation and held-out observations cannot seed these cases.
 if(!measured.length||measured.some(({row,result})=>!result||result.error||!isDeepStrictEqual(result.value,row.expected)||!(result.modelCalls>1)))continue;
 const id='observed-structural-'+source.slice(0,16);
 const reason='Correct training executions still required '+measured.reduce((sum,item)=>sum+item.result.modelCalls,0)+' model requests. Refine the actual execution structure and completion protocol.';
 rows.push({version:'natlang.improvement-case/1',id,family:'semantic-batching-'+snapshot.mechanism,mechanism:snapshot.mechanism,files:snapshot.files,contract:{...protocol.contract,programId:id},cases:[...protocol.cases.filter(row=>row.split!=='test'),...[
  [['It sometimes disconnects, but I would buy this model again.','Beautiful shell. I regret buying it and will not recommend it.','After six reliable months I bought one for my sister.'],2],
  [['Wonderful, another charger that died in a week. Spare yourself the purchase.','It looks cheap, but it has never let me down. I would choose it again.','The only good thing was getting my money back; stay away.'],1]
 ].map(([texts,expected],index)=>({id:'new-structural-test-'+index,group:'new-structural-test-'+index,split:'test',args:[texts.map((text,i)=>({id:'review-'+i,text}))],expected}))],policy:{...protocol.policy,goal:protocol.policy.goal+' Pursue further savings after an accepted improvement; finish only when no further evidenced change is supported or the declared experiments end.'},budget:protocol.budget,incidents:[{id,cluster:id,split:'train',route:'structural-repair',reason,source:{file:join(study,'native.json'),hash:createHash('sha256').update(await readFile(join(study,'native.json'))).digest('hex')}}],sourceGroups:training.map(row=>row.group),provenance:{kind:'measured-native-structural-inefficiency',source,oracle:'Original independently declared semantic labels; exact aggregation',reconstruction:'Literal source writes parsed without executing model code, and linked to completed train execution identities.',positiveSFT:false,disposition:'requires-current-native-collection-and-independent-replay'}});
}
await mkdir(output,{recursive:true});
await writeFile(join(output,'cases.jsonl'),rows.map(JSON.stringify).join('\n')+'\n');
await writeFile(join(output,'manifest.json'),JSON.stringify({cases:rows.length,mechanisms:rows.map(row=>row.mechanism),positiveSFT:false,heldOutObservationsUsed:false},null,2)+'\n');
console.log(JSON.stringify({cases:rows.length,positiveSFT:false}));
