/** Partially working directory programs; independent labels, exact effects and sealed transfer families. */
import {mkdir,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {loadVirtualNatlang,compileVirtualProject} from '../../dist/runtime/virtual-project.js';
import * as runtime from '../../dist/improvement/target-runtime.js';
const header='---\nkind: directory-reducer\nargs: {}\nreturns: string\n---\n';
const rootType=`type Root={files(pattern:string):Promise<{relativePath:string;readText():Promise<string>}[]>;file(path:string):{readText():Promise<string>;writeText(text:string):Promise<void>}};\n`;
const preserve={'README.md':'Do not edit this unrelated file.\n','archive/old.txt':'Historical material outside the requested inputs.\n'};
export function directoryCampaignCases(){
 const rows=[];
 function add(id,cohort,goal,helper,examples){
  const files={'solve.nl':header+'Run the existing workflow with folder and return its result.\n','solve/workflow.ts':`import {nl} from '@natlang/node';\n${rootType}${helper}\n`};
  const reference={...files,'solve.nl':header+goal+' Read the relevant inputs, judge their meaning here, and use exact code for joins, sorting and file edits. A helper may be replaced or removed when it encodes the wrong behavior. Complete with the declared string.\n'};
  const cases=examples.map((example,i)=>{const folder={...preserve,...example.folder};return {id:id+'-'+i,group:'directory-campaign-'+id+'-'+i,split:i<3?'train':i===3?'validation':'test',args:[],folder,expected:'done',expectedFiles:{...folder,...example.effects}};});
  const row={version:'natlang.improvement-case/1',id:'directory-'+id,family:id,cohort,files,reference,contract:{entry:'solve.nl',exportName:'default',programId:'directory-'+id},cases,policy:{maxExperiments:3,maxPopulation:4,strategy:'adaptive',mode:'structural',objective:'model-calls',goal,allowedFiles:['solve.nl','solve/workflow.ts','types.ts','solve/judge.nl','solve/join.ts','solve/aggregate.ts']},budget:{maxModelCalls:150,maxRollouts:32,maxProposals:3,maxElapsedMs:3600000},sourceGroups:['directory-campaign:'+id],incidents:[{id:'reconstructed-directory-'+id,split:'train',mechanisms:['partial implementation','excess delegation','cross-file evidence'],source:'docs/NATIVE_IMPROVEMENT_FOLLOWUP.md',reason:'Harder reconstruction from observed native file failures; actual current-student reproduction required.'}],provenance:{kind:'failure-mechanism-reconstruction',positiveSFT:false,referenceHiddenFromOptimizer:true,oracle:'Independent semantic labels; exact compact JSON and unchanged unrelated files.'}};
  for(const source of [files,reference]){loadVirtualNatlang(source,'solve.nl');const checked=compileVirtualProject({files:source},runtime,{constrained:true,target:'node'});if(!checked.ok)throw Error(id+': '+JSON.stringify(checked.diagnostics));}
  rows.push(row);
 }
 const reports=(examples,key)=>examples.map(([notes,matching])=>({folder:notes,effects:{'report.json':JSON.stringify({[key]:matching.sort()})}}));
 add('final-refunds','development','Read all notes/**/*.txt, including notes directly under notes/. Determine the customer\'s final requested resolution from the whole note; superseded requests and declined refund offers do not count. Write report.json as JSON.stringify({refunds: sortedMatchingPaths}). Preserve every other file and return "done".',
 `export async function run(folder:Root):Promise<string>{const matching:string[]=[];for(const file of await folder.files('notes/*.txt')){const text=await file.readText();const yes=await nl<boolean>\`Does text contain any request for reversing payment, including earlier proposals?\`(text);if(yes)matching.push(file.relativePath);}await folder.file('report.json').writeText(JSON.stringify({refunds:matching.sort()}));return 'done';}`,
 reports([
  [{'notes/a.txt':'Please reverse my payment.','notes/b.txt':'Please repair the hinge.'},['notes/a.txt']],
  [{'notes/east/one.txt':'Earlier I wanted a refund. My final request is repair.','notes/west/two.txt':'A repair was suggested; my final choice is getting my payment back.','notes/plain.txt':'I decline the refund; mend it.'},['notes/west/two.txt']],
  [{'notes/a.txt':'I asked about a refund. I choose repair instead.','notes/sub/b.txt':'Do not refund me; restore the switch.'},[]],
  [{'notes/north/c.txt':'Please return what I paid, not a replacement.','notes/south/d.txt':'I have decided on reversing payment.','notes/e.txt':'The refund offer is rejected; fix the latch.'},['notes/north/c.txt','notes/south/d.txt']],
  [{'notes/new/z.txt':'I first requested repair, but my final decision is reimbursement.','notes/other/a.txt':'Refunds were discussed. I want the fault repaired.'},['notes/new/z.txt']],
  [{'notes/only.txt':'I considered reimbursement, then chose to keep it after repair.'},[]]
 ],'refunds'));
 const invoices=(sets)=>sets.map(records=>({folder:{'ledger.json':JSON.stringify(records.map(([id,cents])=>({id,cents}))),...Object.fromEntries(records.map(([id,_cents,text])=>['decisions/'+id+'.txt',text]))},effects:{'report.json':JSON.stringify({approved:records.filter(r=>r[3]).map(r=>r[0]).sort(),totalCents:records.filter(r=>r[3]).reduce((sum,r)=>sum+r[1],0)})}}));
 add('invoice-join','development','Read ledger.json rows {id,cents} and join each row to decisions/<id>.txt by ID. Judge whether the final decision actually approves that invoice; pending, superseded approval and cancellation do not count. Write report.json as JSON.stringify({approved: sortedApprovedIds,totalCents: exactSumOfTheirIntegerCents}). Preserve all other files and return "done".',
 `export async function run(folder:Root):Promise<string>{const rows=JSON.parse(await folder.file('ledger.json').readText()) as {id:string;cents:number}[];const notes=(await folder.files('decisions/*.txt')).sort((a,b)=>a.relativePath.localeCompare(b.relativePath));const approved:string[]=[];let totalCents=0;for(let i=0;i<rows.length;i++){const row=rows[i];const text=await notes[i].readText();const yes=await nl<boolean>\`Is the invoice in text finally approved?\`(text);if(yes){approved.push(row.id);totalCents+=row.cents;}}await folder.file('report.json').writeText(JSON.stringify({approved:approved.sort(),totalCents}));return 'done';}`,
 invoices([
  [['a',105,'Final decision: approved.',true],['b',320,'Rejected after review.',false]],
  [['z',170,'Approved, and the approval remains final.',true],['a',905,'Approval was proposed; final decision is rejection.',false],['m',250,'Finally approved after an earlier rejection.',true]],
  [['q',400,'Still pending. No approval yet.',false],['b',730,'Approval was withdrawn. Cancel this invoice.',false]],
  [['r',811,'Earlier rejected; final decision is approval.',true],['c',129,'The approval proposal was declined.',false],['n',267,'Approved for payment.',true]],
  [['x',999,'Final decision: rejected.',false],['d',101,'Approval is now confirmed.',true]],
  [['u',203,'Approved after final review.',true],['a',507,'The final approval remains in force.',true],['k',89,'Do not pay; rejected.',false]]
 ]));
 const edits=(specs)=>specs.map(([files,changes])=>{const effects={...files};for(const change of changes){const text=effects[change.path];if(text.split(change.find).length!==2)throw Error('Fixture edit must have one exact match');effects[change.path]=text.replace(change.find,change.replace_with);}return {folder:{...files,'change-request.json':JSON.stringify({changes})},effects};});
 add('exact-edits','development','Parse change-request.json {changes:[{path,find,replace_with}]}. Apply every change in order as one exact replacement, using the supplied strings without trimming or retyping them. Every match is unique at the point of application. Preserve all text outside each match, including whitespace and final newlines. Preserve request and unrelated files. Return "done".',
 `export async function run(folder:Root):Promise<string>{const request=JSON.parse(await folder.file('change-request.json').readText()) as {changes:{path:string;find:string;replace_with:string}[]};const change=request.changes[0];const text=await folder.file(change.path).readText();await folder.file(change.path).writeText(text.replace(change.find.trim(),change.replace_with.trim()));return 'done';}`,
 edits([
  [{'src/a.txt':'start old end\n'},[{path:'src/a.txt',find:'old',replace_with:'new'}]],
  [{'src/a.txt':'  old\nend\n','src/b.txt':'head\n  beta\n'},[{path:'src/a.txt',find:'  old\n',replace_with:'\tnew\n\n'},{path:'src/b.txt',find:'  beta\n',replace_with:'  gamma\n'}]],
  [{'src/z.txt':'x\n old \ny\n'},[{path:'src/z.txt',find:' old \n',replace_with:' next \n'}]],
  [{'lib/one.txt':'pre\n\tfirst\npost\n','lib/two.txt':' before \n'},[{path:'lib/one.txt',find:'\tfirst\n',replace_with:'  second\n'},{path:'lib/two.txt',find:' before \n',replace_with:' after \n\n'}]],
  [{'docs/note.txt':'lead\n  draft\ntrail\n'},[{path:'docs/note.txt',find:'  draft\n',replace_with:'\tfinal\n'}]],
  [{'a.txt':'one two\n','nested/b.txt':'x\n'},[{path:'a.txt',find:'one',replace_with:'three'},{path:'a.txt',find:'two',replace_with:'four'},{path:'nested/b.txt',find:'x\n',replace_with:'y\n\n'}]]
 ]));
 const reviews=(sets)=>sets.map(records=>({folder:{'reviews.json':JSON.stringify(records.map(([id,revision,text])=>({id,revision,text})))},effects:{'report.json':JSON.stringify({recommended:records.filter((r,i)=>!records.some((other,j)=>j!==i&&other[0]===r[0]&&other[1]>r[1])).filter(r=>r[3]).map(r=>r[0]).sort()})}}));
 add('latest-reviews','development','Read reviews.json rows {id,revision,text}. For each product ID use only the highest numbered revision. Judge whether that final review recommends the product overall, respecting mixed details, sarcasm and negation. Write report.json as JSON.stringify({recommended: sortedUniqueRecommendedIds}). Preserve all other files and return "done".',
 `export async function run(folder:Root):Promise<string>{const rows=JSON.parse(await folder.file('reviews.json').readText()) as {id:string;revision:number;text:string}[];const recommended:string[]=[];for(const row of rows){const text=row.text;const yes=await nl<boolean>\`Does text mention any positive feature of the product?\`(text);if(yes)recommended.push(row.id);}await folder.file('report.json').writeText(JSON.stringify({recommended:recommended.sort()}));return 'done';}`,
 reviews([
  [['a',1,'Dependable and useful. I recommend it.',true],['b',1,'Broken junk. Avoid it.',false]],
  [['x',1,'Wonderful device; worth buying.',true],['x',3,'Lovely box, but constant failures. Avoid this.',false],['b',2,'Difficult setup; still worth buying and I recommend it.',true]],
  [['a',2,'The display is beautiful. I cannot recommend this unreliable device.',false],['b',1,'The best thing was getting my refund. Avoid buying it.',false]],
  [['k',1,'Unusable. Returned.',false],['k',4,'The update fixed it. I now recommend buying it.',true],['q',2,'A lovely shell does not compensate for data loss. Avoid it.',false],['z',3,'I would gladly buy it again.',true]],
  [['v',5,'Pretty packaging. Awful device. Sent back.',false],['w',2,'Not perfect, but it is a keeper and I recommend it.',true],['w',1,'It initially failed. I cannot recommend it.',false]],
  [['a',3,'I love the look but regret buying it. Avoid it.',false],['b',7,'Excellent for my work. Worth buying.',true],['b',6,'Not reliable; do not buy it.',false]]
 ]));
 const applications=(sets)=>sets.map(records=>({folder:{'applications.json':JSON.stringify(records.map(([id,amount,text])=>({id,amount,note:'appeals/'+id+'.txt'}))),...Object.fromEntries(records.map(([id,_amount,text])=>['appeals/'+id+'.txt',text]))},effects:{'report.json':JSON.stringify({eligible:records.filter(r=>r[3]).map(r=>r[0]).sort(),total:records.filter(r=>r[3]).reduce((sum,r)=>sum+r[1],0)})}}));
 add('eligibility-appeals','transfer','Read applications.json {id,amount,note} and each specified note. Eligibility requires completed safety training, unless the note records an explicitly authorized training waiver. Discussion or denial of a waiver is not authorization. Use the final status, excluding withdrawn training completion or withdrawn waivers. Write report.json as JSON.stringify({eligible: sortedEligibleIds,total: exactSumOfEligibleAmounts}). Preserve all other files and return "done".',
 `export async function run(folder:Root):Promise<string>{const rows=JSON.parse(await folder.file('applications.json').readText()) as {id:string;amount:number;note:string}[];const eligible:string[]=[];let total=0;for(const row of rows){const text=await folder.file(row.note).readText();const yes=await nl<boolean>\`Does text confirm completed safety training? Do not accept waivers.\`(text);if(yes){eligible.push(row.id);total+=row.amount;}}await folder.file('report.json').writeText(JSON.stringify({eligible:eligible.sort(),total}));return 'done';}`,
 applications([
  [['a',100,'Safety training is complete.',true],['b',50,'Training remains incomplete. No waiver.',false]],
  [['z',317,'Training is incomplete; an authorized waiver is now granted.',true],['a',601,'A waiver was discussed and denied. Training is incomplete.',false]],
  [['n',211,'Training was certified but the certification has been withdrawn. No waiver.',false],['m',92,'The training waiver is explicitly authorized and remains active.',true]],
  [['c',119,'The earlier waiver was withdrawn; safety training is now completed.',true],['r',233,'Training is incomplete. The manager has approved a training waiver.',true],['d',711,'A waiver request is pending. No completed training.',false]],
  [['p',187,'Safety training is complete and valid.',true],['q',541,'The waiver is denied; training has not been completed.',false]],
  [['t',421,'Training is incomplete but the authorized waiver remains in force.',true],['s',91,'The authorized waiver was revoked. Training remains incomplete.',false]]
 ]));
 const meetings=(sets)=>sets.map(records=>({folder:{...Object.fromEntries(records.map(([path,text])=>[path,text]))},effects:{'report.json':JSON.stringify({meetings:records.filter(r=>r[2]).map(r=>({path:r[0],...r[2]})).sort((a,b)=>a.path.localeCompare(b.path))})}}));
 add('confirmed-meetings','transfer','Read meetings/**/*.txt, including direct children. Extract each currently confirmed meeting\'s final owner and place as {path,owner,place}. Ignore cancelled or proposed meetings and superseded owner/place assignments. Write report.json as JSON.stringify({meetings: recordsSortedByPath}). Preserve all other files and return "done".',
 `export async function run(folder:Root):Promise<string>{const meetings:{path:string;owner:string;place:string}[]=[];for(const file of await folder.files('meetings/*.txt')){const text=await file.readText();const value=await nl<{owner:string;place:string}|null>\`Extract the first named owner and place for a confirmed meeting in text, or null if it is not confirmed.\`(text);if(value)meetings.push({path:file.relativePath,...value});}await folder.file('report.json').writeText(JSON.stringify({meetings:meetings.sort((a,b)=>a.path.localeCompare(b.path))}));return 'done';}`,
 meetings([
  [['meetings/a.txt','Confirmed: Alice at Library.',{owner:'Alice',place:'Library'}],['meetings/b.txt','Proposed only: Bruno at Hall.',null]],
  [['meetings/east/x.txt','Draft: Clara at School. Final confirmed: Diego at Studio.',{owner:'Diego',place:'Studio'}],['meetings/y.txt','Confirmed: Evan at Park. Cancelled; no meeting now.',null]],
  [['meetings/a.txt','Confirmed correction: Alice at Office instead of Bruno at Park.',{owner:'Alice',place:'Office'}],['meetings/b.txt','Proposed Faye at Hall; still unconfirmed.',null]],
  [['meetings/north/a.txt','Earlier: Gail at School. Final confirmed: Hugo at Library.',{owner:'Hugo',place:'Library'}],['meetings/south/b.txt','Confirmed: Iris at Studio.',{owner:'Iris',place:'Studio'}]],
  [['meetings/deep/a.txt','Jules at Office was suggested only. Confirmed: Kira at Hall.',{owner:'Kira',place:'Hall'}]],
  [['meetings/a.txt','Confirmed: Liam at Park. This meeting is now cancelled.',null],['meetings/new/b.txt','Final confirmed: Mira at Studio, replacing Noah at School.',{owner:'Mira',place:'Studio'}]]
 ]));
 return rows;
}
if(process.argv[1]===new URL(import.meta.url).pathname){const output=process.argv[2];if(!output)throw Error('usage: build-directory-campaign.mjs OUTPUT');const rows=directoryCampaignCases();await mkdir(output,{recursive:true});for(const row of rows){await mkdir(join(output,row.id),{recursive:true});await writeFile(join(output,row.id,'case.json'),JSON.stringify(row,null,2));}await writeFile(join(output,'cases.jsonl'),rows.map(JSON.stringify).join('\n')+'\n');await writeFile(join(output,'manifest.json'),JSON.stringify({cases:rows.length,development:rows.filter(r=>r.cohort==='development').map(r=>r.id),transfer:rows.filter(r=>r.cohort==='transfer').map(r=>r.id),positiveSFT:false,confirmationExperiments:1,trainingModel:'Sharp-MiniCPM5-2B',design:'One integrated campaign. Four development families; two fresh transfer families opened only after optimizer source freezes. Exact file oracles. Report every attempt; no cohort replacement after failure.'},null,2));console.log(JSON.stringify({cases:rows.length}));}
