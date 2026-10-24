#!/usr/bin/env node
/** Static, source-bound terminal judgments plus their complete parent programs.
 * This constructs RL tasks with exact authored oracles, never model generations.
 * Policy: practice individual semantic decisions and retain their full programs.
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { pathToFileURL, fileURLToPath } from 'node:url';
const repo=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
const sha=value=>createHash('sha256').update(value).digest('hex');
const runtime=resolve(repo,'ts-host');
const require=createRequire(resolve(runtime,'package.json'));
const ts=require('typescript');
const {definitionProject}=await import(pathToFileURL(resolve(runtime,'dist/teacher/program.js')));
const {verifyCases}=await import(pathToFileURL(resolve(runtime,'dist/teacher/curriculum.js')));
const {TOOLS_PROMPT}=await import(pathToFileURL(resolve(runtime,'dist/native/prompt.js')));
const {generationHoldReason,quarantineReason}=await import(pathToFileURL(resolve(runtime,'dist/teacher/curriculum-policy.js')));

export function terminalJudgments(parent) {
 if(parent.split!=='train')throw Error('semantic RL preparation accepts train parents only');
 if(parent.family!=='curriculum_workbench_tickets'||parent.id.endsWith(':hinted'))return [];
 const service=parent.semantics.services?.helpdesk;
 if(typeof service!=='string')throw Error('ticket parent lacks original service data');
 const file=ts.createSourceFile('helpdesk.ts',service,ts.ScriptTarget.Latest,true);
 const declarations=file.statements.filter(ts.isVariableStatement).flatMap(s=>[...s.declarationList.declarations]);
 const declaration=declarations.find(d=>ts.isIdentifier(d.name)&&d.name.text==='ITEMS');
 if(!declaration?.initializer||!ts.isArrayLiteralExpression(declaration.initializer))throw Error('ticket service needs its exact literal ITEMS array');
 const items=JSON.parse(declaration.initializer.getText(file));
 const labels=new Map((parent.curriculum.evidence.world??[]).map(line=>{
  const match=/^([^:]+): (\{.*\})$/.exec(line);if(!match)throw Error('unsupported ticket oracle evidence');return [match[1],JSON.parse(match[2])];
 }));
 if(new Set(items.map(item=>item.id)).size!==items.length||labels.size!==items.length)throw Error('ticket evidence membership differs');
 const specs=[
  ['churn','boolean','Decide whether ticket signals that the customer may cancel or leave. Read its meaning, including any negation or explicit commitment to stay. Return the Boolean judgment.'],
  ['blocking','boolean','Decide whether ticket describes work that is blocked right now. Distinguish current inability to work from a workaround or an issue that can wait. Return the Boolean judgment.'],
  ['kind','"bug" | "billing" | "how_to" | "feature_request"','Classify ticket: "bug" when something is broken, "billing" for charges and invoices, "how_to" for questions about using the product, "feature_request" for something the product does not do yet. Return the declared label.'],
 ];
 return items.flatMap(ticket=>specs.map(([field,returns,instructions])=>{
  const expected=labels.get(ticket.id)?.[field];
  if(field==='kind'?!['bug','billing','how_to','feature_request'].includes(expected):typeof expected!=='boolean')throw Error('missing or invalid exact ticket label');
  const id=parent.id+':terminal:'+ticket.id+':'+field;
  const project=definitionProject('judge_ticket',{args:{ticket:'unknown'},returns,instructions});
  return {version:'natlang.program/2',id,kind:'lambda_source',family:'rl-semantic-ticket-'+field,
   source:'natlang-inline-curriculum-terminal-judgments',split:'train',source_ids:[...new Set([...(parent.source_ids??[]),parent.id])],source_groups:[...parent.source_groups],
   license:parent.license??'project-generated',gold_sources:['original-procedural-ticket-oracle'],
   semantics:{root:project.root,files:project.files,inputs:{ticket},expected,operation:'exact'},
   curriculum:{version:'natlang.inline_curriculum/1',family:'rl-semantic-ticket-'+field,family_version:1,shape:parent.curriculum.shape,
    variant:parent.curriculum.variant,pair_group:null,split_group:parent.curriculum.split_group,slice:'inline_placement',domain:'other',mode:'single_call',inline:'optional',
    evidence:{world:[JSON.stringify(ticket)],retrieved:[],background:[]},assumptions:[],decisive:[],plausible_actions:[],minimum_sequence:[],
    reference:{root:[['eval',{code:'return '+JSON.stringify(expected)+';',finish:true}]]},
    rl:{level:1,family:'semantic-ticket-'+field,reason:'One supplied item and one semantic judgment; full parent programs are retained separately.'}},
   generation:{method:'static decomposition of an exact procedurally generated train task; no sampled trajectory',
    parent_id:parent.id,parent_record_sha256:sha(JSON.stringify(parent)),ticket_id:ticket.id,oracle_field:field,
    teacher_generation:false,reward:'Actual exact task outcome; Neuralese use is optional and an actual captured graph receives whole-graph updates.'}};
 }));
}

async function main(){
 const args=process.argv.slice(2),option=name=>args[args.indexOf(name)+1];
 if(!args.includes('--parents')||!args.includes('--out'))throw Error('--parents FILE --out DIRECTORY are required');
 const input=resolve(option('--parents')),out=resolve(option('--out')),raw=await readFile(input,'utf8');
 const parents=raw.trim().split('\n').map(JSON.parse);if(parents.some(row=>row.split!=='train'))throw Error('all parents must be train');
 const judgments=parents.flatMap(terminalJudgments);
 const retained=parents.map(row=>({...row,curriculum:{...row.curriculum,rl:{level:row.family==='curriculum_workbench_tickets'?3:2,family:row.family,reason:'Complete original program with its actual execution and outcome.'}}}));
 const rows=[...judgments,...retained];if(new Set(rows.map(row=>row.id)).size!==rows.length)throw Error('duplicate task identity');
 for(const row of rows){const reason=generationHoldReason(row)??quarantineReason(row);if(reason)throw Error(row.id+': '+reason);}
 const verification=await verifyCases(rows,TOOLS_PROMPT),failures=verification.filter(item=>!item.ok);
 await mkdir(out,{recursive:true});
 const report={schema:'natlang.rl-semantic-task-preparation/1',parents:{path:input,sha256:sha(raw),rows:parents.length},
  terminal_judgments:judgments.length,complete_parent_programs:retained.length,rows:rows.length,verified:rows.length-failures.length,failures,
  source_groups_preserved:true,teacher_generation:false,training_admitted:false,policy:'Practice the assigned semantic decision in one call, retaining complete parent programs and exact authored oracles; no requirement to use Neuralese.'};
 await writeFile(resolve(out,'verification.json'),JSON.stringify(report,null,2)+'\n');
 if(failures.length)throw Error('semantic RL tasks failed actual reference execution: '+JSON.stringify(failures.slice(0,3)));
 await writeFile(resolve(out,'cases.jsonl'),rows.map(row=>JSON.stringify(row)+'\n').join(''));
 console.log(JSON.stringify(report));
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
