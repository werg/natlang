#!/usr/bin/env node
/** Reference audit; only exact regenerated project fixtures may execute trusted source. */
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {createTranslationEpisodes} from './build-translation-episodes.mjs';
import {scoreContractNliObjective} from '../../dist/skills/contractnli-objective.js';
import {scoreScifactObjective} from '../../dist/skills/scifact-objective.js';
import {scoreResearchObjective} from '../../dist/skills/research-objective.js';
import {verifyCrosswordReference,scoreCrosswordObjective} from '../../dist/skills/crossword-objective.js';
import {solveFiniteCsp,cspProgressBound,scoreCspProgress} from '../../dist/skills/csp-objective.js';
const [input,out]=process.argv.slice(2);if(!input||!out)throw Error('usage: audit-semantic-references.mjs INPUT OUTPUT');
const bytes=await readFile(input),episodes=bytes.toString().split('\n').filter(Boolean).map(JSON.parse),errors=[],python=[];
const trusted=new Map(createTranslationEpisodes().map(e=>[e.id,JSON.stringify(e)]));
let cases=0,checks=0;
for(const e of episodes)for(const role of ['support','query','transfer'])for(const row of e[role]?.cases??[]){
 cases++;
 try{
  const schema=(role==='transfer'?e.provenance?.transfer_metric:e.provenance?.metric)?.schema;
  if(schema==='natlang.skill-contractnli/1'){
   const annotations=Object.fromEntries(Object.entries(row.expected.hypotheses).map(([id,gold])=>[id,{choice:gold.choice,span_ids:gold.evidence_alternatives[0]??[]}]));
   if(scoreContractNliObjective(row.args[0],JSON.stringify({annotations}),row.expected).quality!==1)throw Error('ContractNLI reference mismatch');checks+=Object.keys(annotations).length;
  }else if(schema==='natlang.skill-scifact/1'){
   const expected=row.expected,alternative=expected.accepted_evidence_sets[0]??[];
   const answer={label:expected.label,citations:alternative};
   if(scoreScifactObjective(row.args[0],JSON.stringify(answer),expected).quality!==1)throw Error('SciFact reference mismatch');checks++;
  }else if(schema==='natlang.skill-research/1'){
   const answer={label:row.expected.label,unresolved:row.expected.unresolved,citations:row.expected.requiredEvidence.map(r=>({sourceId:r.sourceId,evidence:r.text}))};
   if(scoreResearchObjective(row.args[0],answer,row.expected).quality!==1)throw Error('research evidence/reference disagreement');
   if(new Set(row.expected.requiredEvidence.map(r=>r.sourceId)).size<2)throw Error('research case does not require multiple sources');
   checks+=answer.citations.length;
  }else if(schema==='natlang.crossword-csp/1'){
   const reference=verifyCrosswordReference(row.args[0]);
   if(JSON.stringify(reference)!==JSON.stringify(row.expected))throw Error('crossword reference mismatch');
   for(const fills of reference.accepted){if(scoreCrosswordObjective(row.args[0],{fills},row.expected).quality!==1)throw Error('valid fill failed');checks++;}
  }else if(schema==='natlang.skill-csp/1'){
   const solution=solveFiniteCsp(row.args[0]);
   if(solution.status!=='unique'||JSON.stringify(cspProgressBound(row.args[0]))!==JSON.stringify(row.expected))throw Error('CSP reference mismatch');
   if(scoreCspProgress(row.args[0],JSON.stringify(solution.solutions[0]),row.expected).quality!==1)throw Error('CSP solution failed');checks++;
  }else if(schema==='natlang.skill-translation/1'){
   if(trusted.get(e.id)!==JSON.stringify(e))throw Error('translation is not an exact trusted original fixture');
   const visible=JSON.parse(row.args[0]);
   if(visible.source_language==='javascript'){
    const solve=Function(visible.source_program+';return solve;')();
    for(const t of row.expected.tests){if(solve(t.input.x,t.input.y)!==t.expected)throw Error('JS reference mismatch');checks++;}
   }else python.push({source:visible.source_program,tests:row.expected.tests});
  }else throw Error('unsupported reference audit schema '+schema);
 }catch(error){errors.push({episode:e.id,case:row.id,error:String(error)});}
}
if(python.length){
 const script='import json,sys\nn=0\nfor row in json.load(sys.stdin):\n ns={}\n exec(row["source"],ns)\n for t in row["tests"]:\n  assert ns["solve"](**t["input"])==t["expected"]\n  n+=1\nprint(n)\n';
 const result=spawnSync('python3',['-I','-c',script],{input:JSON.stringify(python),encoding:'utf8'});
 if(result.status!==0)errors.push({error:'trusted Python references failed',detail:result.stderr});else checks+=Number(result.stdout.trim());
}
const report={schema:'natlang.semantic-reference-audit/1',input_sha256:createHash('sha256').update(bytes).digest('hex'),episodes:episodes.length,cases,reference_checks:checks,provider_calls:0,passed:!errors.length,errors};
await writeFile(out,JSON.stringify(report,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(report));if(errors.length)process.exitCode=1;
