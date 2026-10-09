import {better,objectiveMeasure} from 'natlang:gepa';
import type {Disposition,ExperimentResult,ExperimentFeedback,Hypothesis,ImprovementPolicy,SearchState} from '../types';
import type {ExperimentFolder,ExperimentEvaluator,Decisions} from './capabilities';
import {brief,request} from './context';
import {inspect} from './feedback';
import {transformationText} from './transformations';

/** A hypothesis names files among the allowed ones, and a statement unless it proposes no edit. */
export function hypothesisProblem(hypothesis:Hypothesis,policy:ImprovementPolicy):string {
  const outside=hypothesis.files.filter(path=>!policy.allowedFiles.includes(path));
  if(outside.length)return 'A hypothesis names files among the allowed ones ('+policy.allowedFiles.join(', ')+'); these are not: '+outside.join(', ')+'.';
  if(!hypothesis.statement.trim())return 'A hypothesis has a statement naming the observed failure or wasted action it addresses.';
  if(hypothesis.kind==='structure'&&policy.mode!=='structural')return 'The kind `structure` belongs to structural mode; this run is in '+policy.mode+' mode.';
  return '';
}

/** One finite experiment. Diagnosis, hypothesis and the source edit are semantic decisions; everything else is exact. */
export async function test(folder:ExperimentFolder,evaluator:ExperimentEvaluator,parentId:string,policy:ImprovementPolicy,state:SearchState,decide:Decisions):Promise<ExperimentResult> {
  const seed=policy.seed??0;
  const parent=folder.at(parentId).branch();
  const measured=await evaluator.evaluate(folder.at(parentId),{split:'validation',seed});
  const feedback=await inspect(folder,evaluator,parentId,policy,decide);
  const ended=(disposition:Disposition,reason:string,candidate?:string,failing?:ExperimentResult['failing'],detail?:ExperimentFeedback):ExperimentResult=>({source:parentId,parent:parentId,accepted:false,disposition,reason,quality:measured.quality,sourceBytes:measured.sourceBytes,...(measured.modelCalls!==undefined?{modelCalls:measured.modelCalls}:{}),scores:measured.scores??[],...(candidate?{candidate}:{}),...(failing?{failing}:{}),...(detail?{feedback:detail}:{})});
  if(feedback.opportunity.kind==='fixture')return ended('fixture-error',feedback.evidence.filter(row=>row.failureKind==='fixture').map(row=>row.error??'Independent fixture failed.').join('\n'));
  if(feedback.opportunity.kind==='none')return ended('no-opportunity',feedback.opportunity.reason);
  const paths=parent.filePaths();
  const sourceFiles=await Promise.all(paths.map(async path=>({path,text:await parent.readText(path)})));
  const history=state.history;
  const card=brief(policy,feedback.evidence,sourceFiles,state.lastExperiment);
  const diagnosis=await decide.diagnose({evidence:feedback.evidence,brief:card,goal:policy.goal,objective:policy.objective??'quality',opportunity:feedback.opportunity,history,...(state.lastExperiment?{lastExperiment:state.lastExperiment}:{})});
  const transformation=policy.transformation?transformationText(policy.transformation):undefined;
  const hypothesis=await decide.hypothesize({diagnosis,sourceFiles,mode:policy.mode,allowedFiles:policy.allowedFiles,history,...(transformation?{transformation}:{})});
  if(hypothesis.kind==='none')return ended('no-hypothesis','No supported hypothesis: '+hypothesis.statement);
  const problem=hypothesisProblem(hypothesis,policy);
  if(problem)return ended('invalid-candidate',problem);
  const proposal=await parent.propose(decide.editor(policy.mode),request(policy,hypothesis,diagnosis,feedback.evidence,sourceFiles,state.lastExperiment));
  if(!proposal.diff.changes.length)return ended('no-hypothesis','The edit left the source unchanged.');
  const candidateId=proposal.folder.digest;
  if(candidateId===parentId||state.population.some(member=>member.source===candidateId)||history.some(outcome=>outcome.candidate===candidateId))return ended('duplicate','This candidate repeats an earlier one; choose an edit that differs from every earlier outcome.',candidateId);
  const candidatePaths=proposal.folder.filePaths();
  const candidateFiles=await Promise.all(candidatePaths.map(async path=>({path,text:await proposal.folder.readText(path)})));
  const checked=await evaluator.check(proposal.folder);
  if(!checked.valid){const reason='Compilation rejected: '+checked.diagnostics.join('\n');return ended('invalid-candidate',reason,candidateId,undefined,{sourceFiles:candidateFiles,training:[],reason});}
  const training=await evaluator.evaluate(proposal.folder,{split:'train',seed});
  const candidate=await evaluator.evaluate(proposal.folder,{split:'validation',seed});
  const objective=policy.objective??'quality';
  const improvement=better({quality:candidate.quality,cost:candidate.sourceBytes,modelCalls:candidate.modelCalls},{quality:measured.quality,cost:measured.sourceBytes,modelCalls:measured.modelCalls},{measure:objectiveMeasure(objective)});
  const reason='Measured candidate: train quality '+training.quality+', validation quality '+candidate.quality+' vs '+measured.quality+', validation calls '+candidate.modelCalls+' vs '+measured.modelCalls+', bytes '+candidate.sourceBytes+' vs '+measured.sourceBytes+'.';
  const trainingRows=evaluator.page(training.evidence);
  const diagnostics=trainingRows.map(row=>'Train '+(row.caseId??'case')+': '+(row.passed?'passed':'failed; actual '+JSON.stringify(row.value)+', expected '+JSON.stringify(row.expected))+'; requests '+(row.modelCalls??'unknown')+(row.error?'; '+row.error:'')).map(text=>text.slice(0,700));
  const candidateFeedback:ExperimentFeedback={diagnostics,sourceFiles:candidateFiles,training:trainingRows,validation:{quality:candidate.quality,...(candidate.modelCalls!==undefined?{modelCalls:candidate.modelCalls}:{})},reason};
  const failing=trainingRows.filter(row=>!row.passed).map(row=>row.caseId??'case');
  if(!training.gatesPassed||!candidate.gatesPassed||!improvement)return ended('rejected',reason,candidateId,failing,candidateFeedback);
  const installed=await parent.accept(proposal);
  return {source:installed.digest,parent:parentId,accepted:true,disposition:'accepted',reason:proposal.value.summary,candidate:candidateId,failing,quality:candidate.quality,sourceBytes:candidate.sourceBytes,...(candidate.modelCalls!==undefined?{modelCalls:candidate.modelCalls}:{}),scores:candidate.scores??[],feedback:candidateFeedback};
}
