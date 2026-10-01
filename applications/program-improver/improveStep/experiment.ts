import type {ExperimentResult,ExperimentFeedback,ImprovementPolicy} from '../types';
import type {ExperimentFolder,ExperimentEvaluator} from './capabilities';
import {request} from './planExperiment/bookkeeping';
import {inspect} from './planExperiment/feedback';
/** One finite experiment. Hypothesis and source edits remain semantic decisions. */
export async function test(folder:ExperimentFolder,evaluator:ExperimentEvaluator,parentId:string,editor:unknown,policy:ImprovementPolicy,hypothesis:string):Promise<ExperimentResult> {
  const parent=folder.at(parentId).branch();
  const measured=await evaluator.evaluate(folder.at(parentId),{split:'validation',seed:policy.seed??0});
  const feedback=await inspect(folder,evaluator,parentId,policy);
  const rejected=(reason:string,feedback?:ExperimentFeedback):ExperimentResult=>({source:parentId,parent:parentId,accepted:false,reason,quality:measured.quality,sourceBytes:measured.sourceBytes,...(measured.modelCalls!==undefined?{modelCalls:measured.modelCalls}:{}),scores:measured.scores??[],...(feedback?{feedback}:{})});
  if(feedback.opportunity.kind==='fixture')return rejected('fixture-error: '+feedback.evidence.filter(row=>row.failureKind==='fixture').map(row=>row.error??'Independent fixture failed.').join('\n'));
  if(!hypothesis.trim())return rejected('No supported hypothesis.');
  const paths=parent.filePaths();
  const sourceFiles=await Promise.all(paths.map(async path=>({path,text:await parent.readText(path)})));
  const proposal=await parent.propose(editor,request(policy,hypothesis,feedback.evidence,sourceFiles));
  if(!proposal.diff.changes.length)return rejected('Editor made no source change.');
  const candidatePaths=proposal.folder.filePaths();
  const candidateFiles=await Promise.all(candidatePaths.map(async path=>({path,text:await proposal.folder.readText(path)})));
  const checked=await evaluator.check(proposal.folder);
  if(!checked.valid){const reason='Compilation rejected: '+checked.diagnostics.join('\n');return rejected(reason,{sourceFiles:candidateFiles,training:[],reason});}
  const training=await evaluator.evaluate(proposal.folder,{split:'train',seed:policy.seed??0});
  const candidate=await evaluator.evaluate(proposal.folder,{split:'validation',seed:policy.seed??0});
  const objective=policy.objective??'quality';
  const improvement=candidate.quality>measured.quality || candidate.quality===measured.quality && (
    objective==='source-size' && candidate.sourceBytes<measured.sourceBytes ||
    objective==='model-calls' && candidate.modelCalls!==undefined && measured.modelCalls!==undefined && candidate.modelCalls<measured.modelCalls);
  const reason='Measured candidate: train quality '+training.quality+', validation quality '+candidate.quality+' vs '+measured.quality+', validation calls '+candidate.modelCalls+' vs '+measured.modelCalls+', bytes '+candidate.sourceBytes+' vs '+measured.sourceBytes+'.';
  const candidateFeedback:ExperimentFeedback={sourceFiles:candidateFiles,training:evaluator.page(training.evidence),validation:{quality:candidate.quality,...(candidate.modelCalls!==undefined?{modelCalls:candidate.modelCalls}:{})},reason};
  if(!training.gatesPassed||!candidate.gatesPassed||!improvement)return rejected(reason,candidateFeedback);
  const installed=await parent.accept(proposal);
  return {source:installed.digest,parent:parentId,accepted:true,reason:hypothesis,quality:candidate.quality,sourceBytes:candidate.sourceBytes,...(candidate.modelCalls!==undefined?{modelCalls:candidate.modelCalls}:{}),scores:candidate.scores??[],feedback:candidateFeedback};
}
