import type {ImprovementPolicy,Opportunity,TrainingEvidence} from '../types';
import type {ExperimentFolder,ExperimentEvaluator,Decisions} from './capabilities';
import {opportunityFacts} from './crisp';
/** Read measured training feedback, never validation answers. */
export async function inspect(folder:ExperimentFolder,evaluator:ExperimentEvaluator,parentId:string,policy:ImprovementPolicy,decide:Pick<Decisions,'findOpportunity'>):Promise<{evidence:TrainingEvidence[];opportunity:Opportunity}> {
  const training=await evaluator.evaluate(folder.at(parentId),{split:'train',seed:policy.seed??0});
  const evidence=evaluator.page(training.evidence);
  return {evidence,opportunity:await decide.findOpportunity(policy,opportunityFacts(policy,evidence))};
}
