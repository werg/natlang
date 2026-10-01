import type {ImprovementPolicy,TrainingEvidence} from '../../types';
import type {ExperimentFolder,ExperimentEvaluator} from '../capabilities';
import {opportunity} from './bookkeeping';
/** Read measured training feedback, never validation answers. */
export async function inspect(folder:ExperimentFolder,evaluator:ExperimentEvaluator,parentId:string,policy:ImprovementPolicy):Promise<{evidence:TrainingEvidence[];opportunity:{kind:'fixture'|'quality'|'efficiency'|'source-size'|'none';reason:string}}> {
  const training=await evaluator.evaluate(folder.at(parentId),{split:'train',seed:policy.seed??0});
  const evidence=evaluator.page(training.evidence);
  return {evidence,opportunity:opportunity(policy,evidence)};
}
