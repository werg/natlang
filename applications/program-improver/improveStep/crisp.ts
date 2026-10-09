import {chooseIncumbent,drawParent,leaders,wonCases,type Member} from 'natlang:gepa';
import type {ExperimentOutcome,ImprovementPolicy,IncumbentChoice,Objective,Opportunity,OpportunityFacts,ParentChoice,PopulationMember,StopDecision,StopFacts,TrainingEvidence} from '../types';

/**
 * The crisp side of the search policies: the defaults, which equal the program's behavior before the policies were
 * pluggable, and the verifiers that bound a natural-language choice. Pure functions; no model is called here.
 * policies.ts wires each default and its natural-language function into one pluggable policy.
 */
/** A choice outside its bound fails the search, not the infrastructure: the host classifies the error by this name. */
const must=(ok:boolean,message:string):void=>{if(!ok)throw Object.assign(Error(message),{name:'PolicyBoundError'});};

/** The selection math reads members by id; a population member is identified by its source digest. */
export function members(population:PopulationMember[]):Member[] {
  return population.map(member=>({id:member.source,quality:member.quality,scores:member.scores??[],...(member.cost!==undefined?{cost:member.cost}:{}),...(member.modelCalls!==undefined?{modelCalls:member.modelCalls}:{})}));
}

/** The rows an opportunity is judged from. */
export function opportunityFacts(policy:ImprovementPolicy,evidence:TrainingEvidence[]):OpportunityFacts {
  return {objective:policy.objective??'quality',rows:evidence.map(row=>({caseId:row.caseId??'case',passed:row.passed,...(row.modelCalls!==undefined?{modelCalls:row.modelCalls}:{}),...(row.failureKind?{failureKind:row.failureKind}:{})}))};
}
/** Identify an evidenced objective; the model still decides which change can help. */
export function crispOpportunity(facts:OpportunityFacts):Opportunity {
  const failed=facts.rows.filter(row=>!row.passed).map(row=>row.caseId);
  if(facts.rows.some(row=>row.failureKind==='fixture'))return {kind:'fixture',reason:'Independent fixture failed before target execution.'};
  if(failed.length)return {kind:'quality',reason:'A current target execution failed; diagnose its actual outcome and trace.'};
  if(facts.objective==='model-calls'&&facts.rows.some(row=>row.modelCalls!==undefined&&row.modelCalls>1))return {kind:'efficiency',reason:'Correct answers still took multiple model requests. Inspect the trace for avoidable inspection, repair, delegation or a separate completion request. Passing quality does not complete the cost objective.'};
  if(facts.objective==='source-size')return {kind:'source-size',reason:'Correct answers permit a behavior-preserving source simplification; test a real size reduction.'};
  return {kind:'none',reason:'No observed failure or cost opportunity supports this objective.'};
}
/** An opportunity must be supported by the rows: `fixture` and `quality` need such a row, `efficiency` a passing one. */
export function verifyOpportunity(found:Opportunity,facts:OpportunityFacts):Opportunity {
  const kinds=['fixture','quality','efficiency','source-size','none'];
  must(kinds.includes(found.kind),'An opportunity kind is one of '+kinds.join(', ')+'.');
  must((found.kind==='fixture')===facts.rows.some(row=>row.failureKind==='fixture'),'The kind is `fixture` exactly when a training row has failureKind `fixture`.');
  must(found.kind!=='quality'||facts.rows.some(row=>!row.passed),'The kind `quality` needs a failed training row.');
  must(found.kind!=='efficiency'||facts.rows.some(row=>row.passed),'The kind `efficiency` needs a passing training row.');
  return found;
}

/** The population as parent candidates: the cases each wins and how often it has been a parent. */
export function parentChoices(population:PopulationMember[],history:ExperimentOutcome[]):ParentChoice[] {
  const won=wonCases(members(population));
  return population.map(member=>({id:member.source,quality:member.quality,wonCases:won[member.source]??[],timesParent:history.filter(outcome=>outcome.parent===member.source).length}));
}
/** The crisp parent: a seeded draw over the per-case frontier. */
export function crispParent(population:PopulationMember[],seed:number):string {
  return drawParent(members(population),seed).id;
}
/** A parent is a population member. */
export function verifyParent(id:string,population:PopulationMember[]):string {
  must(population.some(member=>member.source===id),'A parent is the id of a population member; choose one of: '+population.map(member=>member.source).join(', ')+'.');
  return id;
}

/** The members an incumbent may be: the highest quality, narrowed by the objective's measure. */
export function incumbentChoices(population:PopulationMember[],objective:Objective):IncumbentChoice[] {
  const allowed=leaders(members(population),objective);
  return population.filter(member=>allowed.includes(member.source)).map(member=>({id:member.source,quality:member.quality,...(member.cost!==undefined?{sourceBytes:member.cost}:{}),...(member.modelCalls!==undefined?{modelCalls:member.modelCalls}:{})}));
}
/** The crisp incumbent: the current one while it remains a leader, otherwise the leader with the smallest id. */
export function crispIncumbent(population:PopulationMember[],incumbent:string,objective:Objective):string {
  return chooseIncumbent(members(population),incumbent,objective);
}
/** The incumbent is among the best-quality members. */
export function verifyIncumbent(id:string,population:PopulationMember[],objective:Objective):string {
  const allowed=leaders(members(population),objective);
  must(allowed.includes(id),'The incumbent is one of the best-quality members: '+allowed.join(', ')+'.');
  return id;
}

/** The stop decision of the declared defaults. */
export function crispStop(facts:StopFacts):StopDecision {
  if(facts.disposition==='fixture-error')return {stop:true,reason:facts.reason};
  if(facts.selectedQuality===1&&facts.objective==='quality')return {stop:true,reason:'Declared objective satisfied.'};
  if(facts.disposition==='no-hypothesis'||facts.disposition==='no-opportunity')return {stop:true,reason:'No further evidenced change.'};
  if(facts.iteration+1>=facts.maxExperiments)return {stop:true,reason:'Declared experiments completed.'};
  return {stop:false,reason:'Continue with a distinct hypothesis.'};
}
/** A stop below `maxExperiments` is allowed; a continue at it is not. */
export function verifyStop(decision:StopDecision,facts:StopFacts):StopDecision {
  must(typeof decision.stop==='boolean','A stop decision has a boolean `stop`.');
  must(decision.stop||facts.iteration+1<facts.maxExperiments,'The search stops when iteration + 1 reaches maxExperiments ('+facts.maxExperiments+').');
  return decision;
}
