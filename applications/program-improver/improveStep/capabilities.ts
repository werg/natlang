import type {Diagnosis,DiagnoseRequest,EditRequest,EditResult,ExperimentOutcome,Hypothesis,HypothesizeRequest,ImprovementPolicy,Opportunity,OpportunityFacts,PopulationMember,StopDecision,StopFacts,TrainingEvidence} from '../types';

type Measurement={source:string;quality:number;sourceBytes:number;modelCalls?:number;gatesPassed:boolean;evidence:string;scores?:{caseId:string;quality:number}[]};
type Proposal={folder:Snapshot;value:EditResult;diff:{changes:{path:string}[]}};
export type Snapshot={digest:string;filePaths():string[];readText(path:string):Promise<string>;branch():Draft};
type Draft={filePaths():string[];readText(path:string):Promise<string>;propose(reducer:unknown,request:EditRequest):Promise<Proposal>;accept(proposal:Proposal):Promise<Snapshot>};
export type ExperimentFolder={at(source:string):Snapshot};
export type ExperimentEvaluator={evaluate(source:Snapshot,request:{split:'train'|'validation';seed:number}):Promise<Measurement>;page(reference:string):TrainingEvidence[];check(source:Snapshot):Promise<{valid:boolean;diagnostics:string[]}>};
/** What an experiment decides with: the pluggable search policies and the semantic stages (policies.ts supplies them). */
export type Decisions={
  findOpportunity(policy:ImprovementPolicy,facts:OpportunityFacts):Promise<Opportunity>;
  chooseParent(policy:ImprovementPolicy,population:PopulationMember[],history:ExperimentOutcome[],seed:number):Promise<string>;
  selectIncumbent(policy:ImprovementPolicy,population:PopulationMember[],incumbent:string):Promise<string>;
  shouldStop(policy:ImprovementPolicy,facts:StopFacts):Promise<StopDecision>;
  diagnose(request:DiagnoseRequest):Promise<Diagnosis>;
  hypothesize(request:HypothesizeRequest):Promise<Hypothesis>;
  editor(mode:'instruction'|'structural'):unknown;
};
/** The journal of experiment plans (the `plans` service): a resumed run recalls the plan recorded before the edit. */
export type PlanJournal={recall(iteration:number):Promise<unknown>;record(iteration:number,plan:unknown):Promise<void>};
