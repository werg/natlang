import type {ImprovementPolicy,RewriteRequest,ExperimentContext,TrainingEvidence} from '../types';

type Measurement={source:string;quality:number;sourceBytes:number;modelCalls?:number;gatesPassed:boolean;evidence:string;scores?:{caseId:string;quality:number}[]};
type Proposal={folder:Snapshot;diff:{changes:unknown[]}};
export type Snapshot={digest:string;filePaths():string[];readText(path:string):Promise<string>;branch():Draft};
type Draft={filePaths():string[];readText(path:string):Promise<string>;apply(reducer:unknown,brief:string,context:ExperimentContext,policy:ImprovementPolicy):Promise<string>;propose(reducer:unknown,request:RewriteRequest):Promise<Proposal>;accept(proposal:Proposal):Promise<Snapshot>};
export type ExperimentFolder={at(source:string):Snapshot};
export type ExperimentEvaluator={evaluate(source:Snapshot,request:{split:'train'|'validation';seed:number}):Promise<Measurement>;page(reference:string):TrainingEvidence[];check(source:Snapshot):Promise<{valid:boolean;diagnostics:string[]}>};
