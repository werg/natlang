export type PopulationMember = {source:string;quality:number;parent:string;cost?:number;modelCalls?:number;scores?:{caseId:string;quality:number}[]};
export type ExperimentOutcome = { source: string; parent: string; accepted: boolean; selected: boolean; reason: string };
export type SearchState = { iteration: number; done: boolean; incumbent: string; quality: number; population: PopulationMember[]; history: ExperimentOutcome[]; stopReason: string; lastExperiment?:unknown };
export type ImprovementPolicy = {seed?:number; maxExperiments: number; mode: 'instruction' | 'structural'; strategy: 'gepa' | 'adaptive'; objective?:'quality'|'source-size'|'model-calls'; goal: string; outerLesson?:string; allowedFiles: string[]; maxPopulation: number };
export type SourceFile = {path:string;text:string};
export type RewriteRequest = {brief:string; history?:ExperimentOutcome[]; lastExperiment?:unknown; objective?:'quality'|'source-size'|'model-calls'; goal: string; mode: string; hypothesis: string; sourceFiles:SourceFile[]; evidence: unknown[]; allowedFiles: string[]; preserves?:string[]; changes?:string[]; checks?:string[] };
export type RewriteResult = { summary: string; changed: string[]; preserves: string[] };

export type ComponentValue = {kind:'program.guidance';text:string} | {kind:'lambda.instructions';template:{segments:string[];slotIds:string[]}};
export type Candidate = Record<string,ComponentValue>;
export type ComponentRewriteRequest = {keys:string[];components:unknown[];feedback:unknown};
export type ComponentLoopState = {iteration:number;incumbent:string;done:boolean;stopReason:string};
export type CounterexampleState={round:number;done:boolean;suite:string;remainingChecks:number;quality:number;reason:string};
export type SearchFrame = {current:SearchState;baselineBytes:number};
export type ExperimentResult = {source:string;parent:string;accepted:boolean;reason:string;quality:number;sourceBytes:number;modelCalls?:number;scores:{caseId:string;quality:number}[];feedback?:ExperimentFeedback};


export type TrainingEvidence={caseId?:string;quality?:number;gates?:Record<string,boolean>;evidence?:string;modelTraceTruncated?:boolean;expectedFiles?:Record<string,string>;passed:boolean;modelCalls?:number;failureKind?:string;error?:string;value?:unknown;expected?:unknown;args?:unknown[];modelTrace?:{calls:unknown;observation:string}[];serviceDeclarations?:Record<string,string>};

export type ExperimentFeedback={diagnostics?:string[];sourceFiles:SourceFile[];training:TrainingEvidence[];validation?:{quality:number;modelCalls?:number};reason:string};
