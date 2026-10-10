export type Objective='quality'|'source-size'|'model-calls';
export type PolicyMode='crisp'|'nl'|'shadow';
/** The mode of each pluggable search policy; an absent policy runs its crisp default. */
export type PolicySettings={findOpportunity?:PolicyMode;chooseParent?:PolicyMode;selectIncumbent?:PolicyMode;shouldStop?:PolicyMode;chooseMove?:PolicyMode;chooseComponents?:PolicyMode};
/** How an experiment ended. The text in `reason` is for reading; decisions use this. */
export type Disposition='accepted'|'rejected'|'no-hypothesis'|'no-opportunity'|'fixture-error'|'invalid-candidate'|'duplicate';
export type PopulationMember = {source:string;quality:number;parent:string;cost?:number;modelCalls?:number;scores?:{caseId:string;quality:number}[]};
export type ExperimentOutcome = { source: string; parent: string; accepted: boolean; selected: boolean; reason: string; disposition?: Disposition; candidate?: string; failing?: string[] };
export type SearchState = { iteration: number; done: boolean; incumbent: string; quality: number; population: PopulationMember[]; history: ExperimentOutcome[]; stopReason: string; lastExperiment?:ExperimentFeedback };
export type ImprovementPolicy = {seed?:number; maxExperiments: number; mode: 'instruction' | 'structural'; strategy: 'gepa' | 'adaptive'; objective?:Objective; goal: string; outerLesson?:string; allowedFiles: string[]; maxPopulation: number; transformation?:string; policies?:PolicySettings };
export type SourceFile = {path:string;text:string};

export type Opportunity = {kind:'fixture'|'quality'|'efficiency'|'source-size'|'none';reason:string};
export type Observation = {caseId:string;whatHappened:string;expected:string;category:'wrong-result'|'execution-method'|'wasted-action'|'fixture'};
export type Diagnosis = {observations:Observation[];pattern:string};
export type DiagnoseRequest = {evidence:unknown[];brief:string;goal:string;objective:Objective;opportunity:Opportunity;history:ExperimentOutcome[];lastExperiment?:ExperimentFeedback};
export type Hypothesis = {kind:'instruction'|'helper'|'structure'|'efficiency'|'none';statement:string;files:string[];predictedChange:string};
export type HypothesizeRequest = {diagnosis:Diagnosis;sourceFiles:SourceFile[];mode:'instruction'|'structural';allowedFiles:string[];history:ExperimentOutcome[];transformation?:string};
/** What the edit stage reads. The host derives the changed paths from the diff. */
export type EditRequest = {brief:string;goal:string;mode:'instruction'|'structural';objective?:Objective;hypothesis:Hypothesis;diagnosis:Diagnosis;sourceFiles:SourceFile[];allowedFiles:string[];serviceDeclarations?:Record<string,string>;transformation?:string;preserves?:string[];changes?:string[];checks?:string[]};
export type EditResult = { summary: string; preserves: string[] };

/** Facts the search policies decide from (all computed by crisp code). */
export type ParentChoice = {id:string;quality:number;wonCases:string[];timesParent:number};
export type IncumbentChoice = {id:string;quality:number;sourceBytes?:number;modelCalls?:number};
export type StopDecision = {stop:boolean;reason:string};
export type StopFacts = {disposition:Disposition;reason:string;selectedQuality:number;objective:Objective;iteration:number;maxExperiments:number;recent:{disposition:Disposition;failingCases:string[]}[]};
/** What the counterexample loop's stop decision reads, measured by the host after a round: the examples admitted, the oracle allowance left, and the repair (absent when nothing was admitted). */
export type CounterexampleStopFacts = {search:'counterexample';round:number;admitted:number;remainingChecks:number;repair?:{eligible:boolean;trainingQuality:number;disposition:string}};
export type OpportunityFacts = {objective:Objective;rows:{caseId:string;passed:boolean;modelCalls?:number;failureKind?:string}[]};
export type MoveFacts = {iteration:number;members:{id:string;changedKeys:string[]}[]};
export type ComponentFacts = {eligible:{key:string;coveredCases:number;failingCases:number;proposals:number;accepts:number}[]};

export type ComponentValue = {kind:'program.guidance';text:string} | {kind:'lambda.instructions';template:{segments:string[];slotIds:string[]}};
export type Candidate = Record<string,ComponentValue>;
export type ComponentRewriteRequest = {keys:string[];components:unknown[];feedback:unknown;problem?:string};
export type ComponentLoopState = {iteration:number;incumbent:string;done:boolean;stopReason:string};
export type CounterexampleRequest={goal:string;evidence:unknown[];maxSuggestions?:number};
export type CounterexampleSuggestions={inputs:unknown[][];reason:string};
export type CounterexampleState={round:number;done:boolean;suite:string;remainingChecks:number;quality:number;reason:string};
export type SearchFrame = {current:SearchState;baselineBytes:number};
export type ExperimentResult = {source:string;parent:string;accepted:boolean;disposition:Disposition;reason:string;candidate?:string;failing?:string[];quality:number;sourceBytes:number;modelCalls?:number;scores:{caseId:string;quality:number}[];feedback?:ExperimentFeedback};


export type SkillUseEvent={kind:'skill_use';phase:'offered'|'body_read'|'support_file_read'|'helper_invoked';skill_name:string;skill_revision:string;invocation_id?:string|null;path?:string;helper_export?:string;interpretation?:'listed_in_invocation_opening_not_awareness'};
export type TrainingEvidence={caseId?:string;quality?:number;gates?:Record<string,boolean>;evidence?:string;modelTraceTruncated?:boolean;expectedFiles?:Record<string,string>;files?:Record<string,string>;passed:boolean;modelCalls?:number;failureKind?:string;error?:string;value?:unknown;expected?:unknown;args?:unknown[];modelTrace?:{calls:unknown;observation:string;skillUse?:SkillUseEvent[]}[];skillUseTrace?:SkillUseEvent[];serviceDeclarations?:Record<string,string>};

export type ExperimentFeedback={diagnostics?:string[];sourceFiles:SourceFile[];training:TrainingEvidence[];validation?:{quality:number;modelCalls?:number};reason:string};
