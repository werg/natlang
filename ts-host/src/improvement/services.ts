/** Minimal injected surface. Locked confirmation is a top-level responsibility. */
export const EVALUATOR_DECLARATION = `export function check(folder: FolderSnapshot): Promise<{source:string;valid:boolean;diagnostics:string[]}>;
export type SkillUseEvent = {kind:'skill_use';phase:'offered'|'body_read'|'support_file_read'|'helper_invoked';skill_name:string;skill_revision:string;path?:string;helper_export?:string;interpretation?:'listed_in_invocation_opening_not_awareness'};
export type TrainingOutcome = {caseId:string;passed:boolean;quality:number;gates:Record<string,boolean>;modelCalls?:number;modelTrace?:{calls:unknown;observation:string;skillUse?:SkillUseEvent[]}[];skillUseTrace?:SkillUseEvent[];modelTraceTruncated?:boolean;serviceDeclarations?:Record<string,string>;failureKind?:'fixture'|'target'|'timeout';error?:string;value?:unknown;args?:unknown[];expected?:unknown;expectedFiles?:Record<string,string>;files?:Record<string,string>;evidence:string};
export function evaluate(folder: FolderSnapshot, request: {split:'train'|'validation';caseIds?:string[];seed?:number}): Promise<{source:string;split:string;suiteVersion:string;quality:number;modelCalls?:number;sourceBytes:number;gatesPassed:boolean;passed:number;total:number;evidence:string;outcomes?:TrainingOutcome[];scores?:{caseId:string;quality:number}[]}>;
/** Read diagnostic outcomes using report.evidence from an evaluate(...,{split:'train'}) report.
 * Validation report references are selection receipts, not readable evidence. Use their scores/quality directly.
 * Individual outcomes' evidence IDs are provenance, not page references. */
export function page(evidence:string,start?:number,limit?:number): TrainingOutcome[];`;
