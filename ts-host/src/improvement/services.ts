/** Minimal injected surface. Locked confirmation is a top-level responsibility. */
export const EVALUATOR_DECLARATION = `export function check(folder: FolderSnapshot): Promise<{source:string;valid:boolean;diagnostics:string[]}>;
export type SkillUseEvent = {kind:'skill_use';phase:'offered'|'body_read'|'support_file_read'|'helper_invoked';skill_name:string;skill_revision:string;path?:string;helper_export?:string;interpretation?:'listed_in_invocation_opening_not_awareness'};
export type TrainingOutcome = {caseId:string;passed:boolean;quality:number;gates:Record<string,boolean>;modelCalls?:number;modelTrace?:{calls:unknown;observation:string;skillUse?:SkillUseEvent[]}[];skillUseTrace?:SkillUseEvent[];modelTraceTruncated?:boolean;serviceDeclarations?:Record<string,string>;failureKind?:'fixture'|'target'|'timeout';error?:string;value?:unknown;args?:unknown[];expected?:unknown;expectedFiles?:Record<string,string>;files?:Record<string,string>;evidence:string};
export function evaluate(folder: FolderSnapshot, request: {split:'train'|'validation';caseIds?:string[];seed?:number}): Promise<{source:string;split:string;suiteVersion:string;quality:number;modelCalls?:number;sourceBytes:number;gatesPassed:boolean;passed:number;total:number;evidence:string;outcomes?:TrainingOutcome[];scores?:{caseId:string;quality:number}[]}>;
/** Read diagnostic outcomes using report.evidence from an evaluate(...,{split:'train'}) report.
 * Validation report references are selection receipts, not readable evidence. Use their scores/quality directly.
 * Individual outcomes' evidence IDs are provenance, not page references. */
export function page(evidence:string,start?:number,limit?:number): TrainingOutcome[];`;
/** The journal of experiment plans: what is decided before an edit is recorded, and a resumed run recalls it. */
export const PLANS_DECLARATION = `/** The plan recorded for this experiment number, or null when it has none. */
export function recall(iteration: number): Promise<unknown>;
/** Record the plan of this experiment number before its edit begins. */
export function record(iteration: number, plan: unknown): Promise<void>;`;
/** Plans in a journal directory when there is one, else in memory for the run. */
export function planService(journal?: { read<T>(key: string): { value?: T } | undefined; record<T>(key: string, value: T): void }) {
  const memory = new Map<number, unknown>();
  return {
    recall: async (iteration: number) => journal ? journal.read<unknown>('plan:' + iteration)?.value ?? null : memory.get(iteration) ?? null,
    record: async (iteration: number, plan: unknown) => { if (journal) journal.record('plan:' + iteration, plan); else memory.set(iteration, plan); },
  };
}
