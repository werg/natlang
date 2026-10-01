import { improveProgram, type ImproveProgramOptions } from './program.js';
import type { FolderSnapshot } from '../native/scoped-fs.js';
import type { ProgramEvaluator, TransformationSpec, TransformationResult, ImprovementCase } from './types.js';
/** Report the strength actually supported by evidence; finite test success is empirical support. */
export async function checkTransformation(evaluator: ProgramEvaluator, before: FolderSnapshot, after: FolderSnapshot, spec: TransformationSpec): Promise<TransformationResult> {
  const checked = await evaluator.check(after), evidence:string[]=[], obligations:import('./types.js').TransformationObligation[]=[];
  obligations.push({obligation:'compile and declared external contract',status:checked.valid?'verified':'violated',method:'independent compiler and contract checker',evidence:[]});
  if(spec.allowedFiles){
    const {validateSourceEdit}=await import('./source-policy.js'),{sourceFiles}=await import('./host.js');
    const errors=validateSourceEdit(sourceFiles(before),sourceFiles(after),spec.mode??'structural',spec.allowedFiles);
    obligations.push({obligation:'editable source scope',status:errors.length?'violated':'verified',method:errors.join('\n')||'exact source projection comparison',evidence:[]});
  }
  if(checked.valid){
    const baseline=await evaluator.evaluate(before,{split:'validation'}),candidate=await evaluator.evaluate(after,{split:'validation'});
    evidence.push(baseline.evidence,candidate.evidence);
    const comparable=!!baseline.scores?.length && baseline.scores.length===candidate.scores?.length;
    const paired=comparable&&baseline.scores!.every(left=>candidate.scores!.some(right=>right.caseId===left.caseId&&right.quality>=left.quality));
    const supported=candidate.gatesPassed&&candidate.quality>=baseline.quality&&paired;
    for(const claim of [...spec.preserves,...spec.changes])obligations.push({obligation:claim,status:before.digest===after.digest?(spec.changes.includes(claim)?'violated':'verified'):!comparable?'unverified':supported?'empirically-supported':'violated',method:before.digest===after.digest?(spec.changes.includes(claim)?'requested change left source byte-identical':'byte-identical source'):'paired independent validation cases; a requested change is supported only by author-supplied cases',evidence:[...evidence]});
    for(const check of spec.checks){
      if(check==='compile'||check==='external-contract')continue;
      if(check==='validation-regressions')obligations.push({obligation:check,status:!comparable?'unverified':supported?'empirically-supported':'violated',method:'paired independent validation',evidence:[...evidence]});
      else obligations.push({obligation:check,status:'unverified',method:'no independent checker supplied for this obligation',evidence:[]});
    }
  }
  const violated=obligations.some(row=>row.status==='violated'),unverified=obligations.some(row=>row.status==='unverified');
  const status=violated?'violated':unverified?'unverified':obligations.some(row=>row.status==='empirically-supported')?'empirically-supported':'verified';
  return {status,source:after.digest,spec,evidence,obligations,eligible:!violated&&!unverified,reason:violated?'An independent obligation failed.':unverified?'Required obligations lack independent evidence.':'All required obligations have the stated support; finite cases do not establish universal equivalence.'};
}
/** Exhaustive counterexample search over a supplied finite domain, never over locked cases. */
export async function finiteCounterexamples<T>(domain: readonly T[], check: (input: T) => Promise<boolean>, maxChecks: number): Promise<{ checked: number; complete: boolean; counterexamples: T[] }> {
  if (!Number.isSafeInteger(maxChecks) || maxChecks < 0) throw new RangeError('finite checker bound required');
  const inputs = domain.slice(0, maxChecks), counterexamples: T[] = [];
  for (const input of inputs) if (!await check(input)) counterexamples.push(input);
  return { checked: inputs.length, complete: inputs.length === domain.length, counterexamples };
}
export function trainingCounterexample(id: string, group: string, args: unknown[], expected: unknown): ImprovementCase {
  return { id, group, split: 'train', args: structuredClone(args), expected: structuredClone(expected) };
}
/** Frozen self-improvement adoption is between runs, never a mutable recursive invocation. */
export class FrozenImprover {
  private running = 0;
  constructor(private selected: FolderSnapshot, readonly maxEvaluationDepth = 2) {
    if (!Number.isSafeInteger(maxEvaluationDepth) || maxEvaluationDepth < 1) throw new RangeError('finite evaluation depth required');
  }
  snapshot(): FolderSnapshot { return this.selected; }
  improve(options: Omit<ImproveProgramOptions, 'improverSource'>) {
    return this.run(0, source => improveProgram({ ...options, improverSource: source }));
  }
  async run<T>(depth: number, invoke: (source: FolderSnapshot) => Promise<T>): Promise<T> {
    if (!Number.isSafeInteger(depth) || depth < 0 || depth >= this.maxEvaluationDepth) throw new Error('self-improvement evaluation depth exhausted');
    const frozen = this.selected; this.running++;
    try { return await invoke(frozen); } finally { this.running--; }
  }
  adopt(source: FolderSnapshot): void { if (this.running) throw new Error('adopt a frozen improver only between runs'); this.selected = source; }
}
