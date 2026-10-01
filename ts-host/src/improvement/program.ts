import {NATLANG_COMPILE_VERSION} from '../compiler/intrinsics.js';
import {TOOLS_PROMPT} from '../native/prompt.js';
import { OperationJournal } from './operations.js';
import { fingerprint } from '../adaptation/identity.js';
import { Folder, FolderSnapshot } from '../native/scoped-fs.js';
import { createNatlangRuntime, type InvocationTrace, type ModelDriver } from '../runtime/node.js';
import { loadVirtualNatlang } from '../runtime/virtual-project.js';
import { AUTHORED_IMPROVER } from './authored-source.js';
import { SourceEvaluator, sourceFiles,SOURCE_EVALUATION_VERSION } from './host.js';
import { EVALUATOR_DECLARATION } from './services.js';
import { IterationLimitError,IterationDivergedError } from '../runtime/iterate.js';
import {NatlangCallError} from '../runtime/kernel.js';
import { BudgetExhausted, UsageGateway } from '../evaluation/usage.js';
import type { BudgetLimits } from '../evaluation/types.js';
import type { ImprovementCase, ProgramContract, TransformationSpec } from './types.js';
export type ImprovementPolicy = { maxExperiments: number; mode: 'instruction' | 'structural'; strategy: 'gepa' | 'adaptive'; objective?:'quality'|'source-size'|'model-calls'; goal: string; outerLesson?:string; allowedFiles: string[]; maxPopulation: number };
export type ImprovementState = { iteration: number; done: boolean; incumbent: string; quality: number; population: { source: string; quality: number; parent: string }[];
  history: { source: string; parent: string; accepted: boolean; selected: boolean; reason: string }[]; stopReason: string };
export type ImproveProgramOptions = { folder: Folder; contract: ProgramContract; cases: ImprovementCase[]; policy: ImprovementPolicy;
  improver: ModelDriver; executor: ModelDriver; executorId: string; executorTimeoutMs?:number; budget: BudgetLimits; signal?: AbortSignal; trace?: (trace: InvocationTrace) => void; seed?: number; directory?: string; improverSource?: FolderSnapshot; gateway?:UsageGateway; evaluationLevel?:1|2; executeCase?: import('./host.js').SourceCaseExecution; singleStep?:boolean; transformation?:TransformationSpec };
class InvalidImprovementState extends Error {}
/** The SDK entry runs an authored reducer; native services supply only source and evidence authority. */
export async function improveProgram(options: ImproveProgramOptions) {
  if (!Number.isSafeInteger(options.policy.maxExperiments) || options.policy.maxExperiments < 1 || !Number.isSafeInteger(options.policy.maxPopulation) || options.policy.maxPopulation < 2) throw new RangeError('finite experiment and population limits are required');
  if(options.executorTimeoutMs!==undefined&&(!Number.isSafeInteger(options.executorTimeoutMs)||options.executorTimeoutMs<1))throw new RangeError('positive finite executor timeout required');
  if(options.policy.objective!==undefined&&!['quality','source-size','model-calls'].includes(options.policy.objective))throw new RangeError('unknown improvement objective');
  const authored = options.improverSource ? sourceFiles(options.improverSource) : AUTHORED_IMPROVER;
  // The selected improver is frozen for this invocation; adoption only affects later invocations.
  const step = loadVirtualNatlang(authored, 'improveStep.nl');
  const journal = options.directory ? new OperationJournal(options.directory) : undefined;
  const identity = fingerprint({ runtime:6,compiler:NATLANG_COMPILE_VERSION,evaluation:SOURCE_EVALUATION_VERSION,authored, opening:{prompt:TOOLS_PROMPT,tools:"native-default",contextTokens:16384,limits:{maxTurns:16,maxTokens:24000,turnTokens:2048,maxFailureRepairs:4,maxRequests:options.budget.maxModelCalls}}, seed: options.seed ?? 0, source: options.folder.snapshot().digest, contract: options.contract, cases: options.cases, policy: options.policy, transformation:options.transformation??null, executor: options.executorId, executorTimeoutMs:options.executorTimeoutMs??120000, execution:options.executeCase?.identity ?? null, evaluationLevel:options.evaluationLevel??1, budget: options.budget });
  const savedIdentity = journal?.read<string>('run-identity')?.value;
  if (savedIdentity && savedIdentity !== identity) throw new Error('resume requires the same frozen source, cases, policy, executor and allocation');
  journal?.record('run-identity', identity);
  if (options.gateway && fingerprint(options.gateway.limits) !== fingerprint(options.budget)) throw new Error('nested improvement must use the parent allocation');
  const gateway = options.gateway ?? new UsageGateway(options.budget, journal?.read<import('../evaluation/usage.js').BudgetLedger>('ledger')?.value);
  if (journal && !options.gateway) gateway.onUpdate = ledger => journal.record('ledger', ledger);
  const evaluator = new SourceEvaluator(options.contract, options.cases, options.executor, gateway, { signal: options.signal, timeoutMs:options.executorTimeoutMs, executorId: options.executorId, journal, executeCase: options.executeCase, evaluationLevel:options.evaluationLevel??1, sourcePolicy: { baseline: sourceFiles(options.folder.snapshot()), mode: options.policy.mode, allowedFiles: options.policy.allowedFiles } });
  const baselineSource=options.folder.snapshot();
  const restore=(id:string):FolderSnapshot=>{
    try{return options.folder.at(id);}catch(error){
      const files=journal?.read<Record<string,string>>('source:'+id)?.value;if(!files)throw error;
      const branch=baselineSource.branch();for(const path of branch.filePaths())if(!(path in files))branch.remove(path);
      for(const [path,text]of Object.entries(files))branch.writeText(path,text);
      const snapshot=branch.snapshot();if(snapshot.digest!==id)throw Error('stored source digest mismatch');return snapshot;
    }
  };
  journal?.record('source:'+baselineSource.digest,sourceFiles(baselineSource));
  const evaluate = (folder:FolderSnapshot,request:{split:'train'|'validation';caseIds?:string[];seed?:number}) => {
    const seed=options.seed??0;
    if(request.seed!==undefined&&request.seed!==seed)throw new Error('evaluation seed is pinned by the improvement invocation: '+seed);
    return evaluator.evaluate(folder,{...request,seed});
  };
  const task = createNatlangRuntime({ model: { driver: (request,signal) => gateway.request(options.improver,request,signal??options.signal,'reflection'), maxTurns: 16, maxTokens: 24000, turnTokens: 2048, maxFailureRepairs: 4 }, signal: options.signal, seed:{mode:'derived',root:options.seed??0},codeEdits: 'deny', network: false,
    trace: options.trace, onFolderProposal: () => gateway.reserve('proposals', 1, options.signal), services: { evaluator: { check: evaluator.check.bind(evaluator), evaluate, page: evaluator.page.bind(evaluator) } },
    serviceDeclarations: { evaluator: EVALUATOR_DECLARATION },
    serviceScopes: { evaluator:['improveStep.nl'] }, limits: { maxEpisodes: options.budget.maxModelCalls, timeoutMs: options.budget.maxElapsedMs } });
  const initial: ImprovementState = { iteration: 0, done: false, incumbent: baselineSource.digest, quality: 0, population: [], history: [], stopReason: '' };
  const checkpoint = journal?.checkpoint<{ source: string; state: ImprovementState }>();
  let revision = checkpoint?.revision ?? 0;
  for(const member of checkpoint?.value.state.population??[])restore(member.source);
  const startingFolder = checkpoint ? restore(checkpoint.value.source).branch() : options.folder;
  let lastValid = {folder:startingFolder.snapshot(),state:checkpoint?.value.state ?? initial};
  let result: {folder:FolderSnapshot;state:ImprovementState};
  try { result = await task.run(() => startingFolder.iterateOn<ImprovementState>(step, checkpoint?.value.state ?? initial, { ...options.policy, seed: options.seed ?? 0 })
    .withMeasure(state => options.policy.maxExperiments - state.iteration).withSiteId('program-improver/v1')
    .onStep(async event => {
      if (event.kind !== 'step' || !event.state) return;
      const next=event.state;
      if(next.state.iteration!==lastValid.state.iteration+1)throw new InvalidImprovementState('an experiment must advance iteration exactly once');
      if(next.state.incumbent!==next.folder.digest)throw new InvalidImprovementState('selected state does not match committed source');
      if(next.folder.digest!==baselineSource.digest){const checked=await evaluator.check(next.folder);if(!checked.valid)throw new InvalidImprovementState('selected source failed compilation or edit policy: '+checked.diagnostics.join('\n'));}
      const measured=await evaluate(next.folder,{split:'validation'});
      if(!Number.isFinite(next.state.quality)||Math.abs(measured.quality-next.state.quality)>1e-12)throw new InvalidImprovementState('reported quality does not match independently executed selected source');
      for(const member of next.state.population){
        const memberSource=next.folder.at(member.source);
        const report=await evaluate(memberSource,{split:'validation'});
        if(!Number.isFinite(member.quality)||Math.abs(member.quality-report.quality)>1e-12)throw new InvalidImprovementState('population quality does not match independent measurement');
        const calls=(member as {modelCalls?:number}).modelCalls;
        if(calls!==undefined&&calls!==report.modelCalls)throw new InvalidImprovementState('population request count does not match independent measurement');
      }
      if (journal) {
        journal.record('source:'+event.state.folder.digest,sourceFiles(event.state.folder));
        for(const member of event.state.state.population)journal.record('source:'+member.source,sourceFiles(event.state.folder.at(member.source)));
        revision = journal.commit(revision, { source: event.state.folder.digest, state: event.state.state });
      }
      lastValid=next;
    }).until(state => state.done || options.singleStep === true));
  } catch (error) {
    const committed = journal?.checkpoint<{source:string;state:ImprovementState}>()?.value;
    const folder = committed ? restore(committed.source) : lastValid.folder;
    const state = committed?.state ?? lastValid.state;
    const interrupted = options.signal?.aborted || error instanceof IterationLimitError || error instanceof BudgetExhausted || gateway.ledger.usage.modelCalls >= options.budget.maxModelCalls;
    const semanticFailure=error instanceof NatlangCallError||(error instanceof Error&&error.cause instanceof NatlangCallError);
    const files=sourceFiles(folder),baseFiles=sourceFiles(baselineSource);
    const sourceDiff=[...new Set([...Object.keys(baseFiles),...Object.keys(files)])].sort().filter(path=>baseFiles[path]!==files[path]).map(path=>({path,kind:baseFiles[path]===undefined?'added':files[path]===undefined?'deleted':'modified'}));
    return {folder,state,disposition:interrupted ? 'interrupted' : error instanceof IterationDivergedError||error instanceof InvalidImprovementState||semanticFailure?'incomplete-search':'infrastructure-failure',error:String(error),sourceDiff,baseline:null,validation:null,
      sourceManifest:{schema:'natlang.program-source/1',source:folder.digest,base:baselineSource.digest,contract:options.contract,files,baseFiles},
      authored:{identity:fingerprint(authored),files:authored},ledger:gateway.snapshot(),evaluator};
  }
  if (result.state.incumbent !== result.folder.digest) throw new Error('selected state does not match committed source');
  const measured = await evaluate(result.folder, { split: 'validation' });
  if (Math.abs(measured.quality - result.state.quality) > 1e-12) throw new Error('reported quality does not match independently executed selected source');
  const baseline = await evaluate(baselineSource, { split: 'validation' });
  const before = sourceFiles(baselineSource), files = sourceFiles(result.folder);
  const sourceDiff = [...new Set([...Object.keys(before), ...Object.keys(files)])].sort().filter(path => before[path] !== files[path])
    .map(path => ({path, kind:before[path]===undefined?'added':files[path]===undefined?'deleted':'modified'}));
  const transformation = options.transformation ? await (await import('./transformations.js')).checkTransformation(evaluator,baselineSource,result.folder,options.transformation) : undefined;
  const callGain=measured.modelCalls!==undefined&&baseline.modelCalls!==undefined&&measured.modelCalls<baseline.modelCalls;
  const objectivePassed=measured.quality>baseline.quality||!sourceDiff.length||(options.policy.objective==='model-calls'?callGain:options.policy.objective==='source-size'?measured.sourceBytes!<baseline.sourceBytes!:true);
  const disposition = measured.quality<baseline.quality||!objectivePassed||!measured.gatesPassed || transformation?.eligible===false ? 'no-eligible-promotion' : !sourceDiff.length ? 'baseline-retained' : measured.quality>baseline.quality ? 'improved' : 'transformed';
  return { ...result, disposition, sourceDiff, baseline, transformation,
    sourceManifest:{schema:'natlang.program-source/1',source:result.folder.digest,base:baselineSource.digest,contract:options.contract,files,baseFiles:before},
    authored: { identity: fingerprint(authored), files: authored }, validation: measured, ledger: gateway.snapshot(), evaluator };
}
