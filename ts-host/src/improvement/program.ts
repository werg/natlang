import { cpSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, extname, join, relative, resolve, sep } from 'node:path';
import YAML from 'yaml';
import {NATLANG_COMPILE_VERSION} from '../compiler/intrinsics.js';
import { checkProject } from '../compiler/node-project.js';
import { hexDigest } from '../native/hash.js';
import { canonicalValue } from '../native/refinement.js';
import {SYSTEM_PROMPT} from '../native/prompt.js';
import { isProtectedSource, loadNamedFunction, protectedSources, type SourceFiles } from '../runtime/loader.js';
import { adoptSource, type AdoptionRecord, type SourceManifest } from './adoption.js';
import { trustedCallOutput } from './run-evidence.js';
import { OperationJournal } from './operations.js';
import { fingerprint } from '../adaptation/identity.js';
import { Folder, FolderSnapshot } from '../native/scoped-fs.js';
import { __natlang, createNatlangRuntime, type InvocationTrace, type ModelDriver } from '../runtime/node.js';
import { loadVirtualCallables } from '../runtime/virtual-project.js';
import { AUTHORED_IMPROVER } from './authored-source.js';
import { SourceEvaluator, sourceFiles,SOURCE_EVALUATION_VERSION } from './host.js';
import { planService } from './services.js';
import { IterationLimitError,IterationDivergedError } from '../runtime/iterate.js';
import {isModelOutcomeFailure} from '../runtime/kernel.js';
import { BudgetExhausted, UsageGateway } from '../evaluation/usage.js';
import type { BudgetLimits } from '../evaluation/types.js';
import type { ImprovementCase, ProgramContract, TransformationSpec } from './types.js';
export type ImprovementPolicy = { maxExperiments: number; mode: 'instruction' | 'structural'; strategy: 'gepa' | 'adaptive'; objective?:'quality'|'source-size'|'model-calls'; goal: string; outerLesson?:string; allowedFiles: string[]; maxPopulation: number;
  /** The kind of change a run performs: a key of improveStep/transformations.ts (repairProgram, simplifyProgram, ...). */
  transformation?:string;
  /** The mode of each pluggable search policy: crisp (default), nl or shadow. */
  policies?:Partial<Record<'findOpportunity'|'chooseParent'|'selectIncumbent'|'shouldStop','crisp'|'nl'|'shadow'>> };
export type ImprovementState = { iteration: number; done: boolean; incumbent: string; quality: number; population: { source: string; quality: number; parent: string }[];
  history: { source: string; parent: string; accepted: boolean; selected: boolean; reason: string }[]; stopReason: string };
export type ImproveProgramOptions = { folder: Folder; contract: ProgramContract; cases: ImprovementCase[]; policy: ImprovementPolicy;
  improver: ModelDriver; executor: ModelDriver; executorId: string; executorTimeoutMs?:number; excludeModelWaitFromTimeout?:boolean; metadataOnlySkillFiles?:string[]; budget: BudgetLimits; signal?: AbortSignal; trace?: (trace: InvocationTrace) => void; seed?: number; directory?: string; improverSource?: FolderSnapshot; gateway?:UsageGateway; evaluationLevel?:1|2; executeCase?: import('./host.js').SourceCaseExecution; singleStep?:boolean; scoring?:import('./host.js').SourceResultScoring; transformation?:TransformationSpec;
  captureExactRewriteIO?: boolean };
class InvalidImprovementState extends Error {}
/**
 * Exact capture of the arguments and result of every natural-language stage of the authored improver (the editors, the
 * diagnose and hypothesize stages, the search policies): each `.nl` in a folder of the program. The training exporters
 * replay these invocations, and a bounded state summary cannot be replayed.
 */
function exactStageCapture(authored: Readonly<Record<string, string>>) {
  const sources = Object.keys(authored).filter(path => path.endsWith('.nl') && path.includes('/'));
  const names = new Set<string>();
  for (const path of sources) {
    const header = /^---\n([\s\S]*?)\n---/.exec(authored[path]!)?.[1] ?? '';
    const args = /^args:\n((?:[ ]{2}.*\n?)*)/m.exec(header)?.[1] ?? '';
    for (const match of args.matchAll(/^ {2}([A-Za-z_$][\w$]*)\??:/gm)) names.add(match[1]!);
  }
  return { definitionSources: sources, inputArguments: [...names], captureOutput: true, maxBytes: 8_000_000 };
}
/** The SDK entry runs an authored reducer; native services supply only source and evidence authority. */
export async function improveProgram(options: ImproveProgramOptions) {
  if (!Number.isSafeInteger(options.policy.maxExperiments) || options.policy.maxExperiments < 1 || !Number.isSafeInteger(options.policy.maxPopulation) || options.policy.maxPopulation < 2) throw new RangeError('finite experiment and population limits are required');
  if(options.executorTimeoutMs!==undefined&&(!Number.isSafeInteger(options.executorTimeoutMs)||options.executorTimeoutMs<1))throw new RangeError('positive finite executor timeout required');
  if(options.policy.objective!==undefined&&!['quality','source-size','model-calls'].includes(options.policy.objective))throw new RangeError('unknown improvement objective');
  const authored = options.improverSource ? sourceFiles(options.improverSource) : AUTHORED_IMPROVER;
  // The selected improver is frozen for this invocation; adoption only affects later invocations.
  // One experiment is crisp mechanism (improveStep/lifecycle.ts) that calls the natural-language stages. The folder
  // iteration applies it as a directory reducer: it runs on a private copy of the folder and its changes are installed
  // when it returns. The evaluator and the plan journal are host capabilities handed to it; no natural-language stage
  // can reach them.
  const lifecycle = (loadVirtualCallables(authored) as { improveStep?: { lifecycle?: { step?: (...args: unknown[]) => Promise<ImprovementState> } } }).improveStep?.lifecycle;
  if (typeof lifecycle?.step !== 'function') throw new TypeError('the authored program-improver has no improveStep/lifecycle.ts step');
  const journal = options.directory ? new OperationJournal(options.directory) : undefined;
  const identity = fingerprint({ runtime:6,compiler:NATLANG_COMPILE_VERSION,evaluation:SOURCE_EVALUATION_VERSION,authored, opening:{prompt:SYSTEM_PROMPT,tools:"native-default",contextTokens:16384,limits:{maxTurns:16,maxTokens:24000,turnTokens:2048,maxFailureRepairs:4,maxRequests:options.budget.maxModelCalls}}, seed: options.seed ?? 0, source: options.folder.snapshot().digest, contract: options.contract, cases: options.cases, policy: options.policy, transformation:options.transformation??null, executor: options.executorId, executorTimeoutMs:options.executorTimeoutMs??120000, excludeModelWaitFromTimeout:options.excludeModelWaitFromTimeout??false, metadataOnlySkillFiles:options.metadataOnlySkillFiles??null, execution:options.executeCase?.identity ?? null, scoring:options.scoring?.identity??null, evaluationLevel:options.evaluationLevel??1, budget: options.budget });
  const savedIdentity = journal?.read<string>('run-identity')?.value;
  if (savedIdentity && savedIdentity !== identity) throw new Error('resume requires the same frozen source, cases, policy, executor and allocation');
  journal?.record('run-identity', identity);
  if (options.gateway && fingerprint(options.gateway.limits) !== fingerprint(options.budget)) throw new Error('nested improvement must use the parent allocation');
  const gateway = options.gateway ?? new UsageGateway(options.budget, journal?.read<import('../evaluation/usage.js').BudgetLedger>('ledger')?.value);
  if (journal && !options.gateway) gateway.onUpdate = ledger => journal.record('ledger', ledger);
  const evaluator = new SourceEvaluator(options.contract, options.cases, options.executor, gateway, { signal: options.signal, timeoutMs:options.executorTimeoutMs, excludeModelWaitFromTimeout:options.excludeModelWaitFromTimeout, executorId: options.executorId, journal, executeCase: options.executeCase, scoring: options.scoring, evaluationLevel:options.evaluationLevel??1, sourcePolicy: { baseline: sourceFiles(options.folder.snapshot()), mode: options.policy.mode, allowedFiles: options.policy.allowedFiles, metadataOnlySkillFiles:options.metadataOnlySkillFiles } });
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
  const task = createNatlangRuntime({ model: { driver: gateway.driver(options.improver,'reflection',options.signal), maxTurns: 16, maxTokens: 24000, turnTokens: 2048, maxFailureRepairs: 4 }, signal: options.signal, seed:{mode:'derived',root:options.seed??0},codeEdits: 'deny', network: false,
    exactHostTraceCapture: options.captureExactRewriteIO ? exactStageCapture(authored) : undefined,
    trace: options.trace, onFolderProposal: () => gateway.reserve('proposals', 1, options.signal), limits: { maxEpisodes: options.budget.maxModelCalls, timeoutMs: options.budget.maxElapsedMs } });
  const stepEvaluator = { check: evaluator.check.bind(evaluator), evaluate, page: evaluator.page.bind(evaluator) };
  const plans = planService(journal);
  const step = (folder: unknown, state: ImprovementState, policy: unknown) => lifecycle.step!(folder, stepEvaluator, plans, state, policy);
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
      if(!Number.isFinite(next.state.quality)||Math.abs(measured.quality-next.state.quality)>1e-12)throw new InvalidImprovementState('reported quality '+JSON.stringify(next.state.quality)+' does not match independently executed selected source quality '+measured.quality+'; copy the host validation report quality instead of estimating it');
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
    // A budget the gateway exhausted inside a natlang call is that call's failure's cause.
    const budgetExhausted=(value:unknown,depth=0):boolean=>value instanceof BudgetExhausted||(depth<8&&value instanceof Error&&budgetExhausted(value.cause,depth+1));
    const interrupted = options.signal?.aborted || error instanceof IterationLimitError || budgetExhausted(error) || gateway.ledger.usage.modelCalls >= options.budget.maxModelCalls;
    const boundName=(value:unknown)=>(value as {name?:string}|undefined)?.name==='PolicyBoundError';
    const semanticFailure=isModelOutcomeFailure(error)||(error instanceof Error&&(isModelOutcomeFailure(error.cause)||boundName(error.cause)))||boundName(error);
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

// --- Directory-form program improvement (`natlang improve DIR`, WS8 N4) -------------------------------------------

export type DirectorySplit = 'support' | 'dev' | 'confirmation';

export type DirectoryImprovementCase = {
  id: string;
  name?: string;
  source: 'examples' | 'call-store';
  function: string;
  inputs: Record<string, unknown>;
  expected: unknown;
  group: string;
  split: DirectorySplit;
  protected?: boolean;
};

export type DirectoryCasePack = {
  directory: string;
  seed: number;
  cases: DirectoryImprovementCase[];
  splits: {
    support: DirectoryImprovementCase[];
    dev: DirectoryImprovementCase[];
    confirmation: DirectoryImprovementCase[];
  };
  sources: { examples: number; callStore: number };
};

export type DirectorySplitResult = {
  passed: number;
  total: number;
  passRate: number;
  results: { id: string; function: string; ok: boolean; expected: unknown; received?: unknown; error?: string }[];
};

export type DirectoryProposal = {
  /** Relative file path -> new file text, or `null` to delete the file. */
  files: Record<string, string | null>;
  /** Optional summary of the proposed change (e.g. memetic guidance update or instruction edit). */
  summary?: string;
};

/** Minimal structural subset of `CallStore` read by `buildDirectoryCasePack` and `improveDirectory`. */
export type DirectoryCallStoreLike = {
  calls(filter?: { program?: string; outcome?: string; limit?: number }): readonly { call_id: string }[];
  call(callId: string): {
    call_id: string;
    parent_call_id?: string | null;
    outcome: string;
    detail?: string;
    definition: { source: string | null };
    inputs: Record<string, { complete: boolean; hash?: string }>;
    output: { complete: boolean; hash?: string } | null;
  } | undefined;
  value(ref: { complete: boolean; hash?: string } | null | undefined): unknown;
  /** Read when present: escalation notes and tester feedback decide which recorded outputs are expected results. */
  events?(callId: string): Record<string, unknown>[] | undefined;
  annotations?(callId: string): { kind: string; value: unknown }[];
};

export type ImproveDirectoryOptions = {
  directory: string;
  store?: DirectoryCallStoreLike;
  seed?: number;
  maxStoreCalls?: number;
  /** Explicit proposal to verify and adopt (e.g. from `--proposal FILE` or tests). */
  proposal?: DirectoryProposal;
  /** Plugged improver callback that inspects the program files and `support` split and returns a proposal. */
  propose?: (context: {
    directory: string;
    files: Record<string, string>;
    support: DirectoryImprovementCase[];
    casePack: DirectoryCasePack;
  }) => Promise<DirectoryProposal | null> | DirectoryProposal | null;
  /** Custom static check runner (defaults to `checkProject`). */
  check?: (directory: string) => Promise<{ ok: boolean; diagnostics: string[] }>;
  /** Custom split evaluator (defaults to running each `.nl` function with `options.model`). */
  evaluateSplit?: (directory: string, cases: readonly DirectoryImprovementCase[]) => Promise<DirectorySplitResult>;
  /** Model driver used by the default split evaluator when `evaluateSplit` is not supplied. */
  model?: ModelDriver;
  /** Where to write the adoption record (defaults to `<DIR>/.natlang/adoptions/improve-<timestamp>.json`). */
  recordPath?: string;
  /** Verify without modifying `directory`. */
  dryRun?: boolean;
  /** Require dev or confirmation pass rate to strictly improve (default false: non-regression passes). */
  requireImprovement?: boolean;
};

export type ImproveDirectoryResult = {
  status: 'accepted' | 'rejected' | 'checked' | 'unchanged';
  directory: string;
  changed: string[];
  reasons: string[];
  summary?: string;
  casePack: DirectoryCasePack;
  check?: { ok: boolean; diagnostics: string[] };
  baseline?: { dev: DirectorySplitResult; confirmation: DirectorySplitResult };
  proposed?: { dev: DirectorySplitResult; confirmation: DirectorySplitResult };
  record?: AdoptionRecord;
  recordPath?: string;
};

const DIR_SKIPPED = new Set(['node_modules', '.git', '.natlang', 'dist', '.venv']);

/** Canonical key for deduplicating `[function, inputs]` across `examples.json` and `CallStore`. */
const caseDedupeKey = (fnPath: string, inputs: Record<string, unknown>): string =>
  canonicalValue([fnPath.replace(/\\/g, '/').replace(/^\.\//, ''), inputs]);

/** Read all authored source files of a program directory (excluding generated declarations and metadata dirs). */
function readProgramFiles(directory: string): Record<string, string> {
  const files: Record<string, string> = {};
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir).sort()) {
      if (DIR_SKIPPED.has(entry) || entry.startsWith('.') || entry.endsWith('.d.nl.ts') || entry.endsWith('.d.nz.ts')) continue;
      const full = join(dir, entry);
      const rel = relative(directory, full).split(sep).join('/');
      const stat = statSync(full);
      if (stat.isDirectory()) walk(full);
      else if (stat.isFile()) files[rel] = readFileSync(full, 'utf8');
    }
  };
  walk(directory);
  return files;
}

/** Protected sources of a program (`types.ts` and every `.nl` file's `protect:` list), matching `adopt-call.ts`. */
function programProtectedSources(directory: string, overrides: Record<string, string | null>): { protected: string[]; lists: Record<string, string> } {
  const found: string[] = [];
  const lists: Record<string, string> = {};
  const visit = (dir: string) => {
    for (const entry of readdirSync(dir).sort()) {
      if (DIR_SKIPPED.has(entry) || entry.startsWith('.')) continue;
      const full = join(dir, entry);
      const rel = relative(directory, full).split(sep).join('/');
      const stat = statSync(full);
      if (stat.isDirectory()) { visit(full); continue; }
      if (!entry.endsWith('.nl')) continue;
      if (Object.hasOwn(overrides, rel) && overrides[rel] === null) continue;
      const text = Object.hasOwn(overrides, rel) ? (overrides[rel] as string) : readFileSync(full, 'utf8');
      const front = /^---\r?\n([\s\S]*?)\r?\n---/.exec(text)?.[1];
      let protect: unknown;
      try { protect = front ? (YAML.parse(front) as Record<string, unknown> | null)?.protect : undefined; } catch { protect = undefined; }
      const entries = Array.isArray(protect) ? protect.filter((item): item is string => typeof item === 'string') : [];
      lists[rel] = JSON.stringify(entries);
      found.push(...protectedSources(rel, entries));
    }
  };
  visit(directory);
  return { protected: found, lists };
}

/** Paths of `directory` an improvement may not change: `types.ts` files and every `.nl` file's `protect:` items. */
export function protectedProgramPaths(directory: string): Set<string> {
  const root = resolve(directory);
  const files = Object.keys(readProgramFiles(root));
  const { protected: listed } = programProtectedSources(root, {});
  return new Set(files.filter(path => /(^|\/)types\.ts$/.test(path) || isProtectedSource(path, listed)));
}

/** Materialize a temporary copy of `directory` with `changes` applied (`node_modules` symlinked). */
function materializeDirectoryCopy(directory: string, changes: Record<string, string | null>): string {
  const target = mkdtempSync(join(tmpdir(), 'natlang-improve-dir-'));
  for (const entry of readdirSync(directory)) {
    if (entry === '.git' || entry === '.natlang') continue;
    if (entry === 'node_modules') {
      symlinkSync(join(directory, entry), join(target, entry), 'dir');
      continue;
    }
    cpSync(join(directory, entry), join(target, entry), {
      recursive: true,
      filter: source => !source.split(sep).includes('node_modules'),
    });
  }
  for (const [rel, text] of Object.entries(changes)) {
    const dest = join(target, rel);
    if (text === null) {
      rmSync(dest, { force: true });
    } else {
      mkdirSync(dirname(dest), { recursive: true });
      writeFileSync(dest, text);
    }
  }
  return target;
}

function nodeSourceFilesFor(root: string): SourceFiles {
  return {
    join,
    dirname,
    basename,
    extname,
    isFile: path => existsSync(path) && statSync(path).isFile(),
    isDirectory: path => existsSync(path) && statSync(path).isDirectory(),
    read: path => readFileSync(path, 'utf8'),
    list: dir => readdirSync(dir),
    relative: path => relative(root, path).split(sep).join('/'),
  };
}

/**
 * Build a deterministic support / dev / confirmation improvement case pack from `<DIR>/examples.json`
 * plus recorded successful calls in `CallStore` for `<DIR>` (WS8 N4).
 *
 * Deduplicates by `[function, inputs]` (with `examples.json` taking precedence), keeps each case group
 * in exactly one split, pins `protected: true` examples to `confirmation`, and orders groups by
 * `sha256(seed + ":" + groupKey)` so split assignment is deterministic and seeded.
 */
type SavedExampleLike = { name?: string; function: string; inputs: Record<string, unknown>; expected?: unknown; group?: string; protected?: boolean };

function readDirectoryExamples(root: string): SavedExampleLike[] {
  const path = join(root, 'examples.json');
  if (!existsSync(path)) return [];
  const raw = JSON.parse(readFileSync(path, 'utf8'));
  if (!Array.isArray(raw)) throw new Error(`${path} must be an array of { function, inputs, expected } examples`);
  for (const item of raw) {
    if (!item || typeof item !== 'object' || typeof item.function !== 'string' || !item.function.endsWith('.nl') ||
        !item.inputs || typeof item.inputs !== 'object' || Array.isArray(item.inputs)) {
      throw new Error(`${path}: each example needs a .nl function and an inputs object (and an expected result to be tested)`);
    }
    const delta = relative(resolve(root), resolve(root, item.function));
    if (delta === '..' || delta.startsWith('..' + sep) || resolve(root, item.function) === resolve(root)) {
      throw new Error('Example functions must be inside the program');
    }
  }
  return raw as SavedExampleLike[];
}

export function buildDirectoryCasePack(
  directory: string,
  options: { store?: DirectoryCallStoreLike; seed?: number; maxStoreCalls?: number } = {},
): DirectoryCasePack {
  const root = resolve(directory);
  const seed = options.seed ?? 0;
  if (!Number.isSafeInteger(seed)) throw new RangeError(`seed must be a safe integer, got ${seed}`);

  type RawCase = Omit<DirectoryImprovementCase, 'split'>;
  const raw: RawCase[] = [];
  const seenKeys = new Set<string>();
  let examplesCount = 0;
  let storeCount = 0;

  // 1. Saved examples from <DIR>/examples.json (only entries with expected values).
  for (const [index, example] of readDirectoryExamples(root).entries()) {
    if (!Object.hasOwn(example, 'expected')) continue;
    const fnRel = example.function.replace(/\\/g, '/').replace(/^\.\//, '');
    const key = caseDedupeKey(fnRel, example.inputs);
    if (seenKeys.has(key)) continue;
    seenKeys.add(key);
    examplesCount++;
    const group = typeof example.group === 'string' && example.group.trim() ? example.group.trim() : `case:${hexDigest(key).slice(0, 16)}`;
    raw.push({
      id: `example:${index}:${hexDigest(key).slice(0, 12)}`,
      ...(example.name ? { name: example.name } : {}),
      source: 'examples',
      function: fnRel,
      inputs: example.inputs,
      expected: example.expected,
      group,
      ...(example.protected === true ? { protected: true } : {}),
    });
  }

  // 2. Recorded successful calls in CallStore for this program directory.
  if (options.store) {
    const summaries = options.store.calls({ program: root, outcome: 'done', limit: options.maxStoreCalls ?? 500 });
    for (const summary of summaries) {
      const record = options.store.call(summary.call_id);
      if (!record || record.outcome !== 'done') continue;
      const source = record.definition.source;
      if (!source || !source.endsWith('.nl')) continue;
      // Normalize source relative to root and ensure it lives inside the program directory.
      const resolvedSource = resolve(root, source);
      const rel = relative(root, resolvedSource).split(sep).join('/');
      if (!rel || rel.startsWith('..') || !existsSync(resolvedSource)) continue;
      if (!record.output?.complete) continue;
      // The local half of an escalated decision and a call a tester marked bad are not expected results; a fixed
      // call's expected value is the tester's (run-evidence.ts trustedCallOutput).
      const trust = trustedCallOutput(options.store, record);
      if (!trust.trusted) continue;
      const expected = Object.hasOwn(trust, 'expected') ? trust.expected : options.store.value(record.output);
      if (expected === undefined) continue;
      const inputEntries = Object.entries(record.inputs);
      if (inputEntries.some(([, ref]) => !ref.complete)) continue;
      const inputs: Record<string, unknown> = {};
      let ok = true;
      for (const [param, ref] of inputEntries) {
        const value = options.store.value(ref);
        if (value === undefined) { ok = false; break; }
        inputs[param] = value;
      }
      if (!ok) continue;
      const key = caseDedupeKey(rel, inputs);
      if (seenKeys.has(key)) continue;
      seenKeys.add(key);
      storeCount++;
      raw.push({
        id: `call:${record.call_id}`,
        source: 'call-store',
        function: rel,
        inputs,
        expected,
        group: `case:${hexDigest(key).slice(0, 16)}`,
      });
    }
  }

  // Group cases so related cases stay together and never leak across splits.
  const grouped = new Map<string, { items: RawCase[]; protected: boolean }>();
  for (const item of raw) {
    let bucket = grouped.get(item.group);
    if (!bucket) { bucket = { items: [], protected: false }; grouped.set(item.group, bucket); }
    bucket.items.push(item);
    if (item.protected) bucket.protected = true;
  }

  const protectedGroups: [string, RawCase[]][] = [];
  const regularGroups: [string, RawCase[]][] = [];
  for (const [group, bucket] of grouped) {
    if (bucket.protected) protectedGroups.push([group, bucket.items]);
    else regularGroups.push([group, bucket.items]);
  }

  regularGroups.sort(([a], [b]) => hexDigest(`${seed}:${a}`).localeCompare(hexDigest(`${seed}:${b}`)));
  protectedGroups.sort(([a], [b]) => hexDigest(`${seed}:${a}`).localeCompare(hexDigest(`${seed}:${b}`)));

  const cases: DirectoryImprovementCase[] = [];
  for (const [, items] of protectedGroups) {
    for (const item of items) cases.push({ ...item, split: 'confirmation' });
  }

  const assignSplit = (index: number, total: number): DirectorySplit => {
    if (total === 1) return 'support';
    if (total === 2) return index === 0 ? 'support' : 'confirmation';
    // Guarantee non-empty support, dev, and confirmation splits whenever >= 3 groups exist (~50% support / 25% dev / 25% confirmation).
    if (index === 0) return 'support';
    if (index === 1) return 'dev';
    if (index === 2) return 'confirmation';
    const mod = index % 4;
    return mod === 2 ? 'dev' : mod === 3 ? 'confirmation' : 'support';
  };

  regularGroups.forEach(([, items], index) => {
    const split = assignSplit(index, regularGroups.length);
    for (const item of items) cases.push({ ...item, split });
  });

  return {
    directory: root,
    seed,
    cases,
    splits: {
      support: cases.filter(item => item.split === 'support'),
      dev: cases.filter(item => item.split === 'dev'),
      confirmation: cases.filter(item => item.split === 'confirmation'),
    },
    sources: { examples: examplesCount, callStore: storeCount },
  };
}

/** Default split evaluator: loads each `.nl` function in `directory` and checks exact output equality against `expected`. */
export async function evaluateDirectorySplitWithModel(
  directory: string,
  cases: readonly DirectoryImprovementCase[],
  model: ModelDriver,
): Promise<DirectorySplitResult> {
  if (!cases.length) return { passed: 0, total: 0, passRate: 1, results: [] };
  const files = nodeSourceFilesFor(directory);
  const loaded = new Map<string, { params: string[]; fn: (...args: unknown[]) => Promise<unknown> }>();
  const results: DirectorySplitResult['results'] = [];
  let passed = 0;
  for (const item of cases) {
    let entry = loaded.get(item.function);
    if (!entry) {
      const record = loadNamedFunction(join(directory, item.function), files);
      const params = Object.keys(record.args).map(name => name.replace(/\?$/, ''));
      const fn = __natlang.named(record.name, record) as unknown as (...args: unknown[]) => Promise<unknown>;
      entry = { params, fn };
      loaded.set(item.function, entry);
    }
    const args = entry.params.map(name => item.inputs[name]);
    try {
      const runtime = createNatlangRuntime({ model, calls: false, workspace: directory });
      const received = await runtime.run(() => entry!.fn(...args));
      const ok = canonicalValue(received) === canonicalValue(item.expected);
      if (ok) passed++;
      results.push({ id: item.id, function: item.function, ok, expected: item.expected, received });
    } catch (error) {
      results.push({ id: item.id, function: item.function, ok: false, expected: item.expected, error: error instanceof Error ? error.message : String(error) });
    }
  }
  return { passed, total: cases.length, passRate: cases.length ? passed / cases.length : 1, results };
}

/**
 * Directory-form program improvement (`natlang improve DIR`, WS8 N4):
 * builds the support/dev/confirmation case pack, obtains a proposal (from `options.proposal` or `options.propose`),
 * verifies it in an isolated temp copy (`natlang check` + dev & confirmation split evaluation, rejecting if
 * confirmation pass rate drops or protected files change), and installs atomically via `adoptSource`
 * (returning an `AdoptionRecord` for byte-identical `rollbackSource`).
 */
export async function improveDirectory(options: ImproveDirectoryOptions): Promise<ImproveDirectoryResult> {
  const directory = resolve(options.directory);
  const casePack = buildDirectoryCasePack(directory, { store: options.store, seed: options.seed, maxStoreCalls: options.maxStoreCalls });
  const currentFiles = readProgramFiles(directory);

  const proposal = options.proposal ?? (options.propose
    ? await options.propose({ directory, files: { ...currentFiles }, support: casePack.splits.support, casePack })
    : null);

  if (!proposal || !Object.keys(proposal.files).length) {
    return { status: 'unchanged', directory, changed: [], reasons: ['no source changes proposed'], casePack };
  }

  // Validate proposed paths stay inside `directory` and compute actual changes.
  const normalizedChanges: Record<string, string | null> = {};
  const reasons: string[] = [];
  for (const [rawPath, content] of Object.entries(proposal.files)) {
    const clean = rawPath.replace(/\\/g, '/').replace(/^\.\//, '');
    const target = resolve(directory, clean);
    const rel = relative(directory, target).split(sep).join('/');
    if (!rel || rel.startsWith('..') || rel !== clean) {
      reasons.push(`${rawPath} is outside ${directory}`);
      continue;
    }
    normalizedChanges[clean] = content;
  }

  const changed = Object.keys(normalizedChanges).filter(path => {
    const next = normalizedChanges[path];
    const prev = Object.hasOwn(currentFiles, path) ? currentFiles[path] : null;
    return next !== prev;
  }).sort();

  if (reasons.length) {
    return { status: 'rejected', directory, changed, reasons, summary: proposal.summary, casePack };
  }
  if (!changed.length) {
    return { status: 'unchanged', directory, changed: [], reasons: ['proposal leaves every file unchanged'], summary: proposal.summary, casePack };
  }

  // Reject edits to protected sources (`types.ts` and `.nl` `protect:` lists).
  const beforeProtected = programProtectedSources(directory, {});
  const afterProtected = programProtectedSources(directory, normalizedChanges);
  for (const path of changed) {
    if (/(^|\/)types\.ts$/.test(path)) {
      reasons.push(`${path} declares the types results are checked against and cannot be modified by improvement`);
    } else if (isProtectedSource(path, beforeProtected.protected) || isProtectedSource(path, afterProtected.protected)) {
      reasons.push(`${path} is protected (protect:) and cannot be modified by improvement`);
    } else if (beforeProtected.lists[path] !== afterProtected.lists[path]) {
      reasons.push(`${path}: improvement cannot change a function's protect: list`);
    }
  }
  if (reasons.length) {
    return { status: 'rejected', directory, changed, reasons, summary: proposal.summary, casePack };
  }

  const runCheck = options.check ?? (async (dir: string) => {
    const built = checkProject(dir, { write: false, writeDeclarations: false });
    return {
      ok: built.ok,
      diagnostics: built.diagnostics.filter(item => item.severity === 'error').map(item => `${item.file}:${item.line} ${item.message}`),
    };
  });

  const runSplit = options.evaluateSplit ?? (async (dir: string, cases: readonly DirectoryImprovementCase[]) => {
    if (!cases.length) return { passed: 0, total: 0, passRate: 1, results: [] };
    if (!options.model) throw new Error('evaluating examples requires a model driver or an evaluateSplit callback');
    return evaluateDirectorySplitWithModel(dir, cases, options.model);
  });

  const emptySplit = (): DirectorySplitResult => ({ passed: 0, total: 0, passRate: 1, results: [] });

  // Evaluate baseline on dev and confirmation splits before applying changes.
  const baselineDev = casePack.splits.dev.length ? await runSplit(directory, casePack.splits.dev) : emptySplit();
  const baselineConfirmation = casePack.splits.confirmation.length ? await runSplit(directory, casePack.splits.confirmation) : emptySplit();
  const baseline = { dev: baselineDev, confirmation: baselineConfirmation };

  // Build SourceManifest over the changed files (`adoptSource` handles create, modify, and delete).
  const baseFiles: Record<string, string> = {};
  const afterFiles: Record<string, string> = {};
  for (const path of changed) {
    if (Object.hasOwn(currentFiles, path)) baseFiles[path] = currentFiles[path]!;
    const next = normalizedChanges[path];
    if (typeof next === 'string') afterFiles[path] = next;
  }
  const manifest: SourceManifest = {
    schema: 'natlang.program-source/1',
    base: Folder.fromFiles(baseFiles).snapshot().digest,
    source: Folder.fromFiles(afterFiles).snapshot().digest,
    baseFiles,
    files: afterFiles,
    contract: { kind: 'directory-improvement', directory, seed: casePack.seed, ...(proposal.summary ? { summary: proposal.summary } : {}) },
  };

  let checkReport: { ok: boolean; diagnostics: string[] } | undefined;
  let proposedSplits: { dev: DirectorySplitResult; confirmation: DirectorySplitResult } | undefined;

  const verifyProposal = async () => {
    const tempDir = materializeDirectoryCopy(directory, normalizedChanges);
    try {
      checkReport = await runCheck(tempDir);
      if (!checkReport.ok) {
        return { source: manifest.source, valid: false, diagnostics: ['natlang check failed', ...checkReport.diagnostics], contract: manifest.contract as never };
      }
      const dev = casePack.splits.dev.length ? await runSplit(tempDir, casePack.splits.dev) : emptySplit();
      const confirmation = casePack.splits.confirmation.length ? await runSplit(tempDir, casePack.splits.confirmation) : emptySplit();
      proposedSplits = { dev, confirmation };
      const diagnostics: string[] = [];
      if (casePack.splits.dev.length && dev.passRate < baselineDev.passRate) {
        diagnostics.push(`dev pass rate dropped from ${(baselineDev.passRate * 100).toFixed(1)}% (${baselineDev.passed}/${baselineDev.total}) to ${(dev.passRate * 100).toFixed(1)}% (${dev.passed}/${dev.total})`);
      }
      if (casePack.splits.confirmation.length && confirmation.passRate < baselineConfirmation.passRate) {
        diagnostics.push(`confirmation pass rate dropped from ${(baselineConfirmation.passRate * 100).toFixed(1)}% (${baselineConfirmation.passed}/${baselineConfirmation.total}) to ${(confirmation.passRate * 100).toFixed(1)}% (${confirmation.passed}/${confirmation.total})`);
      }
      if (options.requireImprovement && dev.passRate <= baselineDev.passRate && confirmation.passRate <= baselineConfirmation.passRate) {
        diagnostics.push('proposal did not improve dev or confirmation pass rate');
      }
      return { source: manifest.source, valid: diagnostics.length === 0, diagnostics, contract: manifest.contract as never };
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  };

  if (options.dryRun) {
    const checked = await verifyProposal();
    return {
      status: checked.valid ? 'checked' : 'rejected',
      directory,
      changed,
      reasons: checked.diagnostics,
      summary: proposal.summary,
      casePack,
      check: checkReport,
      baseline,
      proposed: proposedSplits,
    };
  }

  const recordPath = resolve(options.recordPath ?? join(directory, '.natlang', 'adoptions', `improve-${Date.now()}-${manifest.source.slice(0, 8)}.json`));
  try {
    const record = await adoptSource(directory, manifest, verifyProposal, recordPath);
    return {
      status: 'accepted',
      directory,
      changed,
      reasons: [],
      summary: proposal.summary,
      casePack,
      check: checkReport,
      baseline,
      proposed: proposedSplits,
      record,
      recordPath,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (!message.startsWith('adoption build failed')) throw error;
    return {
      status: 'rejected',
      directory,
      changed,
      reasons: message.replace(/^adoption build failed:\s*/, '').split('\n').filter(Boolean),
      summary: proposal.summary,
      casePack,
      check: checkReport,
      baseline,
      proposed: proposedSplits,
    };
  }
}
