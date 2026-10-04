import {TOOLS_PROMPT} from '../native/prompt.js';
import { OperationJournal } from './operations.js';
import { NATLANG_COMPILE_VERSION } from '../compiler/intrinsics.js';
import { validateSourceEdit, exportedSignature } from './source-policy.js';
import { Worker } from 'node:worker_threads';
import { isDeepStrictEqual } from 'node:util';
import ts from 'typescript';
import { FolderSnapshot } from '../native/scoped-fs.js';
import { compileVirtualProject,loadVirtualNatlang } from '../runtime/virtual-project.js';
import {callableMeta} from '../runtime/callable.js';
import {parseType,formatType} from '../native/types.js';
import {inlineDeclaredTypes} from '../native/eval-types.js';
import {declarationNamespace} from '../native/external.js';
import * as runtime from '../runtime/node.js';
import type { ModelDriver } from '../runtime/runtime.js';
import { checkConstrainedSource } from '../compiler/policy.js';
import { fingerprint, immutable } from '../adaptation/identity.js';
import { validateCases } from '../evaluation/suite.js';
import { UsageGateway } from '../evaluation/usage.js';
import type { CheckReport, EvaluationReport, ImprovementCase, Outcome, ProgramContract, ProgramEvaluator } from './types.js';
import { loadSkills, memorySkillSource } from '../skills/registry.js';
import { checkSkillMetadataOnlyEdit } from '../skills/edit-policy.js';
import type { SkillUseEvent } from '../skills/observability.js';

export const SOURCE_EVALUATION_VERSION='source-evaluation/24';
/** These records carry diagnostics, not observations of target answer quality. */
export function hasUnscoredEvaluationFailure(report: {outcomes?: Outcome[]}): boolean {
  return report.outcomes?.some(row => row.failureKind === 'fixture' || row.failureKind === 'timeout') ?? false;
}
export function sourceFiles(snapshot: FolderSnapshot): Record<string, string> {
  if (!(snapshot instanceof FolderSnapshot)) throw new TypeError('evaluate requires an immutable folder snapshot');
  return Object.fromEntries(snapshot.filePaths().map(path => [path, new TextDecoder('utf-8', { fatal: true }).decode(snapshot.readBytesSync(path))]));
}
/** Keep volatile host telemetry out of authored feedback without changing the raw stored outcome. */
function authoredSkillUse(event: SkillUseEvent): SkillUseEvent {
  return Object.fromEntries(Object.entries(event).filter(([key]) => !['observed_at', 'invocation_id', 'elapsed_ms', 'seq', 'version'].includes(key))) as SkillUseEvent;
}
function authoredOutcome(outcome: Outcome): Outcome {
  return {
    ...outcome,
    ...(outcome.skillUseTrace ? { skillUseTrace: outcome.skillUseTrace.map(authoredSkillUse) } : {}),
    ...(outcome.modelTrace ? { modelTrace: outcome.modelTrace.map(entry => ({ ...entry,
      ...(entry.skillUse ? { skillUse: entry.skillUse.map(authoredSkillUse) } : {}) })) } : {}),
  };
}
/** Compiler and finite case execution only. Experiment selection belongs in authored source. */
export type SourceCaseResult = { files?:Record<string,string>; value?: unknown; modelCalls?:number; modelTrace?:{calls:unknown;observation:string;skillUse?:SkillUseEvent[]}[]; skillUseTrace?:SkillUseEvent[]; modelTraceTruncated?:boolean; failureKind?:'fixture'|'target'|'timeout'; error?: string; score?: {quality:number;gates:Record<string,boolean>} };
export type SourceCaseExecution = ((folder: FolderSnapshot, row: ImprovementCase, seed: number, gateway: UsageGateway) => Promise<SourceCaseResult>) & {readonly identity:string;readonly evaluationLevel:1|2};
/** Host-owned scoring of actual outputs. Never supplied by the target or editor. */
export type SourceResultScoring = { readonly identity: string;
  score(row: ImprovementCase, result: SourceCaseResult): {quality:number;gates:Record<string,boolean>} };
export class SourceEvaluator implements ProgramEvaluator {
  private readonly evidence = new Map<string, Outcome[]>();
  private readonly executions = new Map<string,Promise<SourceCaseResult>>();
  private readonly checks = new Map<string, CheckReport>();
  readonly suiteVersion: string;
  private confirmed = false;
  constructor(readonly contract: ProgramContract, private cases: ImprovementCase[], readonly driver: ModelDriver,
    readonly gateway: UsageGateway, readonly options: { signal?: AbortSignal; timeoutMs?: number; excludeModelWaitFromTimeout?: boolean; maxCasesPerRequest?: number; executorId: string; evaluationLevel?:1|2; executeCase?: SourceCaseExecution; scoring?: SourceResultScoring; journal?: OperationJournal; sourcePolicy?: { baseline: Record<string, string>; mode: 'instruction' | 'structural'; allowedFiles: string[]; metadataOnlySkillFiles?: string[] } } ) {
    if ((options.executeCase?.evaluationLevel ?? 1) > (options.evaluationLevel ?? 1)) throw new Error('evaluation level cannot be increased by a target');
    if (options.executeCase && !options.executeCase.identity) throw new Error('independent execution requires a frozen identity');
    if (options.scoring && !options.scoring.identity) throw new Error('independent scoring requires a frozen identity');
    // Compile-only checks need no cases. Evaluation itself still requires a nonempty named split.
    if(cases.length)validateCases(cases.map(row => ({ ...row, input: row.args })));
    this.cases = structuredClone(cases);
    this.suiteVersion = fingerprint({ api:SOURCE_EVALUATION_VERSION,cases, contract, executor: options.executorId, execution:options.executeCase?.identity ?? null, scoring:options.scoring?.identity??null, evaluationLevel:options.evaluationLevel??1, compiler: NATLANG_COMPILE_VERSION, opening:{prompt:TOOLS_PROMPT,tools:"native-default",contextTokens:16384}, policy: { network: false, codeEdits: 'deny', maxEpisodes: 30, maxActions: 100, timeoutMs: options.timeoutMs ?? 120000, excludeModelWaitFromTimeout: options.excludeModelWaitFromTimeout ?? false } });
  }
  async check(folder: FolderSnapshot): Promise<CheckReport> {
    const found = this.checks.get(folder.digest); if (found) return found;
    const files = sourceFiles(folder), diagnostics: string[] = this.options.sourcePolicy ? validateSourceEdit(this.options.sourcePolicy.baseline, files, this.options.sourcePolicy.mode, this.options.sourcePolicy.allowedFiles) : [];
    if (this.options.sourcePolicy?.metadataOnlySkillFiles) diagnostics.push(...checkSkillMetadataOnlyEdit(this.options.sourcePolicy.baseline, files, this.options.sourcePolicy.metadataOnlySkillFiles));
    // Data-only skill files must be valid before an experiment may measure or accept them.
    const skillRoots=new Set(Object.keys(files).filter(path=>path.endsWith('/SKILL.md')).flatMap(path=>{
      if(path.startsWith('skills/'))return ['skills'];
      const at=path.lastIndexOf('/skills/');return at<0?[]:[path.slice(0,at+7)];
    }));
    for(const root of skillRoots){
      const set=await loadSkills(memorySkillSource(files),{root});
      diagnostics.push(...set.diagnostics.filter(item=>item.severity==='error').map(item=>`${item.path}: ${item.code}: ${item.message}`));
    }
    for (const [path, source] of Object.entries(files).filter(([path]) => /\.m?ts$/.test(path) && !path.endsWith('.d.ts'))) {
      diagnostics.push(...checkConstrainedSource(ts.createSourceFile(path, source, ts.ScriptTarget.ES2022, true)).map(item => `${path}:${item.line}:${item.column}: ${item.message}`));
      const file = ts.createSourceFile(path, source, ts.ScriptTarget.ES2022, true);
      for (const node of file.statements) if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) && node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier)) {
        const name = node.moduleSpecifier.text;
        if (!name.startsWith('./') && !name.startsWith('../') && !['@natlang/node','@natlang/browser'].includes(name)) diagnostics.push('target import is outside the declared checked core: ' + name);
        if (['@natlang/node','@natlang/browser'].includes(name) && ts.isImportDeclaration(node) && node.importClause?.namedBindings && ts.isNamedImports(node.importClause.namedBindings)) {
          const forbidden = node.importClause.namedBindings.elements.filter(item => ['createNatlangRuntime','SourceEvaluator','improveProgram','suiteExecution','FrozenImprover','OperationJournal','AssignmentBudget'].includes(item.propertyName?.text ?? item.name.text));
          if (forbidden.length) diagnostics.push('target cannot import program-evaluation authority');
        }
      }
    }
    const build = compileVirtualProject({ files }, runtime, { programId: this.contract.programId, constrained: true, target: 'node' });
    diagnostics.push(...build.diagnostics.filter(item => item.severity === 'error').map(item => `${item.file}:${item.line}: ${item.message}`));
    if (!files[this.contract.entry]) diagnostics.push('external entry was removed');
    // Check declarations against the pinned external signature without executing user code.
    if (this.contract.signature) {
      const entry = files[this.contract.entry] ?? '';
      const expected = exportedSignature('export function ' + this.contract.signature + ' {}', this.contract.exportName);
      if (!expected || exportedSignature(entry, this.contract.exportName) !== expected) diagnostics.push('pinned external signature changed');
    }
    const baseline = this.options.sourcePolicy?.baseline[this.contract.entry];
    if (baseline && /\.m?ts$/.test(this.contract.entry)) {
      const pinned = exportedSignature(baseline, this.contract.exportName);
      if (pinned && exportedSignature(files[this.contract.entry] ?? '', this.contract.exportName) !== pinned) diagnostics.push('external application function contract changed');
    }
    if (baseline && this.contract.entry.endsWith('.nl')) {
      const signature=(sources:Record<string,string>)=>{
        const definition=callableMeta(loadVirtualNatlang(sources,this.contract.entry))!.definition;
        const types=Object.fromEntries(Object.entries(definition.types??{}).map(([name,type])=>[name,parseType(type)]));
        const normalize=(type:string)=>formatType(inlineDeclaredTypes(parseType(type),types));
        return JSON.stringify({kind:definition.subtype,params:definition.params.map(param=>({name:param.name,optional:!!param.optional,type:normalize(param.type)})),returns:normalize(definition.returns)});
      };
      try {if(signature(this.options.sourcePolicy!.baseline)!==signature(files))diagnostics.push('external natlang function contract changed');}
      catch(error){diagnostics.push('external natlang contract could not be checked: '+String(error));}
    }
    const report = immutable({ source: folder.digest, valid: diagnostics.length === 0, diagnostics, contract: this.contract });
    this.checks.set(folder.digest, report); return report;
  }
  evaluate(folder: FolderSnapshot, request: { split: 'train' | 'validation'; caseIds?: string[]; seed?: number }): Promise<EvaluationReport> {
    if (!['train', 'validation'].includes(request.split)) throw new Error('locked test cases are unavailable to the improver');
    return this.evaluateSplit(folder, request);
  }
  /** One final confirmation after development freezes; deliberately absent from ProgramEvaluator. */
  async confirm(folder: FolderSnapshot, freeze: { source: string; experiment: string }): Promise<EvaluationReport> {
    if (this.confirmed || freeze.source !== folder.digest || !freeze.experiment) throw new Error('confirmation must match the frozen source and can run only once');
    const identity = { ...freeze, suite: this.suiteVersion };
    const previous = this.options.journal?.read<typeof identity>('confirmation-freeze');
    if (previous && fingerprint(previous.value) !== fingerprint(identity)) throw new Error('confirmation already frozen for a different candidate or experiment');
    this.options.journal?.record('confirmation-freeze', identity);
    this.confirmed = true;
    // A crash resumes this exact frozen experiment; it cannot allocate a fresh test attempt.
    return this.options.journal ? this.options.journal.run('confirmation-result', () => this.evaluateSplit(folder, { split: 'test' }), () => this.evaluateSplit(folder, { split: 'test' })) : this.evaluateSplit(folder, { split: 'test' });
  }
  /** Predeclared paired confirmation. Both sources freeze before any test evidence is exposed. */
  async confirmQuality(baseline: FolderSnapshot, selected: FolderSnapshot, experiment: string) {
    if (!experiment || this.confirmed) throw new Error('invalid or repeated paired quality comparison');
    const freeze = { baseline: baseline.digest, source: selected.digest, suite: this.suiteVersion,
      experiment, metric: 'paired-independent-quality' };
    const previous = this.options.journal?.read<typeof freeze>('confirmation-freeze');
    if (previous && fingerprint(previous.value) !== fingerprint(freeze))
      throw new Error('confirmation already frozen for a different experiment');
    this.options.journal?.record('confirmation-freeze', freeze); this.confirmed = true;
    const execute = async () => {
      const before = await this.evaluateSplit(baseline, {split:'test'});
      if (hasUnscoredEvaluationFailure(before)) throw new Error('unscored paired baseline: fixture failure or resource timeout; no quality gain can be claimed');
      const after = await this.evaluateSplit(selected, {split:'test'});
      if (hasUnscoredEvaluationFailure(after)) throw new Error('unscored paired selected source: fixture failure or resource timeout; no quality gain can be claimed');
      const differences = before.outcomes!.map(left => {
        const right = after.outcomes!.find(row => row.caseId === left.caseId);
        if (!right) throw new Error('paired query case identity changed');
        return { caseId: left.caseId, before: left.quality, after: right.quality,
          difference: right.quality-left.quality };
      });
      return immutable({ freeze, baseline: before, selected: after, differences,
        wins: differences.filter(row => row.difference > 0).length,
        losses: differences.filter(row => row.difference < 0).length,
        ties: differences.filter(row => row.difference === 0).length,
        effect: after.quality-before.quality });
    };
    return this.options.journal ? this.options.journal.run('confirmation-result', execute, execute) : execute();
  }

  /** Binary sign-test comparison retained for callers that explicitly request it. */
  async confirmPair(baseline: FolderSnapshot, selected: FolderSnapshot, experiment: string, alpha = 0.05) {
    if (!Number.isFinite(alpha) || alpha <= 0 || alpha >= 1 || !experiment || this.confirmed) throw new Error('invalid or repeated paired confirmation');
    const freeze = { baseline: baseline.digest, source: selected.digest, suite: this.suiteVersion, experiment, alpha, metric: 'paired-binary-success' };
    const previous = this.options.journal?.read<typeof freeze>('confirmation-freeze');
    if (previous && fingerprint(previous.value) !== fingerprint(freeze)) throw new Error('confirmation already frozen for a different experiment');
    this.options.journal?.record('confirmation-freeze', freeze); this.confirmed = true;
    const execute = async () => {
      const before = await this.evaluateSplit(baseline, {split:'test'}), after = await this.evaluateSplit(selected, {split:'test'});
      if (hasUnscoredEvaluationFailure(before) || hasUnscoredEvaluationFailure(after))
        throw new Error('unscored paired confirmation: fixture failure or resource timeout');
      if ([...before.outcomes!, ...after.outcomes!].some(row => row.quality !== 0 && row.quality !== 1)) throw new Error('paired sign confirmation requires the declared binary success metric');
      const pairs = before.outcomes!.map(left => ({left, right: after.outcomes!.find(row => row.caseId === left.caseId)!}));
      const wins = pairs.filter(pair => !pair.left.passed && pair.right.passed).length;
      const losses = pairs.filter(pair => pair.left.passed && !pair.right.passed).length;
      const discordant = wins + losses;
      // Exact one-sided paired sign test: a single frozen comparison, no significance retries.
      let mass = Math.pow(0.5, discordant), p = 0;
      for (let k = 0; k <= discordant; k++) { if (k >= wins) p += mass; mass *= (discordant - k) / (k + 1); }
      return immutable({freeze, baseline:before, selected:after, wins, losses, effect:after.quality-before.quality,
        pValue:Math.min(1,p), supported:after.gatesPassed && after.quality>before.quality && p<=alpha});
    };
    return this.options.journal ? this.options.journal.run('confirmation-result', execute, execute) : execute();
  }
  page(reference: string, start = 0, limit = 20): Outcome[] {
    if (!Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(limit) || limit < 1 || limit > 100) throw new RangeError('invalid evidence page');
    const rows = this.evidence.get(reference); if (!rows) throw new Error('Training evidence reference unavailable. Only a training report evidence reference can be paged. Validation exposes report.scores for selection; its outputs cannot be read. Evaluate split train to obtain diagnostic examples.');
    return structuredClone(rows.slice(start, start + limit).map(authoredOutcome));
  }
  private async evaluateSplit(folder: FolderSnapshot, request: { split: 'train' | 'validation' | 'test'; caseIds?: string[]; seed?: number }): Promise<EvaluationReport> {
    const check = await this.check(folder);
    const cases = this.cases.filter(row => row.split === request.split && (!request.caseIds || request.caseIds.includes(row.id)));
    if (!cases.length || cases.length > (this.options.maxCasesPerRequest ?? 100) || request.caseIds?.some(id => !cases.some(row => row.id === id))) throw new Error('missing, foreign, or excessive case IDs');
    const reference = fingerprint({ source: folder.digest, suite: this.suiteVersion, split: request.split, ids: cases.map(row => row.id), seed: request.seed ?? 0 });
    if (!check.valid) {
      const outcomes=cases.map(row=>({caseId:row.id,passed:false,quality:0,gates:{compiles:false},modelCalls:0,error:'Compilation failed: '+check.diagnostics.join('\n'),evidence:reference+':'+row.id,...(request.split==='train'?{args:row.args,expected:row.expected,...(row.expectedFiles?{expectedFiles:row.expectedFiles}:{})}:{})}));
      if(request.split==='train')this.evidence.set(reference,outcomes);
      return immutable({source:folder.digest,split:request.split,suiteVersion:this.suiteVersion,quality:0,modelCalls:0,gatesPassed:false,passed:0,total:cases.length,evidence:reference,sourceBytes:folder.filePaths().reduce((sum,path)=>sum+folder.readBytesSync(path).length,0),
        ...(request.split==='validation'?{scores:cases.map(row=>({caseId:row.id,quality:0}))}:{outcomes})});
    }
    const outcomes: Outcome[] = [];
    for (const row of cases) {
      // Give the editor the same public service surface the target executes.
      // Implementations and held-out cases never enter training feedback.
      const serviceDeclarations = request.split === 'train' && row.services ? Object.fromEntries(Object.entries(row.services).map(([name, source]) =>
        [name, declarationNamespace(name, ts.transpileDeclaration(source, {fileName:name+'.ts', compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText)])) : undefined;
      const execute = async () => {
        this.gateway.reserve('rollouts', 1, this.options.signal);
        return this.options.executeCase ? this.options.executeCase(folder, row, request.seed ?? 0, this.gateway) : this.runCase(folder, row, request.seed ?? 0);
      };
      const key=reference+':'+row.id;
      let execution=this.executions.get(key);
      if(!execution){execution=this.options.journal?this.options.journal.run(key,execute,execute):execute();this.executions.set(key,execution);}
      let result:SourceCaseResult;try{result=await execution;}catch(error){this.executions.delete(key);throw error;}
      if(result.modelCalls!==undefined&&(!Number.isSafeInteger(result.modelCalls)||result.modelCalls<0))throw new Error('invalid independent request count');
      const exact = !result.error && isDeepStrictEqual(result.value, row.expected) && (!row.expectedFiles || isDeepStrictEqual(result.files,row.expectedFiles));
      const scored = this.options.scoring?.score(row, result) ?? result.score;
      const quality = scored?.quality ?? (exact ? 1 : 0), gates = scored?.gates ?? {compiles:true,completed:!result.error,requiredCorrect:!row.required||exact};
      if (!Number.isFinite(quality) || quality < 0 || quality > 1 || Object.values(gates).some(value => typeof value !== 'boolean')) throw new Error('invalid independent fixture score');
      outcomes.push({ caseId: row.id, passed:quality===1 && Object.values(gates).every(Boolean), quality, gates, ...(serviceDeclarations?{serviceDeclarations}:{}), ...(result.modelCalls!==undefined?{modelCalls:result.modelCalls}:{}), ...(result.skillUseTrace?{skillUseTrace:result.skillUseTrace}:{}), ...(request.split==='train'&&result.modelTrace?{modelTrace:result.modelTrace,modelTraceTruncated:result.modelTraceTruncated??false}:{}), ...(result.value !== undefined ? {value:result.value} : {}), ...(result.error !== undefined ? {error:result.error,...(result.failureKind?{failureKind:result.failureKind}:{})} : {}), evidence: reference + ':' + row.id,...(request.split==='train'?{args:row.args,expected:row.expected,...(row.expectedFiles?{expectedFiles:row.expectedFiles,...(result.files?{files:result.files}:{})}:{})}:{}) });
    }
    // Only training evidence can expose case outputs through the injected reader.
    if (request.split === 'train') this.evidence.set(reference, outcomes);
    const passed = outcomes.filter(row => row.passed).length;
    return immutable({ source: folder.digest, split: request.split, suiteVersion: this.suiteVersion, quality: outcomes.reduce((sum,row)=>sum+row.quality,0) / outcomes.length, ...(outcomes.every(row=>row.modelCalls!==undefined)?{modelCalls:outcomes.reduce((sum,row)=>sum+row.modelCalls!,0)}:{}), gatesPassed:outcomes.every(row=>Object.values(row.gates).every(Boolean)), passed, total: outcomes.length, evidence: reference,sourceBytes:folder.filePaths().reduce((sum,path)=>sum+folder.readBytesSync(path).length,0),
      ...(request.split === 'train' || request.split === 'test' ? { outcomes: request.split === 'train' ? outcomes.map(authoredOutcome) : outcomes } : request.split === 'validation' ? { scores: outcomes.map(row => ({ caseId: row.caseId, quality: row.quality })) } : {}) });
  }
  private runCase(folder: FolderSnapshot, row: ImprovementCase, seed: number): Promise<SourceCaseResult> {
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL('./source-worker.js', import.meta.url), { execArgv: process.execArgv.filter(arg => !arg.startsWith('--input-type')), workerData: { files: sourceFiles(folder), contract: this.contract, args: row.args, services: row.services, folder: row.folder, seed,
        limits: { maxEpisodes: 30, maxActions: 100, ...(this.options.excludeModelWaitFromTimeout ? {} : {timeoutMs: this.options.timeoutMs ?? 120000}) } } });
      const lifetime = new AbortController();
      const signal = this.options.signal ? AbortSignal.any([lifetime.signal, this.options.signal]) : lifetime.signal;
      let closed = false,modelCalls=0;
      // This explicit collection policy counts local execution, excluding queued/inference time.
      let timer: ReturnType<typeof setTimeout> | undefined;
      let remainingMs = this.options.timeoutMs ?? 120000, runningSince = performance.now(), pendingModels = 0;
      const pauseTimer = () => {
        if (!this.options.excludeModelWaitFromTimeout || pendingModels++ !== 0) return;
        remainingMs -= performance.now() - runningSince;
        clearTimeout(timer);
      };
      const resumeTimer = () => {
        if (!this.options.excludeModelWaitFromTimeout || --pendingModels !== 0 || closed) return;
        runningSince = performance.now();
        timer = setTimeout(timedOut, Math.max(0, remainingMs));
      };
      const modelTrace:{calls:unknown;observation:string;skillUse?:SkillUseEvent[]}[]=[];
      const skillUseTrace:SkillUseEvent[]=[];
      const pendingTrace=new Map<string,{entry:{calls:unknown;observation:string;skillUse?:SkillUseEvent[]}}> ();
      const close = (error?: Error, result?: SourceCaseResult) => {
        if (closed) return; closed = true; clearTimeout(timer); this.options.signal?.removeEventListener('abort', abort); lifetime.abort(); void worker.terminate();
        error ? reject(error) : resolve(result!);
      };
      const abort = () => close(new Error('source evaluation cancelled'));
      const timedOut = () => close(undefined,{error:'source evaluation timed out',failureKind:'timeout',modelCalls,skillUseTrace,...(row.split==='train'?{modelTrace:modelTrace.slice(-6),modelTraceTruncated:modelTrace.length>6}:{})});
      timer = setTimeout(timedOut, remainingMs);
      this.options.signal?.addEventListener('abort', abort, { once: true });
      if (this.options.signal?.aborted) { abort(); return; }
      worker.on('message', async message => {
        if (message.type === 'result') close(undefined, { modelCalls,skillUseTrace,...(row.split==='train'?{modelTrace:modelTrace.slice(-6),modelTraceTruncated:modelTrace.length>6}:{}), ...(message.error ? { error: message.error,failureKind:message.failureKind } : { value: message.value,files:message.files }) });
        else if(message.type==='trace'){
          const events=message.events as Record<string,unknown>[];
          const skillEvents=events.filter(event=>event.kind==='skill_use') as SkillUseEvent[];skillUseTrace.push(...skillEvents);
          const pending=pendingTrace.get(message.callId);
          if(pending){pending.entry.skillUse=skillEvents;let last=-1;for(let index=events.length-1;index>=0;index--)if(events[index]!.kind==='model_request'&&events[index]!.phase==='end'&&events[index]!.call_id===message.callId){last=index;break;}if(last>=0){const actions=events.slice(last+1).filter(event=>event.kind==='action'&&event.call_id===message.callId);pending.entry.observation=actions.map(event=>String(event.name)+' ('+String(event.outcome)+'): '+String(event.result_text??'')).join('\n').slice(0,1000);}}
        }
        else if (message.type === 'request') {
          const pending=pendingTrace.get(message.request.invocation_id);
          if(pending){const messages=message.request.messages;let last=-1;for(let index=messages.length-1;index>=0;index--)if(messages[index].role==='assistant'&&messages[index].tool_calls?.length){last=index;break;}if(last>=0)pending.entry.observation=messages.slice(last+1).filter((item:{role:string})=>item.role==='tool').map((item:{content:unknown})=>String(item.content)).join('\n').slice(0,1000);}
          modelCalls++; pauseTimer();
          try { const turn = await this.gateway.request(this.driver, {...message.request,invocation_id:'case:'+fingerprint({source:folder.digest,caseId:row.id,seed})+'/'+message.request.invocation_id}, signal, 'executor');
            if(row.split==='train'){const entry={calls:(turn.calls??[]).map(([name,args])=>({name,arguments:Object.fromEntries(Object.entries(args).map(([key,value])=>[key,typeof value==='string'?value.slice(0,1500):value])),truncated:Object.values(args).some(value=>typeof value==='string'&&value.length>1500)})),observation:''};modelTrace.push(entry);pendingTrace.set(message.request.invocation_id,{entry});}
            if (!closed) worker.postMessage({ id: message.id, turn }); }
          catch (error) { close(error instanceof Error ? error : new Error(String(error))); }
          finally { resumeTimer(); }
        }
      });
      worker.on('error', error => close(error));
      worker.on('exit', code => { if (!closed) close(new Error('source worker exited without result: ' + code)); });
    });
  }
}
