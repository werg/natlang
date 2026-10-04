/** Crisp skill authoring: search on support, freeze, then host-only paired query evaluation. */
import { Folder } from '../native/scoped-fs.js';
import { fingerprint } from '../adaptation/identity.js';
import { UsageGateway } from '../evaluation/usage.js';
import type { BudgetLimits } from '../evaluation/types.js';
import type { ModelDriver } from '../runtime/runtime.js';
import type { InvocationTrace } from '../runtime/node.js';
import { validateEpisode, evaluationTicket, type SkillEpisode, type EpisodeCase } from '../skills/episode.js';
import { loadSkills, memorySkillSource } from '../skills/registry.js';
import { buildSkillAblations, summarizeSkillAblation } from '../skills/ablation.js';
import { AUTHORED_IMPROVER } from './authored-source.js';
import { improveProgram } from './program.js';
import { SourceEvaluator, sourceFiles, type SourceResultScoring } from './host.js';
import { OperationJournal } from './operations.js';
import type { ImprovementCase, ProgramContract } from './types.js';

const safePath = (path: string) => path.length > 0 && !path.startsWith('/') && !path.includes('\\') &&
  !path.split('/').some(part => !part || part === '.' || part === '..');

/** Functions bind skills from their companion context, not the project root. */
export function skillRoot(entry: string): string {
  if (!safePath(entry) || !entry.endsWith('.nl')) throw new Error('skill episodes currently require a .nl target entry');
  return entry.slice(0, -3) + '/skills';
}

export function skillEpisodeFiles(episode: SkillEpisode): Record<string, string> {
  const diagnostics = validateEpisode(episode);
  if (diagnostics.length) throw new Error('invalid skill episode: ' + JSON.stringify(diagnostics));
  const files = { ...episode.target.files }, root = skillRoot(episode.target.entry);
  for (const path of Object.keys(files)) if (!safePath(path)) throw new Error('unsafe target path: ' + path);
  // The episode library is authoritative; hidden target skills would invalidate its baseline.
  for (const path of Object.keys(files)) if (path.startsWith(root + '/'))
    throw new Error('target already contains skills; put its starting skills in episode.library');
  for (const [name, contents] of Object.entries(episode.library.skills)) {
    if (!/^[a-z][a-z0-9-]*$/.test(name)) throw new Error('unsafe skill name: ' + name);
    for (const [path, text] of Object.entries(contents)) {
      if (!safePath(path) || typeof text !== 'string') throw new Error('unsafe skill file: ' + path);
      files[`${root}/${name}/${path}`] = text;
    }
  }
  return files;
}

/** Selection feedback is drawn only from support groups. Sealed cases never enter improveProgram. */
export function supportSearchCases(episode: SkillEpisode): ImprovementCase[] {
  const groups = [...new Set(episode.support.cases.map(row => row.group))].sort();
  if (groups.length < 2) throw new Error('support search requires at least two disjoint source groups');
  const validation = new Set(groups.slice(Math.max(1, Math.floor(groups.length / 2))));
  return episode.support.cases.map(row => {
    if (row.expected === undefined) throw new Error('support execution requires an expected return: ' + row.id);
    return { ...row, args: row.args ?? [], expected: row.expected,
      split: validation.has(row.group) ? 'validation' : 'train' };
  });
}

/** The authored library must be the only skill difference between the paired transfer arms. */
export function checkTransferTarget(episode: SkillEpisode): void {
  if (!episode.transfer) return;
  const target = episode.transfer.target, root = skillRoot(target.entry);
  for (const path of Object.keys(target.files)) {
    if (!safePath(path)) throw new Error('unsafe transfer path: ' + path);
    if (path.startsWith(root + '/')) throw new Error('transfer target already contains skills');
  }
}

function contractFor(episode: SkillEpisode, transfer = false): ProgramContract {
  const target = transfer ? episode.transfer!.target : episode.target;
  return { entry: target.entry, exportName: target.exportName ?? 'default', programId: target.source.id };
}

function sealedCases(rows: EpisodeCase[]): ImprovementCase[] {
  return rows.map(row => {
    if (row.expected === undefined) throw new Error('sealed execution requires an expected return: ' + row.id);
    return { ...row, args: row.args ?? [], expected: row.expected, split: 'test' };
  });
}

export type SkillAuthoringOptions = {
  episode: SkillEpisode;
  author: ModelDriver;
  executor: ModelDriver;
  executorId: string;
  searchBudget: BudgetLimits;
  evaluationBudget: BudgetLimits;
  directory: string;
  signal?: AbortSignal;
  maxExperiments?: number;
  /** Host-only, after authoring freezes; never feeds query evidence back into the author. */
  maxAblations?: number;
  scoring?: SourceResultScoring;
  transferScoring?: SourceResultScoring;
  scoringDescriptor?: Record<string, unknown>;
  trace?: (trace: InvocationTrace) => void;
  onSearch?: (artifact: unknown) => Promise<void>;
  /** Fixed data paths that the author may create; other target files stay immutable. */
  newSkillNames?: string[];
};

export async function authorSkillEpisode(options: SkillAuthoringOptions) {
  const { episode } = options;
  const maxAblations = options.maxAblations ?? 0;
  if (!Number.isSafeInteger(maxAblations) || maxAblations < 0 || maxAblations > 12) throw Error('maxAblations must be 0..12');
  const baselineFiles = skillEpisodeFiles(episode), root = skillRoot(episode.target.entry);
  checkTransferTarget(episode);
  const support = supportSearchCases(episode);
  const metadataOnly = episode.provenance.selection_design === 'metadata-tuning';
  const names = metadataOnly ? [] : options.newSkillNames ?? ['task-procedure', 'evidence-review', 'exact-bookkeeping'];
  if (names.some(name => !/^[a-z][a-z0-9-]*$/.test(name))) throw new Error('invalid proposed skill name');
  const allowedFiles = [...new Set([
    ...Object.keys(baselineFiles).filter(path => path.startsWith(root + '/') && (!metadataOnly || path.endsWith('/SKILL.md'))),
    ...names.flatMap(name => ['SKILL.md', 'references/procedure.md', 'examples/support.md']
      .map(path => `${root}/${name}/${path}`)),
  ])];
  const authored = { ...AUTHORED_IMPROVER };
  authored['improveStep/rewriteProgram.nl'] += `\nThis experiment edits reusable CRISP SKILLS only. The target executable files are frozen.\nCalls awaiting evaluator or model work may wait for queued inference; omit timeout_ms for these async calls. Finite model-call and local-execution resource budgets are host-owned.\nCreate, revise, select or retire allowed SKILL.md and reference/example data files under ${root}.\nUse standard YAML frontmatter with name and description, followed by useful general instructions.\nTargets see bound skill descriptions and can read_code("skills.<name>") for their bodies.\nTune descriptions and optional summaries as applicability signals, as well as instructions and examples.\nThe goal is to use exactly the necessary and helpful skills for each task: neither miss useful skills nor load or apply irrelevant or redundant ones.\nA description-only improvement is valid. Explain when a skill applies and when it does not; keep its description consistent with its body.\nEvaluate on support evidence whether descriptions lead to appropriate discovery and use. Do not minimize skill count at the expense of task quality, and do not force every bound skill to be read.\nDiagnose support execution evidence and teach a reusable procedure, rather than copying instance answers.\nPreserve exact computational checks. Existing data and code outside allowedFiles cannot change.\nBefore evaluator.check or evaluator.evaluate, pass an immutable snapshot: \`const candidate = folder.snapshot(); await evaluator.evaluate(candidate, {split: 'train'});\`. Never pass the mutable folder itself.\nThe remaining-turn indicator counts model turns, not files. Batch independent allowed-file writes in one response or one eval, and reserve a final turn for bookkeeping.finish. If the work cannot fit, stop before making partial edits and report blocked honestly.\n`;
  if (metadataOnly) authored['improveStep/rewriteProgram.nl'] += '\nThis is a metadata-only experiment: change only description and optional summary in existing SKILL.md files. Preserve all names, other metadata, instructions and supporting files. Do not create or retire skills.\n';
  const metadataOnlySkillFiles = metadataOnly ? allowedFiles : undefined;
  const baseline = Folder.fromFiles(baselineFiles).snapshot();
  const policy = { maxExperiments: options.maxExperiments ?? Math.min(3, episode.limits.maxSteps),
    mode: 'structural' as const, strategy: 'gepa' as const, objective: 'quality' as const, maxPopulation: 4,
    allowedFiles, goal: 'Improve independent support execution and appropriate skill use by tuning skill descriptions, summaries and reusable procedures. Keep target code frozen.' };
  const searchDefinition = { version: 'natlang.improvement-case/1', id: episode.id, family: episode.family,
    files: baselineFiles, contract: contractFor(episode), cases: support, policy,
    budget: options.searchBudget, sourceGroups: [...new Set(support.map(row => row.group))],
    scoringIdentity: options.scoring?.identity ?? null, scoringDescriptor: options.scoringDescriptor ?? {kind:'exact-return-and-files'},
    executorId: options.executorId, metadataOnlySkillFiles: metadataOnlySkillFiles ?? null, seed: 0, authoredFiles: authored, authoredDigest: Folder.fromFiles(authored).snapshot().digest };
  const traces: InvocationTrace[] = [];
  const searched = await improveProgram({ folder: baseline.branch(), contract: contractFor(episode), cases: support, policy,
    improverSource: Folder.fromFiles(authored).snapshot(), improver: options.author, executor: options.executor,
    executorId: options.executorId, budget: options.searchBudget, signal: options.signal,
    metadataOnlySkillFiles, scoring: options.scoring, trace: trace => { traces.push(trace); options.trace?.(trace); },
    captureExactRewriteIO: true,
    excludeModelWaitFromTimeout: true, seed: 0, directory: options.directory + '/search' });
  // An interrupted search is not an outcome: surface it so collectors resume instead of recording "incomplete".
  if (options.signal?.aborted) throw options.signal.reason ?? new Error('skill authoring interrupted');
  const selected = searched.folder;
  const selectedFiles = sourceFiles(selected);
  const skills = await loadSkills(memorySkillSource(selectedFiles), { root });
  const skillErrors = skills.diagnostics.filter(item => item.severity === 'error');
  const finalIdentity = fingerprint({ episode, baseline: baseline.digest, selected: selected.digest,
    executor: options.executorId, evaluationBudget: options.evaluationBudget, scoring: options.scoring?.identity ?? null, transferScoring: options.transferScoring?.identity ?? null, maxAblations:options.maxAblations??0, seed: 0 });
  const resultBase = { version: 'natlang.skill-authoring-trajectory/1', episode: episode.id, split: episode.split,
    family: episode.family, source_groups: episode.source_groups, license: episode.license,
    evaluation_ticket: evaluationTicket(episode), baseline: baseline.digest, selected: selected.digest,
    search: searched, searchDefinition, traces, selectedFiles, skillDiagnostics: skills.diagnostics, identity: finalIdentity };
  await options.onSearch?.(resultBase);
  if (!searched.validation || skillErrors.length || !searched.state.done)
    return { ...resultBase, disposition: 'incomplete' as const, query: null, transfer: null, positive: false };
  // A retained baseline or rejected search cannot enter SFT. Do not spend sealed
  // executor calls comparing identical sources or confirming a candidate that was never promoted.
  const eligibleSearch = searched.state.done && searched.validation.gatesPassed &&
    searched.disposition !== 'no-eligible-promotion' && selected.digest !== baseline.digest &&
    searched.state.history.some(item => item.source === selected.digest && item.accepted);
  if (!eligibleSearch)
    return {...resultBase,disposition:'not-promoted' as const,query:null,transfer:null,positive:false,ablations:[]};
  // This journal and its case table are never installed in the author's runtime.
  const journal = new OperationJournal(options.directory + '/sealed-query');
  const saved = journal.read<string>('identity')?.value;
  if (saved && saved !== finalIdentity) throw new Error('sealed evaluation resume identity changed');
  journal.record('identity', finalIdentity);
  const gateway = new UsageGateway(options.evaluationBudget,
    journal.read<import('../evaluation/usage.js').BudgetLedger>('ledger')?.value);
  gateway.onUpdate = ledger => journal.record('ledger', ledger);
  const evaluator = new SourceEvaluator(contractFor(episode), sealedCases(episode.query.cases), options.executor,
    gateway, { executorId: options.executorId, signal: options.signal, scoring: options.scoring, excludeModelWaitFromTimeout: true, journal });
  const query = await evaluator.confirmQuality(baseline, selected, episode.id);
  let transfer = null;
  if (episode.transfer) {
    const target = episode.transfer.target, transferRoot = skillRoot(target.entry);
    const before = { ...target.files }, after = { ...target.files };
    for (const [path, text] of Object.entries(baselineFiles)) if (path.startsWith(root + '/'))
      before[transferRoot + path.slice(root.length)] = text;
    for (const [path, text] of Object.entries(selectedFiles)) if (path.startsWith(root + '/'))
      after[transferRoot + path.slice(root.length)] = text;
    const transferJournal = new OperationJournal(options.directory + '/sealed-transfer');
    const transferEvaluator = new SourceEvaluator(contractFor(episode, true), sealedCases(episode.transfer.cases),
      options.executor, gateway, { executorId: options.executorId, signal: options.signal, scoring: options.transferScoring ?? options.scoring, excludeModelWaitFromTimeout: true, journal: transferJournal });
    transfer = await transferEvaluator.confirmQuality(Folder.fromFiles(before).snapshot(), Folder.fromFiles(after).snapshot(), episode.id + ':transfer');
  }
  const ablations: unknown[] = [];
  if (maxAblations) {
    const variants = buildSkillAblations(baselineFiles, selectedFiles, root);
    // Prefer semantic description/body contrasts before removing a whole skill.
    const candidates = [...variants.candidates].sort((a,b) =>
      Number(a.kind === 'leave_one_skill_out') - Number(b.kind === 'leave_one_skill_out')).slice(0,maxAblations);
    for (const [index,candidate] of candidates.entries()) {
      const ablationJournal = new OperationJournal(options.directory + '/sealed-ablation-' + index);
      try {
        const paired = await new SourceEvaluator(contractFor(episode), sealedCases(episode.query.cases), options.executor,
          gateway, {executorId:options.executorId,signal:options.signal,scoring:options.scoring,
            excludeModelWaitFromTimeout:true,journal:ablationJournal}).confirmQuality(selected,
              Folder.fromFiles(candidate.files).snapshot(), episode.id + ':ablation:' + index);
        const before = new Map(paired.baseline.outcomes?.map(row => [row.caseId,row]));
        const outcomePairs = paired.selected.outcomes?.map(row => ({baselinePassed:before.get(row.caseId)?.passed,ablatedPassed:row.passed})) ?? [];
        const events = paired.baseline.outcomes?.flatMap(row => row.skillUseTrace ?? []) ?? [];
        ablations.push({kind:candidate.kind,skill:candidate.skillName,source:paired.selected.source,
          quality_effect_on_removal_or_restore:paired.effect,paired,
          observations:summarizeSkillAblation(candidate.skillName,outcomePairs,events)});
      } catch (error) {
        // Attribution is ancillary: keep the already completed paired query evidence intact.
        ablations.push({kind:candidate.kind,skill:candidate.skillName,disposition:'incomplete',error:String(error)});
        break;
      }
    }
  }
  // Report raw paired gains; statistical significance is not invented as a fixed programme gate.
  const positive = query.selected.gatesPassed && query.effect > 0 &&
    (!transfer || (transfer.selected.gatesPassed && transfer.effect >= 0));
  return { ...resultBase, disposition: 'evaluated' as const, query, transfer, positive, ablations,
    evaluationUsage: gateway.ledger, selectedFiles,
    admission: 'Candidate only: publication requires trace, causal visibility and source policy checks.' };
}
