/** One place that turns an episode metric into a host-only scorer, for collection and offline replay alike. */
import { evaluateCodeObjective, type CodeObjective } from './code-objective.js';
import { resolve } from 'node:path';
import { OBJECTIVE_KINDS, scoreSkillObjective, type ObjectiveKind } from './objective.js';
import { EFFICIENCY_OBJECTIVE_KINDS } from './efficiency-objective.js';
import { EXTENDED_OBJECTIVE_KINDS } from './extended-objective.js';
import { scoreContractNliObjective } from './contractnli-objective.js';
import { scoreScifactObjective } from './scifact-objective.js';
import { scoreResearchObjective } from './research-objective.js';
import { scoreCrosswordObjective } from './crossword-objective.js';
import { scoreCspProgress } from './csp-objective.js';
import { scoreSpecifiedTranslation, type TranslationTask } from './translation-objective.js';
import { scoreStaticPage } from './visual-objective.js';
import { GRADED_KINDS, scoreGraded, type GradedKind } from './graded.js';

export type EpisodeMetric = { schema: string; kind: string };
export type EpisodeScoring = {
  identity: string;
  score: (row: { args: unknown[]; expected: unknown }, output: { value?: unknown; error?: unknown }) => { quality: number; gates: Record<string, boolean> };
};

/** `pins` are the code hashes recorded with the collection; they make scorer identity part of the episode result. */
export function episodeScoring(metric: EpisodeMetric | undefined, context: { pins: Record<string, string>; databaseRoot?: string }): EpisodeScoring | undefined {
  if (!metric) return undefined;
  const failed = { quality: 0, gates: { completed: false } };
  if (metric.schema === 'natlang.skill-objective/1' && (OBJECTIVE_KINDS as readonly string[]).includes(metric.kind)) {
    if (!context.pins['skills/objective.js']) throw Error('objective code pin required');
    const kind = metric.kind as ObjectiveKind;
    const extensionFile = (EXTENDED_OBJECTIVE_KINDS as readonly string[]).includes(kind) ? 'skills/extended-objective.js' : (EFFICIENCY_OBJECTIVE_KINDS as readonly string[]).includes(kind) ? 'skills/efficiency-objective.js' : null;
    const extension = extensionFile ? context.pins[extensionFile] : null;
    if (extensionFile && !extension) throw Error('extended objective code pin required');
    return { identity: `natlang.skill-objective/1:${kind}:${context.pins['skills/objective.js']}${extension ? ':'+extension : ''}`,
      score: (row, output) => output.error ? failed : scoreSkillObjective(kind, row.args[0], output.value, row.expected) };
  }
  if (metric.schema === 'natlang.skill-code-objective/1' && metric.kind === 'python-source-bytes') {
    const codePin = context.pins['skills/code-objective.js'], sandboxPin = context.pins['skills/graded.js'];
    if (!codePin || !sandboxPin) throw Error('code objective and sandbox code pins required');
    return {identity:`natlang.skill-code-objective/1:${metric.kind}:${codePin}:${sandboxPin}`,
      score: (row, output) => {
        if (output.error) return failed;
        if (typeof output.value !== 'string') return {quality:0,gates:{source_string:false}};
        const fenced = /```(?:python|py)?\s*([\s\S]*?)```/i.exec(output.value);
        const source = (fenced ? fenced[1]! : output.value).trim();
        const expected = row.expected as CodeObjective;
        const task = {...expected, ...(expected?.sizeObjective ? {sizeObjective:{...expected.sizeObjective,referenceSource:expected.sizeObjective.referenceSource.trim()}} : {})};
        const result = evaluateCodeObjective(source, task);
        if (result.status === 'infrastructure-error' || result.status === 'invalid-task')
          throw Error('code objective cannot be scored: '+JSON.stringify(result.detail));
        return {quality:result.quality!,gates:result.gates};
      }};
  }
  const exact = metric.schema === 'natlang.skill-contractnli/1' && metric.kind === 'contract-nli-classification' ? {file:'skills/contractnli-objective.js',score:(row:{args:unknown[];expected:unknown},value:unknown)=>scoreContractNliObjective(row.args[0],value,row.expected)}
    : metric.schema === 'natlang.skill-scifact/1' && metric.kind === 'scifact-claim-evidence' ? {file:'skills/scifact-objective.js',score:(row:{args:unknown[];expected:unknown},value:unknown)=>scoreScifactObjective(row.args[0],value,row.expected)}
    : metric.schema === 'natlang.skill-research/1' && metric.kind === 'research-classification' ? {file:'skills/research-objective.js',score:(row:{args:unknown[];expected:unknown},value:unknown)=>scoreResearchObjective(row.args[0],value,row.expected)}
    : metric.schema === 'natlang.crossword-csp/1' && metric.kind === 'clue-constraints' ? {file:'skills/crossword-objective.js', score:(row: {args:unknown[];expected:unknown}, value:unknown)=>scoreCrosswordObjective(row.args[0],value,row.expected)}
    : metric.schema === 'natlang.skill-csp/1' && metric.kind === 'csp-progress' ? {file:'skills/csp-objective.js',score:(row: {args:unknown[];expected:unknown},value:unknown)=>scoreCspProgress(row.args[0],value,row.expected)}
    : metric.schema === 'natlang.skill-translation/1' && metric.kind === 'specified-expression' ? {file:'skills/translation-objective.js',score:(row: {args:unknown[];expected:unknown},value:unknown)=>scoreSpecifiedTranslation(row.expected as TranslationTask,value)}
    : metric.schema === 'natlang.static-page-objective/1' && metric.kind === 'static-responsive-html-pilot' ? {file:'skills/visual-objective.js',score:(row: {args:unknown[];expected:unknown},value:unknown)=>scoreStaticPage(value,row.expected)} : undefined;
  if(exact){
    const pin=context.pins[exact.file];if(!pin)throw Error('exact objective code pin required: '+exact.file);
    return {identity:`${metric.schema}:${metric.kind}:${pin}`,score:(row,output)=>output.error?failed:exact.score(row,output.value)};
  }
  if (metric.schema === 'natlang.skill-graded/1' && (GRADED_KINDS as readonly string[]).includes(metric.kind)) {
    if (metric.kind === 'python-tests') throw new Error('python-tests collection is held pending private oracle isolation and unscored infrastructure failures; use the isolated code-objective contract');
    if (!context.pins['skills/graded.js']) throw Error('graded objective code pin required');
    if (metric.kind === 'sql-result-f1' && !context.databaseRoot) throw new Error('graded SQL episodes require a database root');
    const graded = { schema: 'natlang.skill-graded/1' as const, kind: metric.kind as GradedKind,
      ...(context.databaseRoot ? { database_root: resolve(context.databaseRoot) } : {}) };
    return { identity: `natlang.skill-graded/1:${graded.kind}:${context.pins['skills/graded.js']}`,
      score: (row, output) => output.error ? failed : scoreGraded(graded, output.value, row.expected) };
  }
  throw new Error('unsupported episode metric: ' + JSON.stringify(metric));
}

/** Both arms of an episode need scorers when either is metric-scored. */
export function episodeScorings(provenance: { metric?: EpisodeMetric; transfer_metric?: EpisodeMetric } | undefined, hasTransfer: boolean,
  context: { pins: Record<string, string>; databaseRoot?: string }) {
  const metric = provenance?.metric, transferMetric = provenance?.transfer_metric;
  if (metric && hasTransfer && !transferMetric) throw new Error('transfer requires its own metric');
  return { metric, scoring: episodeScoring(metric, context), transferScoring: episodeScoring(transferMetric, context) };
}
