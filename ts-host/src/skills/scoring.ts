/** One place that turns an episode metric into a host-only scorer, for collection and offline replay alike. */
import { resolve } from 'node:path';
import { OBJECTIVE_KINDS, scoreSkillObjective, type ObjectiveKind } from './objective.js';
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
    const kind = metric.kind as ObjectiveKind;
    return { identity: `natlang.skill-objective/1:${kind}:${context.pins['skills/objective.js']}`,
      score: (row, output) => output.error ? failed : scoreSkillObjective(kind, row.args[0], output.value, row.expected) };
  }
  if (metric.schema === 'natlang.skill-graded/1' && (GRADED_KINDS as readonly string[]).includes(metric.kind)) {
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
