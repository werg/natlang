/** `natlang check` output for fused pipelines: the plan's edges with reasons, and counts (plans/FUSED_PIPELINES.md). */
import type { EdgeFacts, FusionFacts } from './facts.js';
import type { FusionPlan } from './plan.js';
import type { FusionSettings } from './settings.js';

export type FusionReportEdge = { id: string; scope: string; chain: string; type: string; flow: EdgeFacts['flow']; decision: 'fuse' | 'keep-text';
  reason: string; readers: { kind: string; detail: string; certain: boolean }[]; engaged: boolean };
export type FusionReport = {
  mode: FusionSettings['mode']; planner: FusionSettings['planner']; plannedBy: FusionPlan['planner'];
  counts: { edges: number; fuse: number; keepText: number; scopes: number; nlScopeFuse: number; typescriptScopeFuse: number };
  edges: FusionReportEdge[];
  scopes: FusionFacts['scopes']; skipped: FusionFacts['skipped'];
};

export function fusionReport(facts: FusionFacts, plan: FusionPlan, settings: FusionSettings): FusionReport {
  const decisions = new Map(plan.edges.map(entry => [entry.edge, entry]));
  const edges = facts.edges.map(edge => {
    const entry = decisions.get(edge.id);
    const decision = entry?.decision ?? 'keep-text';
    return { id: edge.id, scope: edge.scope, chain: edge.chain, type: edge.type, flow: edge.flow, decision,
      reason: entry?.reason ?? 'not decided', readers: edge.readers.map(reader => ({ kind: reader.kind, detail: reader.detail, certain: reader.certain })),
      engaged: decision === 'fuse' && edge.scopeKind === 'nl' };
  });
  const fuse = edges.filter(edge => edge.decision === 'fuse');
  const kindOf = new Map(facts.edges.map(edge => [edge.id, edge.scopeKind]));
  return { mode: settings.mode, planner: settings.planner, plannedBy: plan.planner,
    counts: { edges: edges.length, fuse: fuse.length, keepText: edges.length - fuse.length, scopes: facts.scopes.length,
      nlScopeFuse: fuse.filter(edge => kindOf.get(edge.id) === 'nl').length, typescriptScopeFuse: fuse.filter(edge => kindOf.get(edge.id) === 'typescript').length },
    edges, scopes: facts.scopes, skipped: facts.skipped };
}

/** Text lines for `natlang check`: a summary, and with `all` every edge. */
export function formatFusionReport(report: FusionReport, all: boolean): string {
  const lines = [`fusion: ${report.counts.edges} candidate hand-off${report.counts.edges === 1 ? '' : 's'} in ${report.counts.scopes} scope${report.counts.scopes === 1 ? '' : 's'}, ` +
    `${report.counts.fuse} fusible (${report.counts.nlScopeFuse} in natural-language scopes, ${report.counts.typescriptScopeFuse} in TypeScript), ` +
    `mode ${report.mode}, planner ${report.planner}`];
  for (const edge of report.edges) {
    if (!all && edge.decision !== 'fuse') continue;
    lines.push(`  ${edge.decision === 'fuse' ? 'fuse     ' : 'keep-text'} ${edge.chain} (${edge.type}) in ${edge.scope}: ${edge.reason}`);
  }
  return lines.join('\n');
}
