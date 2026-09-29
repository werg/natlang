import type { Candidate, ExecutorIdentity } from '../adaptation/types.js';
import type { EvaluationBatch } from '../evaluation/types.js';
import type { BudgetLedger } from '../evaluation/usage.js';
import { pairedChanges } from '../evaluation/report.js';
type Report = { runId: string; strategy: string; stopReason?: string; baseline: EvaluationBatch;
  selectedValidation: EvaluationBatch; lockedTest: EvaluationBatch | null; uncovered: readonly string[];
  ledger: BudgetLedger; executor: ExecutorIdentity; history: readonly Record<string, unknown>[];
  historyTruncated?: number;
  promotion?: string; holdoutStatus?: string;
  instructions: { baseline: Candidate; selected: Candidate } };
const cell = (value: unknown) => String(value ?? 'unknown').replace(/\|/g, '\\|').replace(/\r?\n/g, ' ');
const block = (value: unknown) => {
  const text = JSON.stringify(value, null, 2);
  const width = Math.max(3, ...[...text.matchAll(/`+/g)].map(match => match[0].length + 1));
  const fence = '`'.repeat(width); return fence + 'json\n' + text + '\n' + fence + '\n';
};
/** Deterministic evidence rendering; no model is asked to summarize the run. */
export function optimizationMarkdown(report: Report): string {
  let text = '# Adaptation run ' + cell(report.runId) + '\n\n' +
    'Strategy: ' + report.strategy + '. Stop: ' + report.stopReason + '. Executor: ' + cell(report.executor.id) + '.\n\n' +
    'Promotion: ' + cell(report.promotion) + '. Locked test: ' + cell(report.holdoutStatus) + '.\n\n' +
    '| Evaluation | Quality | Cases | Gates |\n| --- | ---: | ---: | --- |\n';
  for (const [label, batch] of [['Baseline validation', report.baseline], ['Selected validation', report.selectedValidation], ['Locked test', report.lockedTest]] as const)
    text += '| ' + label + ' | ' + cell(batch?.quality) + ' | ' + cell(batch?.results.length) + ' | ' + (batch ? batch.gatesPassed ? 'pass' : 'fail' : 'not completed') + ' |\n';
  text += '\nValidation selects the incumbent. Locked test results do not feed proposals or selection.\n\n' +
    '## Paired validation changes\n\n| Case | Baseline | Selected | Change | Outcome |\n| --- | ---: | ---: | ---: | --- |\n';
  for (const row of pairedChanges(report.baseline, report.selectedValidation)) text += '| ' +
    [row.caseId, row.baseline, row.selected, row.delta, row.baselineOutcome + ' → ' + row.selectedOutcome].map(cell).join(' | ') + ' |\n';
  text += '\n## Usage and coverage\n\nUnknown token or cost values indicate unavailable provider evidence.\n\n' + block(report.ledger) +
    '\nUncovered selected components: ' + (report.uncovered.length ? report.uncovered.map(cell).join(', ') : 'none') + '.\n\n' +
    '## Search history\n\n' + (report.historyTruncated ? `${report.historyTruncated} older events are retained in the run event log.\n\n` : '') +
    block(report.history) + '\n## Original and selected instructions\n\n' + block(report.instructions);
  return text;
}
