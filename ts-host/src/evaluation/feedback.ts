import type { EvaluationBatch, PreparedSuite } from './types.js';
import { cloneData } from '../adaptation/identity.js';
export class EvaluationFeedbackError extends Error {}
/** Feedback projection is explicit; gold cases from validation/test are never sent to reflection. */
export function trainingFeedback(prepared: PreparedSuite, batch: EvaluationBatch, keys: readonly string[]): unknown[] {
  return batch.results.filter(result => result.split === 'train').map(result => {
    const testCase = prepared.cases.find(testCase => testCase.id === result.caseId)!;
    let custom;
    try {
      const value = prepared.suite.feedback?.(testCase, result);
      custom = value === undefined ? undefined : cloneData(value);
    } catch (error) { throw new EvaluationFeedbackError('evaluation feedback projection must return finite JSON: ' + String(error)); }
    return { caseId: result.caseId, quality: result.quality, gates: result.gates, feedback: result.feedback,
      outcome: result.outcome.kind, ...(custom === undefined ? {} : { projected: custom }),
      invocations: result.traces.filter(trace => keys.includes(String(trace.adaptation?.component))).map(trace =>
        ({ callId: trace.callId, component: trace.adaptation?.component, outcome: trace.outcome, detail: trace.detail })) };
  });
}
