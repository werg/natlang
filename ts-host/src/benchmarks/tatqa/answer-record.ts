import { decimalForms } from '../../oracle-kit/decimal.js';
import { jsonStringRecordCanonical } from '../../oracle-kit/json-record.js';
import { registerRecordComparator } from '../../oracle-kit/oracle-registry.js';

const TATQA_SCALES = new Set(['', 'percent', 'thousand', 'million', 'billion']);

/** TaTQA-only comparison: strict record schema and scale, with source scorer's numeric rounding. */
export function tatqaAnswerRecordCanonical(value: unknown): string | null {
  const recordText = jsonStringRecordCanonical(value);
  if (recordText === null || typeof value !== 'string') return null;
  try {
    const record = JSON.parse(value) as Record<string, unknown>;
    if (Object.keys(record).length !== 2 || !Object.hasOwn(record, 'answer') || !Object.hasOwn(record, 'scale') ||
        typeof record.answer !== 'string' || typeof record.scale !== 'string' || !TATQA_SCALES.has(record.scale)) return null;
    const numeric = decimalForms(record.answer);
    return JSON.stringify({ answer: numeric ? { numeric: true, exact: numeric.exact, rounded: numeric.rounded } :
      { numeric: false, text: record.answer },
      scale: record.scale });
  } catch { return null; }
}

export function tatqaAnswerRecordsEqual(actual: unknown, expected: unknown, numericComparison: 'rounded-2dp' | 'exact' = 'rounded-2dp'): boolean {
  if (typeof actual !== 'string' || typeof expected !== 'string') return false;
  const left = tatqaAnswerRecordCanonical(actual), right = tatqaAnswerRecordCanonical(expected);
  if (left === null || right === null) return false;
  if (left === right) return true;
  try {
    const a = JSON.parse(left) as { answer: { numeric: boolean; exact?: string; rounded?: string | null; text?: string }; scale: string };
    const b = JSON.parse(right) as { answer: { numeric: boolean; exact?: string; rounded?: string | null; text?: string }; scale: string };
    if (a.scale !== b.scale || !a.answer.numeric || !b.answer.numeric) return false;
    if (numericComparison === 'exact') return a.answer.exact === b.answer.exact;
    return a.answer.exact === b.answer.exact || (!!a.answer.rounded && !!b.answer.rounded &&
      a.answer.rounded === b.answer.rounded);
  } catch { return false; }
}

registerRecordComparator('tatqa-answer-record', { answerFile: 'answer.json',
  equal: (actual, expected) => tatqaAnswerRecordsEqual(actual, expected) });
registerRecordComparator('tatqa-answer-record-exact', { answerFile: 'answer.json',
  equal: (actual, expected) => tatqaAnswerRecordsEqual(actual, expected, 'exact') });
