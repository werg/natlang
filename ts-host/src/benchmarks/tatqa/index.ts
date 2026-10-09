import { registerRuntimeFailureRule, registerSourceContract, type RuntimeFailureRule } from '../registry.js';
import { tatqaAnswerRecordCanonical, tatqaAnswerRecordsEqual } from './answer-record.js';
import { isReviewedTatqaLakhVariant, TATQA_LAKH_SOURCE_ID } from './unit-contract.js';

export * from './answer-record.js';
export * from './unit-contract.js';

/** Rejected TaTQA outcomes that predate the current answer-record oracle or display contract are not negatives. */
export const tatqaRuntimeFailureRule: RuntimeFailureRule = (row, record) => {
  const oracle = record.semantics.oracle;
  if (typeof oracle !== 'object' ||
      !['json-string-record', 'tatqa-answer-record', 'tatqa-answer-record-exact'].includes(oracle.normalization ?? ''))
    return 'obsolete_json_format_oracle';
  const numericExpected = oracle.normalization === 'json-string-record' &&
    JSON.parse(tatqaAnswerRecordCanonical(record.semantics.expected) ?? '{}').answer?.numeric === true;
  if (numericExpected && tatqaAnswerRecordsEqual(row.outcome?.value, record.semantics.expected) &&
      tatqaAnswerRecordsEqual((row.outcome?.files as Record<string, string> | undefined)?.['answer.json'], row.outcome?.value))
    return 'legacy_tatqa_numeric_display_oracle';
  // Older snapshots omit the reviewed numeric/unit display instructions. A
  // mismatch there is not reliable evidence for a preference-training negative.
  if (numericExpected &&
      (record.generation as Record<string, unknown> | undefined)?.numeric_answer_contract_revision !== 'tatqa-numeric-answer-v1')
    return 'legacy_tatqa_numeric_contract';
};

export function registerTatqa(): void {
  registerSourceContract({ dataset: 'tatqa', sourceId: TATQA_LAKH_SOURCE_ID, isReviewedVariant: isReviewedTatqaLakhVariant });
  registerRuntimeFailureRule('tatqa', tatqaRuntimeFailureRule);
}
