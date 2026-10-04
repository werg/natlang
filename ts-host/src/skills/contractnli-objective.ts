/** Host-only ContractNLI score over one supplied contract and its fixed hypothesis set. */
export const CONTRACTNLI_OBJECTIVE_KIND = 'contract-nli-classification' as const;
export type ContractNliExpected = { kind: typeof CONTRACTNLI_OBJECTIVE_KIND; scope: 'single-supplied-contract';
  hypotheses: Record<string, { choice: 'Entailment' | 'Contradiction' | 'NotMentioned'; evidence_alternatives: string[][] }>;
  available_span_ids: string[] };
export type ContractNliScore = { quality: number; gates: Record<string, boolean>;
  detail: { hypotheses: number; labels_correct: number; evidence_sufficient: number; invalid_span_ids: number } };

const object = (x: unknown): x is Record<string, unknown> => !!x && typeof x === 'object' && !Array.isArray(x);
const idString = (value: unknown): string | null => (typeof value === 'string' || typeof value === 'number') && String(value).trim() ? String(value).trim() : null;
const choice = (value: unknown): 'Entailment' | 'Contradiction' | 'NotMentioned' | null => {
  if (typeof value !== 'string') return null;
  const normalized = value.toLowerCase().replace(/[\s_-]+/gu, '');
  return normalized === 'entailment' ? 'Entailment' : normalized === 'contradiction' ? 'Contradiction'
    : normalized === 'notmentioned' ? 'NotMentioned' : null;
};
function parse(value: unknown): unknown {
  if (typeof value !== 'string') return value;
  const fenced = /^\s*```(?:json)?\s*([\s\S]*?)```\s*$/u.exec(value);
  return JSON.parse(fenced ? fenced[1]! : value);
}

/** Classifies all 17 hypothesis IDs; positive-class evidence must include a complete approved span alternative. */
export function scoreContractNliObjective(packetValue: unknown, responseValue: unknown, expectedValue: unknown): ContractNliScore {
  let packet: unknown, response: unknown;
  try { packet = parse(packetValue); } catch { packet = null; }
  try { response = parse(responseValue); } catch { response = null; }
  if (!object(expectedValue) || expectedValue.kind !== CONTRACTNLI_OBJECTIVE_KIND || expectedValue.scope !== 'single-supplied-contract' ||
      !object(expectedValue.hypotheses) || !Array.isArray(expectedValue.available_span_ids)) throw new Error('invalid ContractNLI host reference');
  const expected = expectedValue as unknown as ContractNliExpected;
  if (!object(packet) || packet.schema !== 'natlang.contractnli-task/1' || !Array.isArray(packet.hypotheses) ||
      !object(packet.document) || !Array.isArray(packet.document.span_ids)) throw new Error('invalid ContractNLI visible task packet');
  const knownSpans = new Set(expected.available_span_ids.map(idString));
  if (knownSpans.has(null) || knownSpans.size !== expected.available_span_ids.length) throw new Error('invalid ContractNLI span reference');
  const publicSpanIds = new Set(packet.document.span_ids.map(idString));
  if (publicSpanIds.has(null) || publicSpanIds.size !== knownSpans.size || [...knownSpans].some(id => !publicSpanIds.has(id)))
    throw new Error('visible span catalog does not match host reference');
  const publicHypotheses = new Map(packet.hypotheses.map((row: unknown) => object(row) ? [row.id, row] : [null, null]));
  const expectedIds = Object.keys(expected.hypotheses);
  if (publicHypotheses.size !== expectedIds.length || expectedIds.some(id => !publicHypotheses.has(id)) ||
      packet.hypotheses.some((row: unknown) => !object(row) || typeof row.text !== 'string')) throw new Error('visible hypotheses do not match host reference');
  for (const [hypothesisId, gold] of Object.entries(expected.hypotheses)) {
    if (!gold || !['Entailment', 'Contradiction', 'NotMentioned'].includes(gold.choice) || !Array.isArray(gold.evidence_alternatives))
      throw new Error(`invalid ContractNLI hypothesis reference ${hypothesisId}`);
    if (gold.choice === 'NotMentioned' && gold.evidence_alternatives.length) throw new Error('NotMentioned has no evidence annotation');
    if (gold.choice !== 'NotMentioned' && !gold.evidence_alternatives.length) throw new Error('positive label lacks an evidence alternative');
    for (const alternative of gold.evidence_alternatives) {
      if (!Array.isArray(alternative) || !alternative.length || alternative.some(id => !knownSpans.has(idString(id)!)))
        throw new Error('approved evidence alternative contains an invalid source span');
    }
  }
  const annotations = object(response) && object(response.annotations) ? response.annotations : null;
  if (!annotations) return { quality: 0, gates: { valid_response: false, all_labels_correct: false, all_required_evidence_supported: false },
    detail: { hypotheses: expectedIds.length, labels_correct: 0, evidence_sufficient: 0, invalid_span_ids: 0 } };
  const extraHypothesis = Object.keys(annotations).some(id => !expectedIds.includes(id));
  let labelsCorrect = 0, evidenceSufficient = 0, invalidSpanIds = 0, total = expectedIds.length;
  for (const hypothesisId of expectedIds) {
    const gold = expected.hypotheses[hypothesisId]!;
    const prediction = annotations[hypothesisId];
    if (!object(prediction)) continue;
    const predictedChoice = choice(prediction.choice);
    const spans = Array.isArray(prediction.span_ids) ? prediction.span_ids.map(idString) : [];
    if (spans.some(id => id === null || !knownSpans.has(id))) { invalidSpanIds += spans.filter(id => id === null || !knownSpans.has(id!)).length; continue; }
    const spanIds = new Set(spans as string[]);
    if (spanIds.size !== spans.length) { invalidSpanIds++; continue; }
    const correctLabel = predictedChoice === gold.choice;
    if (correctLabel) labelsCorrect++;
    const supported = gold.choice === 'NotMentioned' || gold.evidence_alternatives.some(alternative => alternative.every(id => spanIds.has(id)));
    if (supported) evidenceSufficient++;
  }
  const quality = total ? (0.75 * labelsCorrect + 0.25 * evidenceSufficient) / total : 0;
  return { quality, gates: { valid_response: !extraHypothesis && invalidSpanIds === 0,
    all_labels_correct: labelsCorrect === total && !extraHypothesis,
    all_required_evidence_supported: evidenceSufficient === total && !extraHypothesis },
    detail: { hypotheses: total, labels_correct: labelsCorrect, evidence_sufficient: evidenceSufficient, invalid_span_ids: invalidSpanIds } };
}
