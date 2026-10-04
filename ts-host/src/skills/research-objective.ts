/** Host-only scorer for fictional research classification tasks with source-verified citations. */
export const RESEARCH_OBJECTIVE_KIND = 'research-classification' as const;
export type ResearchObjectiveExpected = { kind: typeof RESEARCH_OBJECTIVE_KIND; label: string | null;
  unresolved: boolean; requiredEvidence: { sourceId: string; text: string }[];
  documents: { id: string; text: string }[] };
export type ResearchObjectiveScore = { quality: number; gates: Record<string, boolean>;
  detail: { evidenceCoverage: number; verifiedCitations: number; requiredCitations: number } };

const invalid = (gate: string): ResearchObjectiveScore => ({ quality: 0,
  gates: { citations_verified: false, [gate]: false }, detail: { evidenceCoverage: 0, verifiedCitations: 0, requiredCitations: 0 } });
const obj = (value: unknown): value is Record<string, any> => !!value && typeof value === 'object' && !Array.isArray(value);
function parse(value: unknown): any {
  if (typeof value !== 'string') return value;
  const fenced = /^\s*```(?:json)?\s*([\s\S]*?)```\s*$/.exec(value);
  return JSON.parse(fenced ? fenced[1]! : value);
}

/** Quotes must occur byte-for-byte in a retrieved document; only verified required quotes earn evidence credit. */
export function scoreResearchObjective(packet: unknown, value: unknown, expected: unknown): ResearchObjectiveScore {
  let brief: any, gold: Partial<ResearchObjectiveExpected> | null;
  try { brief = parse(packet); gold = expected as Partial<ResearchObjectiveExpected> | null; }
  catch (error) { throw new Error(`invalid host research packet JSON: ${String(error)}`); }
  if (!obj(brief) || !Array.isArray(brief.catalog) || !gold || gold.kind !== RESEARCH_OBJECTIVE_KIND ||
      (typeof gold.label !== 'string' && gold.label !== null) || typeof gold.unresolved !== 'boolean' || !Array.isArray(gold.documents) ||
      !Array.isArray(gold.requiredEvidence) || gold.requiredEvidence.length === 0)
    throw new Error('invalid host research reference');
  const catalogIds = new Set(brief.catalog.map((row: any) => row?.id));
  const documentIds = new Set(gold.documents.map(document => document?.id));
  if (brief.catalog.some((row: any) => !row || typeof row.id !== 'string' || typeof row.title !== 'string') ||
      catalogIds.size !== brief.catalog.length || documentIds.size !== gold.documents.length || catalogIds.size !== documentIds.size ||
      gold.documents.some(document => !document || typeof document.id !== 'string' || typeof document.text !== 'string' || !catalogIds.has(document.id)) ||
      gold.unresolved !== (gold.label === null))
    throw new Error('host research catalog and private documents disagree');
  const documents = new Map(gold.documents.map(document => [document.id, document.text]));
  const requiredRows = gold.requiredEvidence;
  if (requiredRows.some(row => !row || typeof row.sourceId !== 'string' || typeof row.text !== 'string' ||
      !documents.has(row.sourceId) || !documents.get(row.sourceId)!.includes(row.text)))
    throw new Error('host research required evidence is absent from its pinned document');
  if (new Set(requiredRows.map(row => JSON.stringify([row.sourceId, row.text]))).size !== requiredRows.length)
    throw new Error('host research reference repeats a required evidence span');
  let answer: any;
  try { answer = parse(value); } catch { return invalid('valid_json'); }
  if (!obj(answer) || !Array.isArray(answer.citations)) return invalid('answer_shape');
  try {
    const verified = new Set<string>();
    for (const citation of answer.citations) {
      if (!obj(citation) || typeof citation.sourceId !== 'string' || typeof citation.evidence !== 'string' || !citation.evidence ||
          !catalogIds.has(citation.sourceId) || !documents.has(citation.sourceId) || !documents.get(citation.sourceId)!.includes(citation.evidence))
        return invalid('citations_verified');
      verified.add(JSON.stringify([citation.sourceId, citation.evidence]));
    }
    const required = new Set(requiredRows.map(row => JSON.stringify([row.sourceId, row.text])));
    const hit = [...required].filter(key => verified.has(key)).length;
    const evidenceCoverage = hit / required.size;
    const classificationCorrect = answer.label === gold.label;
    const uncertaintyHandled = answer.unresolved === gold.unresolved;
    // Source-verified evidence can earn limited credit when the final decision is wrong or overconfident.
    const decisionFactor = classificationCorrect && uncertaintyHandled ? 1 : 0.5;
    return { quality: evidenceCoverage * decisionFactor,
      gates: { citations_verified: true, required_evidence_supported: evidenceCoverage === 1,
        classification_correct: classificationCorrect, unresolved_status_correct: uncertaintyHandled },
      detail: { evidenceCoverage, verifiedCitations: hit, requiredCitations: required.size } };
  } catch { return invalid('answer_shape'); }
}
