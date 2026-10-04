/** Host-only exact scorer for SciFact classification on the explicitly supplied cited abstracts. */
export const SCIFACT_OBJECTIVE_KIND = 'scifact-claim-evidence' as const;
export type ScifactEvidenceSet = { doc_id: string | number; sentence_ids: Array<number | string> };
export type ScifactExpected = { kind: typeof SCIFACT_OBJECTIVE_KIND; label: 'SUPPORT' | 'CONTRADICT' | 'NOT_ENOUGH_INFO';
  accepted_evidence_sets: ScifactEvidenceSet[][]; available_documents: { doc_id: string | number; sentence_ids: Array<number | string> }[];
  scope: 'supplied-cited-documents-only' };
export type ScifactScore = { quality: number; gates: Record<string, boolean>;
  detail: { classification_correct: boolean; sufficient_evidence: boolean; cited_documents: number; cited_sentences: number } };

const record = (x: unknown): x is Record<string, unknown> => !!x && typeof x === 'object' && !Array.isArray(x);
const normId = (x: unknown): string | null => (typeof x === 'string' || typeof x === 'number') && String(x).trim() ? String(x).trim() : null;
const normLabel = (x: unknown): string | null => typeof x === 'string' ? x.trim().toUpperCase().replace(/[ -]+/gu, '_') : null;
function parse(x: unknown): unknown {
  if (typeof x !== 'string') return x;
  const fenced = /^\s*```(?:json)?\s*([\s\S]*?)```\s*$/u.exec(x);
  return JSON.parse(fenced ? fenced[1]! : x);
}
function normalizePairs(value: unknown): string[] | null {
  if (!Array.isArray(value)) return null;
  const pairs: string[] = [];
  for (const item of value) {
    if (!record(item)) return null;
    const doc = normId(item.doc_id);
    const raw = item.sentence_ids;
    if (!doc || !Array.isArray(raw) || raw.length === 0) return null;
    const sentences = raw.map(normId);
    if (sentences.some(id => id === null)) return null;
    const sorted = [...new Set(sentences as string[])].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    if (sorted.length !== sentences.length) return null;
    pairs.push(JSON.stringify([doc, sorted]));
  }
  return pairs.sort();
}

/**
 * Scores exact labels and one of the host-approved sufficient annotation sets. No answer-provided score,
 * feasibility flag, or evidence text is trusted. NOT_ENOUGH_INFO is explicitly scoped to supplied documents.
 */
export function scoreScifactObjective(packetValue: unknown, response: unknown, expected: unknown): ScifactScore {
  if (!record(expected) || expected.kind !== SCIFACT_OBJECTIVE_KIND ||
      !['SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO'].includes(String(expected.label)) ||
      expected.scope !== 'supplied-cited-documents-only' || !Array.isArray(expected.available_documents) ||
      !Array.isArray(expected.accepted_evidence_sets)) throw new Error('invalid SciFact host reference');
  const docs = new Map<string, Set<string>>();
  for (const doc of expected.available_documents) {
    if (!record(doc) || !normId(doc.doc_id) || !Array.isArray(doc.sentence_ids)) throw new Error('invalid SciFact document reference');
    const id = normId(doc.doc_id)!;
    if (docs.has(id)) throw new Error('duplicate SciFact document reference');
    const sentenceIds = doc.sentence_ids.map(normId);
    if (sentenceIds.some(x => x === null) || new Set(sentenceIds).size !== sentenceIds.length) throw new Error('invalid SciFact sentence IDs');
    docs.set(id, new Set(sentenceIds as string[]));
  }
  const accepted = expected.accepted_evidence_sets.map(normalizePairs);
  if (accepted.some(x => x === null)) throw new Error('invalid SciFact sufficient evidence set');
  if (expected.label === 'NOT_ENOUGH_INFO' && (accepted.length !== 0 || docs.size === 0))
    throw new Error('NOT_ENOUGH_INFO must be scoped to supplied documents and have no positive evidence set');
  if (expected.label !== 'NOT_ENOUGH_INFO' &&
      (accepted.length === 0 || accepted.some(set => set!.length === 0)))
    throw new Error('classified SciFact answer needs nonempty evidence alternatives');
  for (const set of accepted as string[][]) for (const pair of set) {
    const [docId, ids] = JSON.parse(pair) as [string, string[]];
    const known = docs.get(docId);
    if (!known || ids.some(id => !known.has(id))) throw new Error('accepted evidence references an absent document sentence');
  }
  let packet: unknown;
  try { packet = parse(packetValue); } catch { throw new Error('invalid public SciFact packet'); }
  if (!record(packet) || !Array.isArray(packet.catalog) ||
      JSON.stringify(packet.allowedLabels) !== JSON.stringify(['SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO']) ||
      !packet.catalog.every((row: unknown) => record(row) && typeof row.id === 'string' && typeof row.title === 'string') ||
      new Set(packet.catalog.map((row: Record<string, unknown>) => row.id)).size !== docs.size ||
      packet.catalog.some((row: Record<string, unknown>) => !docs.has(row.id as string)))
    throw new Error('public SciFact catalog differs from host document reference');
  let answer: unknown;
  try { answer = parse(response); } catch { answer = null; }
  const invalid = { quality: 0, gates: { valid_answer: false, classification_correct: false, evidence_supported: false },
    detail: { classification_correct: false, sufficient_evidence: false, cited_documents: 0, cited_sentences: 0 } };
  if (!record(answer) || !Array.isArray(answer.citations)) return invalid;
  const label = normLabel(answer.label);
  if (!['SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO'].includes(String(label))) return invalid;
  let pairs: string[] | null = null;
  if (answer.citations.length === 0) pairs = [];
  else {
    pairs = [];
    const seenDocs = new Set<string>();
    for (const citation of answer.citations) {
      if (!record(citation)) return invalid;
      const docId = normId(citation.doc_id);
      const ids = Array.isArray(citation.sentence_ids) ? citation.sentence_ids.map(normId) : [];
      if (!docId || !docs.has(docId) || seenDocs.has(docId) || ids.length === 0 || ids.some(id => id === null)) return invalid;
      seenDocs.add(docId);
      const known = docs.get(docId)!;
      const normalized = [...new Set(ids as string[])].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
      if (normalized.length !== ids.length || normalized.some(id => !known.has(id))) return invalid;
      pairs.push(JSON.stringify([docId, normalized]));
    }
    if (new Set(pairs).size !== pairs.length) return invalid;
  }
  const goldPairs = accepted as string[][];
  const correct = label === expected.label;
  const answerSentenceMap = new Map<string, Set<string>>();
  for (const pair of pairs) {
    const [docId, sentenceIds] = JSON.parse(pair) as [string, string[]];
    answerSentenceMap.set(docId, new Set(sentenceIds));
  }
  const sufficient = expected.label === 'NOT_ENOUGH_INFO' ? pairs.length === 0
    : goldPairs.some(set => set.every(pair => {
      const [docId, sentenceIds] = JSON.parse(pair) as [string, string[]];
      const cited = answerSentenceMap.get(docId);
      return !!cited && sentenceIds.every(id => cited.has(id));
    }));
  return { quality: correct && sufficient ? 1 : 0,
    gates: { valid_answer: true, classification_correct: correct, evidence_supported: sufficient,
      evidence_scope_is_supplied_documents: true },
      detail: { classification_correct: correct, sufficient_evidence: sufficient,
      cited_documents: pairs.length, cited_sentences: pairs.reduce((sum, pair) => sum + (JSON.parse(pair) as [string, string[]])[1].length, 0) } };
}
