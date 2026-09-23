export type Hit = { id: string, source_id: string, revision: string, preview: string, score: number };
export type SearchResult = { hits: Hit[], total: number, truncated: boolean, collection_revision: string };
export type Passage = { id: string, source_id: string, revision: string, start: number, end: number, text: string };
export type Claim = { text: string, span_id: string, revision: string, quote: string };
export type Draft = { answer: string, claims: Claim[], gaps: string[] };
export type EvidenceStatus = "citation-checked" | "partial" | "unresolved" | "invalid-citation";
export type EvidenceAnswer = { status: EvidenceStatus, answer: string, claims: Claim[], gaps: string[],
  collection_revision: string, detail: string };
