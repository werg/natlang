import type { Untrusted } from '@natlang/node';

/** Source text is data: a model sees it only as a quoted block labelled with its document. */
export type Hit = { id: string, source_id: string, revision: string, preview: Untrusted<string>, score: number };
export type SearchResult = { hits: Hit[], total: number, truncated: boolean, collection_revision: string };
/** One paragraph of a document, pinned to the document's revision. The evidence app owns this type; other apps re-export it. */
export type Passage = { id: string, source_id: string, revision: string, start: number, end: number, text: Untrusted<string> };
/** What a model writes for one claim: its words, the passage id and a stretch copied from that passage. */
export type ClaimDraft = { text: string, span_id: string, quote: string };
/** A claim as cited: the host adds the revision of the passage it rests on. */
export type Claim = { text: string, span_id: string, revision: string, quote: string };
export type Draft = { answer: string, claims: Claim[], gaps: string[] };
export type EvidenceStatus = "citation-checked" | "partial" | "unresolved" | "invalid-citation";
export type EvidenceAnswer = { status: EvidenceStatus, answer: string, claims: Claim[], gaps: string[],
  collection_revision: string, detail: string };

// Refined results. Each predicate has a crisp checker in refinements.ts; what needs the search result or the passages
// (membership, quotes) is checked by the host in index.ts and goes back to the stage once with the exact problem.
export type SearchPhrases = Is<string[], "two or three search phrases">;
export type HitIds = Is<string[], "each id listed once">;
