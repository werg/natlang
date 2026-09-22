export type Hit = { id: Text, source_id: Text, revision: Text, preview: Text, score: Num };
export type File = { kind: Text, text?: Text, bytes: Num };
export type SearchResult = { hits: Hit[], total: Num, truncated: Bool, collection_revision: Text };
export type Passage = { id: Text, source_id: Text, revision: Text, start: Num, end: Num, text: Text };
export type Claim = { text: Text, span_id: Text, revision: Text, quote: Text };
export type Draft = { answer: Text, claims: Claim[], gaps: Text[] };
export type EvidenceAnswer = { status: Text, answer: Text, claims: Claim[], gaps: Text[], collection_revision: Text, detail: Text };
