export type MergeProfile = { model: Text, source: Text, seed: Num };
export type WikiBlock = { id: Text, kind: Text, text: Text,
  language?: Text, returns?: Text };
export type WikiPage = { id: Text, revision: Text, blocks: WikiBlock[],
  unresolved?: Conflict[] };
export type WikiUpdate = { id: Text, block_id: Text, base_revision: Text,
  author: Text, text: Text };
export type Conflict = { update_id: Text, alternatives: Text[] };
export type PreparedMerge = { valid: Bool, detail: Text,
  updates: WikiUpdate[], presentation: Text };
export type MergeDraft = { blocks: WikiBlock[], accounted: Text[],
  unresolved: Conflict[] };
export type MergeReport = { status: Text, page: WikiPage, detail: Text };
export type CellResult = { status: Text, page_revision: Text,
  source_revision: Text, value_text: Text, trace_events: Num };
