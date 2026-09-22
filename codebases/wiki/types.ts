export type MergeProfile = { model: string, source: string, seed: number };
export type WikiBlock = { id: string, kind: string, text: string,
  language?: string, returns?: string };
export type WikiPage = { id: string, revision: string, blocks: WikiBlock[],
  unresolved?: Conflict[] };
export type WikiUpdate = { id: string, block_id: string, base_revision: string,
  author: string, text: string };
export type Conflict = { update_id: string, alternatives: string[] };
export type PreparedMerge = { valid: boolean, detail: string,
  updates: WikiUpdate[], presentation: string };
export type MergeDraft = { blocks: WikiBlock[], accounted: string[],
  unresolved: Conflict[] };
export type MergeReport = { status: string, page: WikiPage, detail: string };
export type CellResult = { status: string, page_revision: string,
  source_revision: string, value_text: string, trace_events: number };
