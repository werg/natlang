export type File = { kind: "text", text: string, bytes: number } | { kind: "binary", bytes: number };
export type Passage = { id: string, source_id: string, revision: string,
  start: number, end: number, text: string };
export type Claim = { text: string, span_id: string, revision: string, quote: string };
export type Section = { heading: string, body: string, claims: Claim[], table_id?: string };
export type Outline = { title: string, headings: string[], selected_tables: string[], selected_assets: string[] };
export type Document = { title: string, evidence_revision: string, sections: Section[], assets: string[] };
export type PublishCheck = { ok: boolean, revision: string, detail: string };
export type PublishReport = { status: string, target: string, revision: string,
  markdown_sha256: string, html_sha256: string, detail: string };
