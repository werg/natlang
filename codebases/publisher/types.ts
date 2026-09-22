export type File = { kind: "text", text: Text, bytes: Num } | { kind: "binary", bytes: Num };
export type Passage = { id: Text, source_id: Text, revision: Text,
  start: Num, end: Num, text: Text };
export type Claim = { text: Text, span_id: Text, revision: Text, quote: Text };
export type Section = { heading: Text, body: Text, claims: Claim[], table_id?: Text };
export type Outline = { title: Text, headings: Text[], selected_tables: Text[], selected_assets: Text[] };
export type Document = { title: Text, evidence_revision: Text, sections: Section[], assets: Text[] };
export type PublishCheck = { ok: Bool, revision: Text, detail: Text };
export type PublishReport = { status: Text, target: Text, revision: Text,
  markdown_sha256: Text, html_sha256: Text, detail: Text };
