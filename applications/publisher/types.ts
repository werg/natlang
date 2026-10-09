// One definition of a passage and of a claim, owned by the evidence app and re-exported here (the loader follows the
// re-export, so the .nl functions of this app see the same types).
export type { Passage, Claim, ClaimDraft } from '../evidence/types.js';
import type { Claim, ClaimDraft } from '../evidence/types.js';

/** A table of the table store. Cells are printed by the renderer from the store, never from prose. */
export type Table = { columns: string[], rows: (string | number)[][] };
export type Section = { heading: string, body: string, claims: Claim[], table_id?: string };
/** One planned section: what it is for, which passages it draws on, and the table and assets that belong to it. */
export type OutlineSection = { heading: string, purpose: string, passage_ids: string[], table_id: string | null, asset_ids: string[] };
export type Outline = { title: string, sections: OutlineSection[] };
/** What a model writes for one section; the host adds heading, revisions, table and assets. */
export type SectionDraft = { body: string, claims: ClaimDraft[] };
export type Document = { title: string, evidence_revision: string, sections: Section[], assets: string[] };
