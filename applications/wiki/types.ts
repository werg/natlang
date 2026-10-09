/** Which implementation runs a hot-path policy. Both answer the same interface; the setting selects. */
export type Policy = "crisp" | "natlang";

/** The wiki's policy settings, passed to every stage that has a pluggable part. */
export type WikiSettings = {
  /** Summarizing one update into a Change, once per update: "crisp" compares texts exactly, "natlang" reads the intent. */
  changes: Policy,
  /** Judging whether a recorded cell result survives an edit: "crisp" drops all results, "natlang" follows dependencies. */
  staleness: Policy,
};

export type MergeProfile = { model: string, source: string, seed: number };

/** A stable block of a page. A cell's text is its source: JavaScript is a function body over `input`, natlang is instructions. */
export type WikiBlock = { id: string, kind: "prose" | "cell", text: string,
  language?: "javascript" | "natlang", returns?: "string" | "number" | "boolean" };

/** An edit left for a person: the update that could not be merged, with every competing text for its block. */
export type Conflict = { update_id: string, block_id: string, alternatives: string[] };
export type WikiPage = { id: string, revision: string, blocks: WikiBlock[], unresolved?: Conflict[] };

/** One author's proposal: the whole new text of one block, made against base_revision. */
export type WikiUpdate = { id: string, block_id: string, base_revision: string, author: string, text: string };
export type PreparedMerge = { valid: boolean, detail: string, updates: WikiUpdate[], presentation: string };

/** The merge pipeline's answer: the page's blocks (same IDs, same order), every update ID once, the conflicts left. */
export type MergeDraft = { blocks: WikiBlock[], accounted: string[], unresolved: Conflict[] };

/** What happened to a merge. A merged page may carry its derived structure and the repairs maintenance proposes. */
export type MergeReport = { status: "merged" | "unresolved" | "rejected", page: WikiPage, detail: string,
  structure?: Structure, repairs?: WikiUpdate[] };

/** A recorded cell result. */
export type CellResult = { status: string, page_revision: string, source_revision: string,
  value_text: string, trace_events: number };

// Edit pipeline ---------------------------------------------------------------------------------------------------

/** One update, summarized against the block it edits. */
export type Change = {
  update_id: string,
  block_id: string,
  author: string,
  /** noop: same text. format: same words, other spacing or markup. extend: the old text kept, more added. trim: the old text kept, some removed. rewrite: anything else. */
  kind: "noop" | "format" | "extend" | "trim" | "rewrite",
  /** One sentence: what the block should say or do once this update is in. */
  intent: string,
  /** Statements, steps or behaviours the new text has that the base block does not, as short phrases. */
  adds: string[],
  /** Statements, steps or behaviours the base block has that the new text drops or reverses, as short phrases. */
  removes: string[],
  /** The update's whole new text. */
  text: string,
};

/** How two changes to one block relate. compatible: both can hold at once. redundant: one already carries the other's meaning. contradictory: both cannot hold. */
export type Relation = "compatible" | "redundant" | "contradictory";
export type Related = { a: string, b: string, relation: Relation };

/**
 * What to do with the changes to one block. combine: merge these updates into one new text. covered: update_ids[0] adds
 * nothing beyond `by`, which is combined or itself covered. conflict: these updates exclude each other; leave them for a person.
 */
export type PlanStep = { action: "combine" | "covered" | "conflict", update_ids: string[], by?: string, note: string };
export type MergePlan = { block_id: string, steps: PlanStep[] };

/** A merged text. clear is false when the combined meaning of the changes cannot be written down unambiguously. */
export type Merged = { text: string, clear: boolean, reason: string };

/** Whether a merged text carries one change's intent. */
export type Honor = "honored" | "partly" | "lost";

/** One round of verification: the text and the update IDs whose intent it does not carry (null before the first check). */
export type Attempt = { text: string, clear: boolean, lost: string[] | null };

/** The result for one block: its new form, the updates it accounts for, and the conflicts left. */
export type BlockOutcome = { block: WikiBlock, accounted: string[], unresolved: Conflict[] };

// Page structure --------------------------------------------------------------------------------------------------

/** A block whose first line is a heading: "# Title" through "###### Title". */
export type Heading = { block_id: string, level: number, title: string };
/** A link in a prose block: [[target]] or [[target|shown text]]. */
export type LinkRef = { block_id: string, target: string, text: string };
export type Scan = { headings: Heading[], links: LinkRef[] };

/** A section starts at a heading block and holds the blocks up to the next heading of its level or higher. */
export type Section = { id: string, title: string, level: number, block_ids: string[], parent: string | null };
export type Outline = { sections: Section[], problems: string[] };

/**
 * ok: the target names a section ID, a block ID or a cell ID of this page. renamed: it named a section of the
 * previous page that is now called something else; resolves_to is the new ID. broken: it names nothing.
 * ambiguous: it fits several; resolves_to is null.
 */
export type LinkStatus = { block_id: string, target: string, status: "ok" | "renamed" | "broken" | "ambiguous",
  resolves_to: string | null, reason: string };

/** Blocks whose text, language or return type differ between two pages. */
export type Delta = { changed: string[], added: string[], removed: string[] };

/** What a cell reads: blocks of the page, other cells' results, project files. */
export type Dependency = { cell: string, blocks: string[], cells: string[], files: boolean };

/** What the page remembers about a cell's last result. */
export type CellRecord = { block_id: string, page_revision: string, source_revision: string };
export type CellVerdict = { block_id: string, status: "fresh" | "stale", reason: string };

/** Everything derived from a page revision. */
export type Structure = { revision: string, outline: Outline, links: LinkStatus[], cells: CellVerdict[] };
export type Maintenance = { structure: Structure, repairs: WikiUpdate[] };

// Cell evaluation policy ------------------------------------------------------------------------------------------

export type CellRequest = { block_id: string, input: string, origin: "user" | "auto" };
export type CellRun = { block_id: string, input: string };
/** Runs in batches: the cells of a batch are independent and run together, and a batch starts when the one before has finished. */
export type CellPlan = { batches: CellRun[][], refused: { block_id: string, reason: string }[] };
