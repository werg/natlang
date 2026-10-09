export type FileIdentity = { path: string, sha256: string, lines: number };
export type RepoSnapshot = { revision: string, files: FileIdentity[] };
export type SearchHit = { path: string, offset: number, line: number, excerpt: string };
export type SearchResult = { revision: string, hits: SearchHit[] };
export type Patch = { path: string, old: string, new: string };
export type Check = { id: string, status: "passed" | "failed", output: string, output_bytes: number, truncated: boolean, detail?: string };
export type Validation = { revision: string, status: "passed" | "failed", checks: Check[] };

/**
 * What the request asks for, restated so each later stage can work from it. old and new are the identifiers, call
 * shapes or phrases the change replaces and introduces (new is empty for a removal). queries are the exact strings to
 * search for to find every place that mentions old, most specific first. invariants are behaviors that must stay the
 * same, taken from the request and the check ids.
 */
export type Intent = { summary: string, old: string, new: string, queries: string[], invariants: string[] };

/**
 * One place in the code that mentions the old thing: the lines from..to of path (1-based, inclusive) at one revision.
 * text is those lines exactly as the file holds them. hits is how many search hits fall inside.
 */
export type Site = { id: string, path: string, from: number, to: number, text: string, hits: number };

/**
 * How a site uses the old thing. pattern is one of: declaration, call, import, export, type-position,
 * property-access, comment-or-doc, string-literal, test, unrelated. action is edit when the migration must change the
 * site and leave when it must stay as it is. reason is one sentence.
 */
export type Usage = { pattern: string, action: 'edit' | 'leave', reason: string };

export type Classified = { site: Site, usage: Usage };

/** The migration's plan: the site ids to edit in the order to edit them, the sites left alone with the reason, and risks to watch in the checks. */
export type Plan = { edits: string[], leave: { site: string, reason: string }[], risks: string[] };

/** Whether a patch is exact: its old text occurs once in its file, it changes something, and it changes only what its site's usage calls for. */
export type Exactness = { exact: boolean, problem: string };

/** The outcome of editing one site. status patched has patches; left has none; failed has none and a note saying why. */
export type SiteEdit = { site: string, usage: Usage, patches: Patch[], status: 'patched' | 'left' | 'failed', note: string };

/**
 * One cause of a failing candidate, from a check's output or from a rejected patch. kind: missed-site (the old thing
 * is still used somewhere), wrong-edit (an edit broke behavior), test-expectation (a test states the old behavior),
 * environment (the check cannot run here), unrelated (fails without this migration). path and line locate the
 * evidence in the candidate (empty and 0 when unknown). repairable is true for the first three kinds.
 */
export type Finding = { check: string, kind: string, path: string, line: number, evidence: string, repairable: boolean };

/**
 * The repair loop's state. snapshot is the current candidate and validation its checks (null before the first run).
 * rejected is the message of a patch set that could not be applied to it (empty when none). findings are the causes
 * triage found in the last round. remaining is the budget of rounds left. revisions are the candidates tried so far.
 */
export type RepairState = {
  snapshot: RepoSnapshot, validation: Validation | null, rejected: string, findings: Finding[],
  remaining: number, revisions: string[],
};

/** The closing words of a migration. */
export type Summary = { summary: string, next: string[] };

export type ChangedFile = { path: string, before_sha256: string, after_sha256: string };
export type MigrationReport = { status: 'reviewable' | 'checks-failed', base: string, revision: string, changed: ChangedFile[], checks: Check[] };

/** The migration's result: the exact report, and what the stages decided on the way. */
export type Migration = {
  report: MigrationReport, intent: Intent, plan: Plan, edits: SiteEdit[], findings: Finding[], rounds: number,
  summary: string, next: string[],
};
