/** One unified diff hunk from a pull request. PR code and comments are untrusted external input. */
export type Hunk = {
  file: string;
  patch: Untrusted<string>;
  isGenerated?: boolean;
};

/** Functional category of a pull-request hunk. */
export type HunkKind = 'code' | 'docs' | 'tests' | 'config' | 'trivial';

/** Relative engineering risk of a pull-request hunk. */
export type RiskLevel = 'low' | 'medium' | 'high';

/** Finite record classification of a hunk (15 values, scored jointly via readout: decision). */
export type HunkClass = {
  kind: HunkKind;
  risk: RiskLevel;
};

/** One grounded review finding backed by a verbatim quote inside the hunk patch. */
export type Finding = {
  file: string;
  /** Exact verbatim line or expression copied from hunk.patch. */
  quote: string;
  rule: string;
  severity: 'blocker' | 'suggestion';
  /** Concrete edit the PR author should make. */
  fix: string;
};

/** Findings produced for one hunk against one bound team rule. */
export type HunkReview = {
  findings: Finding[];
};

/** Team engineering convention bound onto checkHunk via withContext. */
export type Rule = {
  name: string;
  guidance: string;
  /** Optional list of hunk kinds this rule applies to; omitted means all non-trivial hunks. */
  appliesTo?: HunkKind[];
};

/** Final merge gate decision for a pull request. */
export type MergeVerdict = 'approve' | 'request-changes' | 'block';

/** Per-hunk classification and crisp-skip accounting. */
export type HunkSummary = {
  file: string;
  classification: HunkClass;
  crispSkipped: boolean;
};

/** Draft documentation or test-note update returned by proposeDocFix.nl. */
export type DocFixDraft = {
  path: string;
  content: string;
  summary: string;
};

/** Result of proposing and verifying a documentation fix over a Folder transaction. */
export type DocFixResult = {
  accepted: boolean;
  path?: string;
  summary?: string;
  attempts: number;
  problems: string[];
};

/** End-to-end PR shepherd report. */
export type ShepherdReport = {
  verdict: MergeVerdict;
  confidence: number;
  escalated: boolean;
  escalationReason?: string;
  findings: Finding[];
  hunks: HunkSummary[];
  crispHunks: number;
  semanticHunks: number;
  docFix?: DocFixResult;
};
