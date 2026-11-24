/** Specification of a repository-wide code migration. */
export type CodemodSpec = {
  id: string;
  title: string;
  description: string;
  guidance: string;
  examples?: { before: string; after: string }[];
};

/** A deterministic zero-token codemod rule (literal or regex find-and-replace). */
export type CrispRule = {
  id: string;
  description: string;
  find: string;
  replace: string;
  /** When true, `find` is compiled as a JavaScript RegExp with `flags` (default `'g'`). */
  regex?: boolean;
  flags?: string;
  origin: 'initial' | 'mined';
};

/** One source file entering the migration pipeline; source contents are outside data. */
export type SourceFile = {
  path: string;
  content: Untrusted<string>;
};

/** Candidate crisp rule synthesized from verified semantic file migrations. */
export type MinedRuleCandidate = {
  find: string;
  replace: string;
  description: string;
  regex?: boolean;
  flags?: string;
};

/** Verified before/after pair supplied to rule mining. */
export type VerifiedMigrationPair = {
  path: string;
  before: Untrusted<string>;
  after: Untrusted<string>;
};

/** Outcome of verifying a mined crisp rule against already-migrated files before admitting it. */
export type MinedRuleOutcome = {
  rule: CrispRule;
  admitted: boolean;
  verifiedPairs: number;
  regressions: number;
  reason?: string;
};

export type FileMigrationStatus = 'unchanged' | 'crisp' | 'semantic' | 'failed';

/** Per-file migration outcome, provenance, and verifier status. */
export type FileResult = {
  path: string;
  status: FileMigrationStatus;
  original: string;
  migrated: string;
  ruleIds: string[];
  attempts: number;
  problems: string[];
};

/** Aggregate codemod report with crisp-vs-semantic coverage and admitted mined rules. */
export type CodemodReport = {
  specId: string;
  totalFiles: number;
  crispFiles: number;
  semanticFiles: number;
  unchangedFiles: number;
  failedFiles: number;
  /** Natural-language runs: one migration over the semantic remainder (when there is one), plus one per mining attempt. */
  llmCalls: number;
  /** The migration loop's closing words for the semantic remainder, when it ran. */
  migrationSummary?: string;
  results: FileResult[];
  admittedRules: CrispRule[];
  minedRules: MinedRuleOutcome[];
};
