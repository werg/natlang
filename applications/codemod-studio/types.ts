/** Specification of a repository-wide code migration. */
export type MigrationSpec = {
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
  after: string;
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

/** Aggregate migration report with crisp-vs-semantic coverage and admitted mined rules. */
export type MigrationReport = {
  specId: string;
  totalFiles: number;
  crispFiles: number;
  semanticFiles: number;
  unchangedFiles: number;
  failedFiles: number;
  llmCalls: number;
  results: FileResult[];
  admittedRules: CrispRule[];
  minedRules: MinedRuleOutcome[];
};
