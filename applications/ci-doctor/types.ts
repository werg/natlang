/** One CI job failure trace; logs come from external build output and are untrusted. */
export type FailureTrace = {
  id: string;
  job: string;
  log: Untrusted<string>;
  commits: string[];
};

/** Finite root-cause classification for a CI failure trace. */
export type FailureKind = 'flake' | 'env' | 'regression';

/** Cluster of CI failures sharing a root-cause class and normalized failure signature. */
export type Cluster = {
  kind: FailureKind;
  signature: string;
  source: 'crisp' | 'nl';
  traces: FailureTrace[];
};

/** Trace whose classification confidence fell below the partition floor. */
export type UncertainTrace = {
  trace: FailureTrace;
  kind: FailureKind;
  confidence: number;
};

/** State tracked during commit bisection. */
export type BisectState = {
  low: number;
  high: number;
  tested: string[];
  culprit?: string;
  done: boolean;
};

/** Structured summary returned by a candidate fix reducer applied to a repository Folder. */
export type FixProposal = {
  summary: string;
  filesChanged: string[];
  explanation: string;
};

/** Candidate fix rejected by exact test or assertion-preservation verification. */
export type RejectedCandidate = {
  index: number;
  problems: string[];
};

/** End-to-end diagnosis and verified repair report for a batch of CI failures. */
export type CiReport = {
  crispFlakes: FailureTrace[];
  clusters: Cluster[];
  uncertain: UncertainTrace[];
  culpritCommit?: string;
  bisectedCommits: string[];
  fixAccepted: boolean;
  chosenBranch: number | null;
  chosenFix?: FixProposal;
  rejectedCandidates: RejectedCandidate[];
};
