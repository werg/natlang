/** Three-way file inputs for a merge operation. External branch texts are untrusted outside data. */
export type ThreeWayFile = {
  path: string;
  base: Untrusted<string>;
  left: Untrusted<string>;
  right: Untrusted<string>;
};

/** One overlapping conflict region extracted from a three-way diff. */
export type ConflictHunk = {
  path: string;
  startLine: number;
  endLine: number;
  base: Untrusted<string>;
  left: Untrusted<string>;
  right: Untrusted<string>;
};

/** Resolution record for one overlapping conflict hunk. */
export type ConflictResolution = {
  hunk: ConflictHunk;
  status: 'merged' | 'unresolved';
  mergedText: string;
  compatible: boolean;
  confidence: number;
  repairAttempts: number;
  problems: string[];
};

/** Final outcome of merging one ThreeWayFile. */
export type MergeOutcome = {
  path: string;
  mode: 'crisp' | 'json-semantic' | 'text-semantic';
  status: 'clean' | 'conflict';
  content: string;
  llmCallsUsed: boolean;
  resolutions: ConflictResolution[];
  unresolvedCount: number;
};
