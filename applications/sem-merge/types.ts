/** Three-way file inputs for a merge operation. */
export type ThreeWayFile = {
  path: string;
  base: string;
  left: string;
  right: string;
};

/** One overlapping conflict region extracted from a three-way diff. */
export type ConflictHunk = {
  path: string;
  startLine: number;
  endLine: number;
  base: string;
  left: string;
  right: string;
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
