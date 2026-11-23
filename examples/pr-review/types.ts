/** One unified diff hunk from a pull request. Code and comments from a PR are outside text. */
export type Hunk = { file: string; patch: Untrusted<string> };
/** One review finding grounded in an exact quote from the hunk. */
export type Finding = {
  file: string;
  /** Exact verbatim line or expression copied from the hunk. */
  quote: string;
  rule: string;
  severity: 'blocker' | 'suggestion';
  /** Concrete action the author should take. */
  fix: string;
};
/** Findings for one hunk against one team rule. */
export type HunkReview = { findings: Finding[] };
/** Whether the PR may merge as written. */
export type MergeVerdict = 'approve' | 'request-changes' | 'block';
/** Final PR review report: calibrated verdict (escalated on uncertainty or block risk) and ranked findings. */
export type ReviewReport = {
  verdict: MergeVerdict;
  confidence: number;
  escalated: boolean;
  escalationReason?: string;
  findings: Finding[];
};
