/** What one instruction step asks the model to do (`stepKind.nl`; the migrator's StepKind). */
export type StepKind = 'judgment' | 'mechanism' | 'guard-for-type' | 'redundant';

export type KindedStep = { text: string; kind: StepKind };

export type IdiomIssueKind =
  | 'mechanism-in-prose'
  | 'guard-for-type'
  | 'redundant'
  | 'prose-interpreter'
  | 'cue-word-rule'
  | 'untyped-external-input';

export type IdiomIssue = {
  file: string;
  kind: IdiomIssueKind;
  stepIndex?: number;
  excerpt: string;
  confidence: number;
  escalated: boolean;
  suggestion: string;
};

/** One `.nl` source of the reviewed program: its path (relative to the program root) and text. `hasHelpers` says
 * whether its callable folder holds TypeScript helpers. */
export type ProgramFile = {
  path: string;
  content: string;
  hasHelpers?: boolean;
};

export type IdiomReport = {
  filesReviewed: number;
  stepsReviewed: number;
  clean: boolean;
  issues: IdiomIssue[];
};
