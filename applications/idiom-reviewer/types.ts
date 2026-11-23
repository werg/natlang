export type GroupKind = 'judgment' | 'mechanism' | 'prose-interpreter';

export type SentenceGroup = {
  file: string;
  index: number;
  text: string;
};

export type IdiomIssueKind =
  | 'mechanism-in-prose'
  | 'prose-interpreter'
  | 'cue-word-rule'
  | 'untyped-external-input';

export type IdiomIssue = {
  file: string;
  kind: IdiomIssueKind;
  groupIndex?: number;
  excerpt: string;
  confidence: number;
  escalated: boolean;
  suggestion: string;
};

export type ProgramFile = {
  path: string;
  content: string;
};

export type IdiomReport = {
  filesReviewed: number;
  groupsReviewed: number;
  clean: boolean;
  issues: IdiomIssue[];
};
