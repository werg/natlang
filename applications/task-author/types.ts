export type OracleKind = 'exact' | 'verifier';

export type TaskSeed = {
  id: string;
  domain: string;
  construct: string;
  oracleKind: OracleKind;
  summary: string;
  splitGroup: string;
};

export type AuthoredTask = {
  id: string;
  split_group: string;
  constructs: string[];
  oracle: OracleKind;
  verifier?: string;
  prompt: string;
  files: Record<string, string>;
  keywordBaselineInput?: string;
  expectedPassOnBaseline?: boolean;
};

export type QualificationResult = {
  admitted: boolean;
  task: AuthoredTask;
  structuralIssues: string[];
  semanticVerdict: boolean;
  oracleVerdict: boolean;
  baselineDefeated: boolean;
  escalated: boolean;
};
