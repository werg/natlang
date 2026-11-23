export type OracleKind = 'exact' | 'verifier';

export type TaskSeed = {
  id: string;
  domain: string;
  construct: string;
  oracleKind: OracleKind;
  summary: string;
  splitGroup: string;
};

/** An authored task with the answers that qualify its oracle: `reference` must pass it, every one of `negatives` and
 * the `keywordBaseline` (the strongest answer a cue-word rule gives) must fail it. */
export type AuthoredTask = {
  id: string;
  split_group: string;
  constructs: string[];
  oracle: OracleKind;
  verifier?: string;
  prompt: string;
  files: Record<string, string>;
  reference: string;
  negatives: string[];
  keywordBaseline: string;
};

export type QualificationResult = {
  admitted: boolean;
  task: AuthoredTask;
  /** Structural, custody and replay problems left after repair (empty when admitted). */
  problems: string[];
  semanticVerdict: boolean;
  oracleVerdict: boolean;
  escalated: boolean;
};
