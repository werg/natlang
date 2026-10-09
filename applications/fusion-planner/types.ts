/** Who reads a value besides, or as, the planned consumer. */
export type ReaderKind = 'consumer' | 'other-call' | 'crisp-code' | 'service' | 'eval' | 'host-return' | 'trace-ui';
export type Reader = { kind: ReaderKind; name?: string; site: string; detail: string; certain: boolean };
export type Stage = { name: string; source: string; model: string | null; returns: string };
export type Consumer = { name: string; source: string; model: string | null; returns: string; param: string | null; paramType: string | null };
/** The facts of one candidate edge: the value one function returns and another receives. */
export type EdgeFacts = { id: string; scope: string; scopeKind: 'nl' | 'typescript'; chain: string; producer: Stage; consumer: Consumer;
  type: string; flow: 'nested' | 'bound' | 'implicit' | 'typed'; variable: string | null; dialect: string; readers: Reader[];
  producerSites: number; consumerSites: number; alreadySoft: boolean; finite: boolean; excerpt?: string;
  /** Present when the readers were seen in recorded runs instead of proven from the source. */
  observed?: Observed };
/** Recorded runs behind a reader set: runs that fed only the consumer, runs that did not, the runs a plan needs, the store revision. */
export type Observed = { runs: number; contradicted: number; minRuns: number; revision: string };
export type Decision = 'fuse' | 'keep-text';
export type EdgeDecision = { edge: string; decision: Decision; reason: string };
/** A rejected earlier answer for one edge, and why. */
export type Problem = { edge: string; problem: string };
/** Whether a person or a later check needs to read the value. */
export type ReadNeed = { needed: boolean; reason: string };
