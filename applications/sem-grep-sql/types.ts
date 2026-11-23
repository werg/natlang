/** A repository or document file whose content is untrusted outside text, content-addressed by SHA-256. */
export type RepoFile = {
  path: string;
  sha256: string;
  content: Untrusted<string>;
  /** Optional precomputed Neuralese compact view slot keyed to this content SHA. */
  viewSlot?: { readonly view: Neuralese<string> };
};

export type TableValue = string | number | boolean | null;

/** One relational row loaded from CSV, JSON, or SQLite metadata tables. */
export type TableRow = Record<string, TableValue>;

export type ExactFilterOp = 'eq' | 'neq' | 'gt' | 'gte' | 'lt' | 'lte' | 'in' | 'includes';

export type ExactFilter = {
  field: string;
  op: ExactFilterOp;
  value: TableValue | TableValue[];
};

export type JoinSpec = {
  /** Table name in `QueryDataset.tables`. */
  table: string;
  /** File attribute to join on (`path` or `sha256`). */
  onFileField: 'path' | 'sha256';
  /** Column name in the joined table. */
  onTableField: string;
  /** Join type (default `'inner'`). */
  type?: 'inner' | 'left';
};

export type AggregateSpec = {
  op: 'count' | 'sum' | 'avg' | 'min' | 'max';
  field?: string;
  as: string;
};

export type SemanticExtractSpec = {
  name: string;
  question: string;
};

export type QueryPlan = {
  /** Optional natural-language predicate evaluated over each surviving file's content. */
  semanticPredicate?: string;
  /** Minimum decision probability for `semanticPredicate` (default `0.5`). */
  predicateFloor?: number;
  /** Exact relational filters pushed down before any semantic LLM call. */
  exactFilters?: ExactFilter[];
  /** Optional exact hash join between repository files and a metadata table. */
  join?: JoinSpec;
  /** Optional semantic field extractions over files that pass both exact and semantic filters. */
  extracts?: SemanticExtractSpec[];
  /** Optional group-by field for exact relational aggregation. */
  groupBy?: string;
  /** Exact relational aggregations computed in TypeScript over matched rows. */
  aggregates?: AggregateSpec[];
  /** Optional exact sort on any file, joined, or extracted field. */
  orderBy?: { field: string; direction?: 'asc' | 'desc' };
  /** Optional row limit applied after ordering. */
  limit?: number;
};

export type QueryDataset = {
  files: RepoFile[];
  tables?: Record<string, TableRow[]>;
};

export type MatchedRow = {
  path: string;
  sha256: string;
  joined?: TableRow;
  extracted: Record<string, string>;
};

export type AggregateRow = Record<string, TableValue>;

export type CacheStats = {
  cacheHits: number;
  cacheMisses: number;
  hitRate: number;
  extractCacheHits: number;
  extractCacheMisses: number;
  viewsUsed: number;
};

export type QueryResult = {
  rows: MatchedRow[];
  aggregates?: AggregateRow[];
  stats: CacheStats;
};
