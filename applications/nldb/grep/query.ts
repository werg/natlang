/**
 * Sem-Grep-SQL (WS6 §3 A8): cross-cutting semantic + relational queries over repositories, docs, and CSV/SQLite tables.
 * Exact hash joins, predicate pushdown, CSV parsing, grouping, sorting, and aggregations stay in TypeScript;
 * content-addressed caching keyed by (sha256, predicate) avoids repeating semantic calls on unchanged files;
 * uncached files run concurrently through `semanticFilter` and `semanticMap` with optional `Neuralese<T>` view hooks.
 */
import { createHash } from 'node:crypto';
import { untrusted } from '@natlang/node';
import { semanticFilter, semanticMap } from '@natlang/node/std';
import extractField from './extractField.nl';
import matchesPredicate from './matchesPredicate.nl';
import type {
  AggregateRow,
  AggregateSpec,
  ExactFilter,
  MatchedRow,
  QueryDataset,
  QueryPlan,
  QueryResult,
  RepoFile,
  TableRow,
  TableValue,
} from './types.js';

/** Crisp carrier holding an opaque `Neuralese<T>` view reference so callers never branch on a `Neuralese` value directly. */
export type NeuraleseViewSlot<T = string> = {
  readonly view: Neuralese<T>;
};

/** Optional Neuralese<T> view hook: lets a caller supply content-SHA views and a view-native predicate judge. */
export type NeuraleseViewHook = {
  getView?: (file: RepoFile) => NeuraleseViewSlot<string> | undefined | Promise<NeuraleseViewSlot<string> | undefined>;
  judgeView?: (path: string, view: Neuralese<string>, predicate: string) => Promise<boolean>;
};

export type QueryOptions = {
  /** Persistent or shared content-addressed predicate + extraction cache across queries. */
  cache?: PredicateCache;
  /** Maximum concurrent semantic calls for uncached files (default `8`). */
  concurrency?: number;
  /** Optional Neuralese<string> view provider / consumer hook keyed by file SHA. */
  neuraleseView?: NeuraleseViewHook;
};

/** Compute lowercase hex SHA-256 of UTF-8 file text. */
export function sha256Hex(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

/** Construct a content-addressed RepoFile whose content is fenced as `Untrusted<string>`. */
export function createRepoFile(path: string, content: string, source?: string): RepoFile {
  return {
    path,
    sha256: sha256Hex(content),
    content: untrusted(content, source ?? path),
  };
}

/** Content-addressed cache keyed by `(sha256, predicate)` for predicates and `(sha256, question)` for extractions. */
export class PredicateCache {
  private readonly predicates = new Map<string, boolean>();
  private readonly extracts = new Map<string, string>();

  private static key(sha256: string, query: string): string {
    return `${sha256}\0${query.trim()}`;
  }

  get(sha256: string, predicate: string): boolean | undefined {
    return this.predicates.get(PredicateCache.key(sha256, predicate));
  }

  set(sha256: string, predicate: string, matched: boolean): void {
    this.predicates.set(PredicateCache.key(sha256, predicate), matched);
  }

  getExtract(sha256: string, question: string): string | undefined {
    return this.extracts.get(PredicateCache.key(sha256, question));
  }

  setExtract(sha256: string, question: string, answer: string): void {
    this.extracts.set(PredicateCache.key(sha256, question), answer);
  }

  get size(): number {
    return this.predicates.size;
  }

  clear(): void {
    this.predicates.clear();
    this.extracts.clear();
  }
}

function coerceCell(raw: string): TableValue {
  const trimmed = raw.trim();
  if (trimmed === '' || trimmed.toLowerCase() === 'null') return null;
  if (trimmed === 'true') return true;
  if (trimmed === 'false') return false;
  if (/^-?\d+(?:\.\d+)?$/.test(trimmed)) return Number(trimmed);
  return trimmed;
}

function splitCsvLine(line: string): string[] {
  const cells: string[] = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!;
    if (inQuotes) {
      if (ch === '"' && line[i + 1] === '"') {
        current += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        current += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      cells.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  cells.push(current);
  return cells;
}

/** Parse RFC-4180 style CSV text with header row into typed relational `TableRow[]`. */
export function parseCsv(csv: string): TableRow[] {
  const lines = csv.replace(/\r\n/g, '\n').split('\n').filter(line => line.trim().length > 0);
  if (lines.length < 2) return [];
  const headers = splitCsvLine(lines[0]!).map(h => h.trim());
  const rows: TableRow[] = [];
  for (let r = 1; r < lines.length; r++) {
    const values = splitCsvLine(lines[r]!);
    const row: TableRow = {};
    for (let c = 0; c < headers.length; c++) {
      row[headers[c]!] = coerceCell(values[c] ?? '');
    }
    rows.push(row);
  }
  return rows;
}

type CandidateRow = {
  file: RepoFile;
  joined?: TableRow;
};

function readField(row: CandidateRow | MatchedRow, field: string): TableValue {
  if (field === 'path') return 'file' in row ? row.file.path : row.path;
  if (field === 'sha256') return 'file' in row ? row.file.sha256 : row.sha256;
  if ('extracted' in row && field in row.extracted) return row.extracted[field] ?? null;
  if (row.joined && field in row.joined) return row.joined[field] ?? null;
  return null;
}

function matchesExactFilter(row: CandidateRow, filter: ExactFilter): boolean {
  const actual = readField(row, filter.field);
  const target = filter.value;
  switch (filter.op) {
    case 'eq': return actual === target;
    case 'neq': return actual !== target;
    case 'gt': return typeof actual === 'number' && typeof target === 'number' && actual > target;
    case 'gte': return typeof actual === 'number' && typeof target === 'number' && actual >= target;
    case 'lt': return typeof actual === 'number' && typeof target === 'number' && actual < target;
    case 'lte': return typeof actual === 'number' && typeof target === 'number' && actual <= target;
    case 'in': return Array.isArray(target) && target.includes(actual);
    case 'includes': return typeof actual === 'string' && typeof target === 'string' && actual.includes(target);
  }
}

function joinFiles(dataset: QueryDataset, plan: QueryPlan): CandidateRow[] {
  if (!plan.join) return dataset.files.map(file => ({ file }));
  const { table, onFileField, onTableField, type = 'inner' } = plan.join;
  const rows = dataset.tables?.[table] ?? [];
  const index = new Map<string, TableRow[]>();
  for (const row of rows) {
    const key = String(row[onTableField] ?? '');
    const bucket = index.get(key);
    if (bucket) bucket.push(row);
    else index.set(key, [row]);
  }
  const joined: CandidateRow[] = [];
  for (const file of dataset.files) {
    const key = file[onFileField];
    const matches = index.get(key);
    if (matches && matches.length > 0) {
      for (const match of matches) joined.push({ file, joined: match });
    } else if (type === 'left') {
      joined.push({ file });
    }
  }
  return joined;
}

function computeAggregates(rows: MatchedRow[], groupBy: string | undefined, specs: AggregateSpec[]): AggregateRow[] {
  const groups = new Map<string, { keyValue: TableValue; items: MatchedRow[] }>();
  for (const row of rows) {
    const keyValue = groupBy ? readField(row, groupBy) : '__all__';
    const key = JSON.stringify(keyValue);
    const bucket = groups.get(key);
    if (bucket) bucket.items.push(row);
    else groups.set(key, { keyValue, items: [row] });
  }
  if (!groupBy && groups.size === 0) {
    groups.set('__all__', { keyValue: null, items: [] });
  }
  const output: AggregateRow[] = [];
  for (const { keyValue, items } of groups.values()) {
    const aggRow: AggregateRow = {};
    if (groupBy) aggRow[groupBy] = keyValue;
    for (const spec of specs) {
      if (spec.op === 'count') {
        aggRow[spec.as] = items.length;
        continue;
      }
      const nums = items
        .map(item => (spec.field ? readField(item, spec.field) : null))
        .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
      if (nums.length === 0) {
        aggRow[spec.as] = null;
      } else if (spec.op === 'sum') {
        aggRow[spec.as] = nums.reduce((acc, n) => acc + n, 0);
      } else if (spec.op === 'avg') {
        aggRow[spec.as] = nums.reduce((acc, n) => acc + n, 0) / nums.length;
      } else if (spec.op === 'min') {
        aggRow[spec.as] = nums.reduce((acc, n) => (n < acc ? n : acc), nums[0]!);
      } else if (spec.op === 'max') {
        aggRow[spec.as] = nums.reduce((acc, n) => (n > acc ? n : acc), nums[0]!);
      }
    }
    output.push(aggRow);
  }
  return output;
}

function compareValues(a: TableValue, b: TableValue): number {
  if (a === b) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  if (typeof a === 'number' && typeof b === 'number') return a - b;
  return String(a).localeCompare(String(b));
}

/**
 * Execute a hybrid relational + semantic query over repository files and tables.
 * Exact joins and filters push down before any LLM call; semantic predicates are cached by `(sha256, predicate)`.
 */
export async function runQuery(
  dataset: QueryDataset,
  plan: QueryPlan,
  options: QueryOptions = {},
): Promise<QueryResult> {
  const cache = options.cache ?? new PredicateCache();
  const concurrency = options.concurrency ?? 8;
  const viewHook = options.neuraleseView;

  // 1. Exact relational join + exact predicate pushdown (0 LLM calls).
  let candidates = joinFiles(dataset, plan);
  if (plan.exactFilters && plan.exactFilters.length > 0) {
    candidates = candidates.filter(row => plan.exactFilters!.every(f => matchesExactFilter(row, f)));
  }

  let cacheHits = 0;
  let cacheMisses = 0;
  let viewsUsed = 0;

  // 2. Content-addressed semantic predicate filter.
  let surviving = candidates;
  const predicate = plan.semanticPredicate?.trim();
  if (predicate) {
    const uncachedBySha = new Map<string, RepoFile>();
    const preCached = new Set<string>();
    for (const candidate of candidates) {
      const sha = candidate.file.sha256;
      const hit = cache.get(sha, predicate);
      if (hit !== undefined) {
        cacheHits++;
        preCached.add(sha);
      } else if (!uncachedBySha.has(sha)) {
        uncachedBySha.set(sha, candidate.file);
      }
    }

    const uncachedFiles = [...uncachedBySha.values()];
    cacheMisses = uncachedFiles.length;

    if (uncachedFiles.length > 0) {
      const matchedUncached = await semanticFilter(
        uncachedFiles,
        async file => {
          const slot = file.viewSlot ?? (viewHook?.getView ? await viewHook.getView(file) : undefined);
          if (slot && viewHook?.judgeView) {
            viewsUsed++;
            return viewHook.judgeView(file.path, slot.view, predicate);
          }
          return matchesPredicate(file.path, file.content, predicate);
        },
        {
          concurrency,
          ...(plan.predicateFloor !== undefined ? { floor: plan.predicateFloor } : {}),
        },
      );
      const matchedShas = new Set(matchedUncached.map(file => file.sha256));
      for (const file of uncachedFiles) {
        cache.set(file.sha256, predicate, matchedShas.has(file.sha256));
      }
    }

    // Count duplicate-SHA rows in the same batch that benefited from deduplication as cache hits.
    const dupHits = candidates.length - cacheHits - cacheMisses;
    if (dupHits > 0) cacheHits += dupHits;

    surviving = candidates.filter(row => cache.get(row.file.sha256, predicate) === true);
  }

  // 3. Content-addressed semantic field extraction over surviving rows.
  let extractCacheHits = 0;
  let extractCacheMisses = 0;
  const extracts = plan.extracts ?? [];
  const extractedBySha = new Map<string, Record<string, string>>();

  if (extracts.length > 0 && surviving.length > 0) {
    const uniqueFiles = new Map<string, RepoFile>();
    for (const row of surviving) {
      if (!uniqueFiles.has(row.file.sha256)) uniqueFiles.set(row.file.sha256, row.file);
    }
    const workItems: { file: RepoFile; name: string; question: string }[] = [];
    for (const file of uniqueFiles.values()) {
      const record: Record<string, string> = {};
      extractedBySha.set(file.sha256, record);
      for (const spec of extracts) {
        const cached = cache.getExtract(file.sha256, spec.question);
        if (cached !== undefined) {
          record[spec.name] = cached;
          extractCacheHits++;
        } else {
          workItems.push({ file, name: spec.name, question: spec.question });
        }
      }
    }
    extractCacheMisses = workItems.length;
    if (workItems.length > 0) {
      const answers = await semanticMap(
        workItems,
        item => extractField(item.file.path, item.file.content, item.question),
        { concurrency },
      );
      workItems.forEach((item, index) => {
        const answer = answers[index]!;
        cache.setExtract(item.file.sha256, item.question, answer);
        extractedBySha.get(item.file.sha256)![item.name] = answer;
      });
    }
  }

  let matchedRows: MatchedRow[] = surviving.map(row => ({
    path: row.file.path,
    sha256: row.file.sha256,
    ...(row.joined ? { joined: row.joined } : {}),
    extracted: { ...(extractedBySha.get(row.file.sha256) ?? {}) },
  }));

  // 4. Exact ordering, limiting, and relational aggregations in TypeScript.
  if (plan.orderBy) {
    const { field, direction = 'asc' } = plan.orderBy;
    const factor = direction === 'desc' ? -1 : 1;
    matchedRows = [...matchedRows].sort((a, b) => factor * compareValues(readField(a, field), readField(b, field)));
  }
  if (plan.limit !== undefined) {
    matchedRows = matchedRows.slice(0, Math.max(0, plan.limit));
  }

  const aggregates = plan.aggregates && plan.aggregates.length > 0
    ? computeAggregates(matchedRows, plan.groupBy, plan.aggregates)
    : undefined;

  const totalPredicates = cacheHits + cacheMisses;
  return {
    rows: matchedRows,
    ...(aggregates ? { aggregates } : {}),
    stats: {
      cacheHits,
      cacheMisses,
      hitRate: totalPredicates === 0 ? 1 : cacheHits / totalPredicates,
      extractCacheHits,
      extractCacheMisses,
      viewsUsed,
    },
  };
}
