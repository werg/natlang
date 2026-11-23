/**
 * Semantic ETL (Tier 1 Flagship A2, plans/semantic-fp/06_FLAGSHIP_APPLICATIONS.md §3 & §3.5):
 * - Synthesizes a declarative `RowMappingSpec` once from sample rows using `buildMapper.nl`, verified and repaired via
 *   `semanticRepair` against exact schema and conservation invariants.
 * - Compiles the verified spec into a pure TypeScript row transform that executes across all bulk rows with zero LLM
 *   calls (targeting >= 95% crisp execution at > 1,000 rows/s).
 * - Routes ONLY anomalous rows that fail crisp validation to `parseAnomalousRow.nl` via `semanticMap`.
 * - Verifies exact conservation invariants (row count preserved, IDs preserved, finite amounts, monetary total).
 * - Links vendor/counterparty entities across feeds using `semanticMatch` with exact blocking and `sameEntity.nl`
 *   (`readout: decision`).
 */
import { semanticMap, semanticMatch, semanticRepair } from '@natlang/node/std';
import buildMapper from './buildMapper.nl';
import parseAnomalousRow from './parseAnomalousRow.nl';
import sameEntity from './sameEntity.nl';
import type {
  CleanRow,
  ConservationReport,
  EntityPair,
  EntityRecord,
  EtlReport,
  RawRow,
  RowMappingSpec,
  TargetSchema,
} from './types.js';

export type EtlOptions = {
  /** Number of leading rows inspected when synthesizing the crisp row mapper (default `5`). */
  sampleSize?: number;
  /** Maximum `semanticRepair` iterations for `buildMapper.nl` (default `2`). */
  repairAttempts?: number;
  /** Optional known expected monetary total for the full batch (verified to within `totalTolerance`). */
  expectedTotal?: number;
  /** Optional known expected monetary total for the sample rows during mapper synthesis. */
  expectedSampleTotal?: number;
  /** Monetary conservation tolerance in major currency units (default `0.01`). */
  totalTolerance?: number;
  /** Maximum concurrent fallback / entity-match calls (default `8`). */
  concurrency?: number;
  /** Optional entity-linking inputs paired via `semanticMatch` + exact blocking. */
  entityLink?: { left: EntityRecord[]; right: EntityRecord[] };
  /** Custom exact blocker for `semanticMatch` (default `defaultEntityBlock`). */
  block?: (left: EntityRecord, right: EntityRecord) => boolean;
  /** Minimum probability of `true` for `semanticMatch` (default `0.65`). */
  entityMatchFloor?: number;
  /** Pre-supplied verified `RowMappingSpec` to skip synthesis (for specialized steady-state runs). */
  mappingSpec?: RowMappingSpec;
};

export type CrispParseResult =
  | { ok: true; row: CleanRow }
  | { ok: false; error: string };

const CURRENCY_SYMBOLS: Record<string, string> = {
  $: 'USD',
  '€': 'EUR',
  '£': 'GBP',
  '¥': 'JPY',
};

const ISO_DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const CURRENCY_CODE_RE = /^[A-Z]{3}$/;

/** Parse a single RFC-4180 delimited line (CSV or TSV) without regex backtracking or `eval`. */
function splitDelimitedLine(line: string, sep: string): string[] {
  const fields: string[] = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!;
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        current += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === sep) {
      fields.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  fields.push(current.trim());
  return fields;
}

/** Parse one raw row payload into a case-insensitive key/value record according to `spec.delimiter`. */
function parseRawRecord(raw: string, spec: RowMappingSpec): Record<string, unknown> | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  if (spec.delimiter === 'json') {
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
      return null;
    } catch {
      return null;
    }
  }
  if (spec.delimiter === 'csv' || spec.delimiter === 'tsv') {
    const sep = spec.delimiter === 'csv' ? ',' : '\t';
    const values = splitDelimitedLine(trimmed, sep);
    const cols = spec.columns ?? [];
    if (!cols.length || values.length !== cols.length) return null;
    const record: Record<string, unknown> = {};
    for (let i = 0; i < cols.length; i++) {
      record[cols[i]!] = values[i]!;
    }
    return record;
  }
  // Key-value format: `k=v; k2=v2` or `k=v | k2=v2`.
  const parts = trimmed.split(/[;|]/).map(p => p.trim()).filter(Boolean);
  if (!parts.length) return null;
  const record: Record<string, unknown> = {};
  let matched = 0;
  for (const part of parts) {
    const eq = part.indexOf('=');
    const colon = eq >= 0 ? eq : part.indexOf(':');
    if (colon <= 0) continue;
    const key = part.slice(0, colon).trim();
    const val = part.slice(colon + 1).trim();
    if (key) {
      record[key] = val;
      matched++;
    }
  }
  return matched > 0 ? record : null;
}

/** Look up the first matching alias in a parsed record (case-insensitive). */
function lookupField(record: Record<string, unknown>, aliases: readonly string[]): unknown {
  const lowerMap = new Map<string, unknown>();
  for (const [k, v] of Object.entries(record)) {
    lowerMap.set(k.trim().toLowerCase(), v);
  }
  for (const alias of aliases) {
    const key = alias.trim().toLowerCase();
    if (lowerMap.has(key)) {
      const val = lowerMap.get(key);
      if (val !== undefined && val !== null && String(val).trim() !== '') return val;
    }
  }
  return undefined;
}

/** Parse a monetary value (including symbols and US/EU grouping separators) into `{ amount, detectedCurrency }`. */
function parseAmount(value: unknown, scale: number): { amount: number; detectedCurrency?: string } | null {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return null;
    return { amount: Math.round(value * scale * 100) / 100 };
  }
  if (typeof value !== 'string') return null;
  let s = value.trim();
  if (!s) return null;
  let detectedCurrency: string | undefined;
  for (const [sym, code] of Object.entries(CURRENCY_SYMBOLS)) {
    if (s.includes(sym)) {
      detectedCurrency = code;
      s = s.replaceAll(sym, '').trim();
    }
  }
  const codeMatch = /\b(USD|EUR|GBP|JPY|CAD|AUD|CHF)\b/i.exec(s);
  if (codeMatch) {
    detectedCurrency = codeMatch[1]!.toUpperCase();
    s = s.replace(codeMatch[0], '').trim();
  }
  // Handle European decimal comma (e.g. `1.234,50` or `42,95`) vs US thousands comma (`1,234.50`).
  if (/^-?\d{1,3}(?:\.\d{3})*,\d{1,2}$/.test(s) || /^-?\d+,\d{1,2}$/.test(s)) {
    s = s.replaceAll('.', '').replace(',', '.');
  } else {
    s = s.replaceAll(',', '');
  }
  if (!/^-?\d+(?:\.\d+)?$/.test(s)) return null;
  const num = Number(s);
  if (!Number.isFinite(num)) return null;
  return { amount: Math.round(num * scale * 100) / 100, detectedCurrency };
}

/** Normalize an ISO, US (`MM/DD/YYYY`), or EU (`DD/MM/YYYY` / `DD.MM.YYYY`) date string to `YYYY-MM-DD`. */
function normalizeDate(value: unknown, format: RowMappingSpec['dateFormat']): string | null {
  if (typeof value !== 'string') return null;
  const s = value.trim();
  const iso = /^(\d{4})-(\d{2})-(\d{2})(?:[T\s].*)?$/.exec(s);
  if (iso) {
    const [, y, m, d] = iso;
    const month = Number(m), day = Number(d);
    if (month >= 1 && month <= 12 && day >= 1 && day <= 31) return `${y}-${m}-${d}`;
    return null;
  }
  const slash = /^(\d{1,2})[/.](\d{1,2})[/.](\d{4})$/.exec(s);
  if (slash) {
    const a = Number(slash[1]!), b = Number(slash[2]!), y = slash[3]!;
    let month: number, day: number;
    if (format === 'eu' || (format === 'auto' && a > 12)) {
      day = a; month = b;
    } else {
      month = a; day = b;
    }
    if (month >= 1 && month <= 12 && day >= 1 && day <= 31) {
      return `${y}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
    }
  }
  return null;
}

/** Validate a CleanRow strictly against TargetSchema invariants. */
export function validateCleanRow(row: CleanRow): string[] {
  const problems: string[] = [];
  if (!row.id || !row.id.trim()) problems.push('Row id is missing or empty.');
  if (typeof row.amount !== 'number' || !Number.isFinite(row.amount)) problems.push(`Row ${row.id}: amount is not a finite number.`);
  if (!CURRENCY_CODE_RE.test(row.currency)) problems.push(`Row ${row.id}: currency "${row.currency}" is not a 3-letter ISO code.`);
  if (!row.vendor || !row.vendor.trim()) problems.push(`Row ${row.id}: vendor is missing.`);
  if (!ISO_DATE_RE.test(row.date)) problems.push(`Row ${row.id}: date "${row.date}" is not YYYY-MM-DD.`);
  if (!row.category || !row.category.trim()) problems.push(`Row ${row.id}: category is missing.`);
  return problems;
}

/** Compile a verified `RowMappingSpec` into a fast, zero-LLM-call deterministic TypeScript row transform. */
export function compileRowTransform(spec: RowMappingSpec, _schema: TargetSchema): (row: RawRow) => CrispParseResult {
  const scale = Number.isFinite(spec.amountScale) && spec.amountScale > 0 ? spec.amountScale : 1;
  const fallbackCurrency = (spec.defaultCurrency || 'USD').trim().toUpperCase();

  return (row: RawRow): CrispParseResult => {
    const record = parseRawRecord(String(row.raw), spec);
    if (!record) return { ok: false, error: `Could not parse raw payload as ${spec.delimiter}.` };

    const rawId = lookupField(record, spec.fieldMap.id);
    const id = rawId !== undefined ? String(rawId).trim() : row.id.trim();
    if (!id) return { ok: false, error: 'Missing row id.' };

    const rawAmount = lookupField(record, spec.fieldMap.amount);
    const parsedAmount = parseAmount(rawAmount, scale);
    if (!parsedAmount) return { ok: false, error: `Invalid amount value: ${String(rawAmount)}.` };

    const rawCurrency = lookupField(record, spec.fieldMap.currency);
    const currencyCandidate = rawCurrency !== undefined
      ? String(rawCurrency).trim().toUpperCase()
      : (parsedAmount.detectedCurrency ?? fallbackCurrency);
    const currency = CURRENCY_SYMBOLS[currencyCandidate] ?? currencyCandidate;
    if (!CURRENCY_CODE_RE.test(currency)) return { ok: false, error: `Invalid currency code: ${currency}.` };

    const rawVendor = lookupField(record, spec.fieldMap.vendor);
    const vendor = rawVendor !== undefined ? String(rawVendor).trim() : '';
    if (!vendor) return { ok: false, error: 'Missing vendor field.' };

    const rawDate = lookupField(record, spec.fieldMap.date);
    const date = normalizeDate(rawDate, spec.dateFormat);
    if (!date) return { ok: false, error: `Invalid date value: ${String(rawDate)}.` };

    const rawCategory = lookupField(record, spec.fieldMap.category);
    const category = rawCategory !== undefined ? String(rawCategory).trim().toLowerCase() : 'uncategorized';

    const clean: CleanRow = {
      id,
      amount: parsedAmount.amount,
      currency,
      vendor,
      date,
      category,
      source: 'crisp',
    };
    const issues = validateCleanRow(clean);
    if (issues.length > 0) return { ok: false, error: issues.join(' ') };
    return { ok: true, row: clean };
  };
}

/** Exact conservation and schema invariant checker over an entire ETL output batch. */
export function verifyConservation(
  inputRows: readonly RawRow[],
  outputRows: readonly CleanRow[],
  expectedTotal?: number,
  tolerance = 0.01,
): ConservationReport {
  const problems: string[] = [];
  if (outputRows.length !== inputRows.length) {
    problems.push(`Row count mismatch: input had ${inputRows.length} rows, output has ${outputRows.length}.`);
  }
  const outputIds = new Set(outputRows.map(r => r.id));
  for (const input of inputRows) {
    if (!input.id || !input.id.trim()) {
      problems.push('Input batch contains an empty row id.');
    } else if (!outputIds.has(input.id)) {
      problems.push(`Input row id "${input.id}" was lost or altered in output.`);
    }
  }
  let sum = 0;
  for (const row of outputRows) {
    const rowProblems = validateCleanRow(row);
    problems.push(...rowProblems);
    if (Number.isFinite(row.amount)) sum += row.amount;
  }
  const actualTotal = Math.round(sum * 100) / 100;
  if (expectedTotal !== undefined && Math.abs(actualTotal - expectedTotal) > tolerance) {
    problems.push(`Monetary total drift: expected ${expectedTotal.toFixed(2)}, got ${actualTotal.toFixed(2)}.`);
  }
  return {
    ok: problems.length === 0,
    inputRows: inputRows.length,
    outputRows: outputRows.length,
    expectedTotal: expectedTotal ?? null,
    actualTotal,
    problems,
  };
}

const LEGAL_SUFFIX_RE = /\b(inc|incorporated|llc|ltd|limited|corp|corporation|gmbh|ag|sa|sarl|bv|nv|plc|co|company|holdings|group)\b/g;

/** Extract normalized distinctive tokens from an organization name for exact blocking. */
function entityTokens(name: string): string[] {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(LEGAL_SUFFIX_RE, ' ')
    .split(/\s+/)
    .filter(tok => tok.length >= 2);
}

/** Exact pre-filter blocker for `semanticMatch`: prunes obvious non-matches before invoking `sameEntity.nl`. */
export function defaultEntityBlock(left: EntityRecord, right: EntityRecord): boolean {
  if (left.taxId && right.taxId) {
    return left.taxId.replace(/\W+/g, '').toLowerCase() === right.taxId.replace(/\W+/g, '').toLowerCase();
  }
  if (left.domain && right.domain && left.domain.toLowerCase() === right.domain.toLowerCase()) {
    return true;
  }
  if (left.country && right.country && left.country.toUpperCase() !== right.country.toUpperCase()) {
    return false;
  }
  const leftTokens = entityTokens(left.name);
  const rightTokens = new Set(entityTokens(right.name));
  if (leftTokens.some(tok => rightTokens.has(tok))) return true;
  // Fallback prefix check on the primary token (handles abbreviations like "Intl" vs "International").
  const l0 = leftTokens[0] ?? '';
  const r0 = entityTokens(right.name)[0] ?? '';
  return l0.length >= 3 && r0.length >= 3 && l0.slice(0, 3) === r0.slice(0, 3);
}

/** Verify a candidate `RowMappingSpec` against the sample rows (used inside `semanticRepair`). */
function checkMappingOnSamples(
  spec: RowMappingSpec,
  samples: readonly RawRow[],
  schema: TargetSchema,
  expectedSampleTotal?: number,
  tolerance = 0.01,
): string[] {
  if (!spec || typeof spec !== 'object' || !spec.fieldMap) {
    return ['RowMappingSpec is missing fieldMap.'];
  }
  const transform = compileRowTransform(spec, schema);
  const parsed: CleanRow[] = [];
  const problems: string[] = [];
  for (const sample of samples) {
    const result = transform(sample);
    if (!result.ok) {
      problems.push(`Sample ${sample.id} failed crisp parse: ${result.error}`);
    } else {
      parsed.push(result.row);
    }
  }
  if (problems.length > 0) return problems;
  const conservation = verifyConservation(samples, parsed, expectedSampleTotal, tolerance);
  return conservation.problems;
}

/**
 * Run Semantic ETL end-to-end:
 * 1. Synthesize and verify a crisp `RowMappingSpec` on sample rows (`buildMapper.nl` + `semanticRepair`).
 * 2. Execute the compiled pure-TypeScript transform over all bulk rows (0 LLM calls per normal row).
 * 3. Fall back to `parseAnomalousRow.nl` ONLY for anomalous rows that failed crisp validation.
 * 4. Verify exact conservation invariants and optionally link entities via `semanticMatch` + exact blocking.
 */
export async function runSemanticEtl(
  rows: RawRow[],
  schema: TargetSchema,
  options: EtlOptions = {},
): Promise<EtlReport> {
  const concurrency = options.concurrency ?? 8;
  const tolerance = options.totalTolerance ?? 0.01;
  const sampleSize = Math.min(options.sampleSize ?? 5, rows.length);
  const samples = rows.slice(0, sampleSize);

  let mappingSpec: RowMappingSpec;
  let mappingAttempts = 0;

  if (options.mappingSpec) {
    mappingSpec = options.mappingSpec;
  } else {
    const initial = await buildMapper(samples, schema, []);
    const repaired = await semanticRepair(initial, {
      check: spec => checkMappingOnSamples(spec, samples, schema, options.expectedSampleTotal, tolerance),
      fix: (_spec, problems) => buildMapper(samples, schema, problems),
      attempts: options.repairAttempts ?? 2,
    });
    mappingSpec = repaired.value;
    mappingAttempts = repaired.attempts;
  }

  const transform = compileRowTransform(mappingSpec, schema);
  const resolved: (CleanRow | undefined)[] = new Array(rows.length);
  const anomalous: { index: number; row: RawRow; error: string }[] = [];
  let crispRows = 0;

  // Bulk crisp pass: synchronous pure TypeScript, zero model calls.
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i]!;
    const result = transform(row);
    if (result.ok) {
      resolved[i] = result.row;
      crispRows++;
    } else {
      anomalous.push({ index: i, row, error: result.error });
    }
  }

  // Semantic fallback pass: invoked ONLY on rows that failed the crisp schema check.
  if (anomalous.length > 0) {
    const fallbackRows = await semanticMap(
      anomalous,
      async entry => {
        const parsed = await parseAnomalousRow(entry.row, schema, entry.error);
        return { index: entry.index, row: { ...parsed, id: parsed.id || entry.row.id, source: 'semantic' as const } };
      },
      { concurrency },
    );
    for (const entry of fallbackRows) {
      resolved[entry.index] = entry.row;
    }
  }

  const cleanRows = resolved.filter((r): r is CleanRow => r !== undefined);
  const conservation = verifyConservation(rows, cleanRows, options.expectedTotal, tolerance);

  let linkedEntities: EntityPair[] = [];
  let unmatchedLeft: EntityRecord[] = [];
  let unmatchedRight: EntityRecord[] = [];

  if (options.entityLink) {
    const matchResult = await semanticMatch(options.entityLink.left, options.entityLink.right, {
      match: (left, right) => sameEntity(left, right),
      block: options.block ?? defaultEntityBlock,
      floor: options.entityMatchFloor ?? 0.65,
      concurrency,
    });
    linkedEntities = matchResult.pairs.map(p => ({ left: p.left, right: p.right, confidence: p.confidence }));
    unmatchedLeft = matchResult.unmatchedLeft;
    unmatchedRight = matchResult.unmatchedRight;
  }

  const semanticFallbackRows = anomalous.length;
  return {
    rows: cleanRows,
    crispRows,
    semanticFallbackRows,
    crispShare: rows.length > 0 ? crispRows / rows.length : 1,
    conservation,
    linkedEntities,
    unmatchedLeft,
    unmatchedRight,
    mappingAttempts,
    mappingSpec,
  };
}
