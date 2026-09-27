/** Answer checks used by generated and dataset-backed teacher cases. */
export type OracleLevel = 'exact' | 'normalized' | 'span' | 'judged';
export type OracleSpec = OracleLevel | { level: OracleLevel; alternates?: unknown[];
  threshold?: number; rubric?: string; [key: string]: unknown };
export type OracleVerdict = { accepted: boolean; level: OracleLevel; score?: number; verdict?: string };

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value as Record<string, unknown>)
    .sort(([a], [b]) => a.localeCompare(b)).map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(',')}}`;
  return JSON.stringify(value);
}
function normalizeText(value: unknown): string {
  const text = String(value ?? '').normalize('NFKC').trim().toLocaleLowerCase('en-US').replace(/\s+/g, ' ');
  const number = text.replace(/(?<=\d),(?=\d{3}(?:\D|$))/g, '');
  if (/^[+-]?(?:\d+\.?\d*|\.\d+)$/.test(number)) return `number:${Number(number)}`;
  const date = /^(\d{1,2})\/(\d{1,2})\/(\d{4})$/.exec(text);
  if (date) return `date:${date[3]}-${date[1]!.padStart(2, '0')}-${date[2]!.padStart(2, '0')}`;
  if (/^\d{4}-\d{2}-\d{2}$/.test(text)) return `date:${text}`;
  return text;
}
function normalized(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(normalized).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value as Record<string, unknown>)
    .sort(([a], [b]) => a.localeCompare(b)).map(([key, child]) => `${JSON.stringify(key)}:${normalized(child)}`).join(',')}}`;
  return normalizeText(value);
}
function words(value: unknown): string[] {
  return String(Array.isArray(value) ? value.join(' ') : value ?? '').normalize('NFKC')
    .toLocaleLowerCase('en-US').match(/[\p{L}\p{N}]+/gu) ?? [];
}
/** Multiset token F1; repeated words count only when they occur on both sides. */
export function spanF1(actual: unknown, expected: unknown): number {
  const left = words(actual), right = words(expected);
  if (!left.length || !right.length) return left.length === right.length ? 1 : 0;
  const counts = new Map<string, number>();
  for (const word of right) counts.set(word, (counts.get(word) ?? 0) + 1);
  let shared = 0;
  for (const word of left) if ((counts.get(word) ?? 0) > 0) { shared++; counts.set(word, counts.get(word)! - 1); }
  return 2 * shared / (left.length + right.length);
}

/** A judge's verdict is supplied by the collection caller; no case can self-certify its own prose. */
export async function checkOracle(actual: unknown, expected: unknown, oracle: OracleSpec = 'exact',
  judge?: (input: { actual: unknown; expected: unknown; rubric: string }) => Promise<{ accepted: boolean; verdict: string }>):
  Promise<OracleVerdict> {
  const spec = typeof oracle === 'string' ? { level: oracle } : oracle;
  const { level } = spec;
  if (!['exact', 'normalized', 'span', 'judged'].includes(level)) throw new RangeError(`unknown oracle level: ${level}`);
  const candidates = [expected, ...('alternates' in spec ? spec.alternates ?? [] : [])];
  if (level === 'exact') return { accepted: candidates.some(candidate => canonical(actual) === canonical(candidate)), level };
  if (level === 'normalized') return { accepted: candidates.some(candidate => normalized(actual) === normalized(candidate)), level };
  if (level === 'span') {
    const threshold = 'threshold' in spec ? spec.threshold ?? 0.5 : 0.5;
    if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new RangeError('span threshold must be between 0 and 1');
    const score = Math.max(...candidates.map(candidate => spanF1(actual, candidate)));
    return { accepted: score >= threshold, level, score };
  }
  if (!judge) return { accepted: false, level, verdict: 'no judge supplied' };
  const verdict = await judge({ actual, expected, rubric: 'rubric' in spec ? spec.rubric ?? '' : '' });
  return { accepted: verdict.accepted === true, level, verdict: verdict.verdict };
}
