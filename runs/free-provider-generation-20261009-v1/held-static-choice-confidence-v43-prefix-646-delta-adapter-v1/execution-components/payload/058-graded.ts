/** Host-only graded scores for skill episodes drawn from real data (no model judge). */
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync, chmodSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { isAbsolute, join, normalize } from 'node:path';

export const GRADED_KINDS = ['sql-result-f1', 'python-tests', 'answer-token-f1', 'ranking-ndcg', 'assignment-accuracy', 'call-f1', 'choice-brier', 'binary-brier', 'ordinal-rps', 'compaction-utility'] as const;
export type GradedKind = typeof GRADED_KINDS[number];
export type GradedMetric = { schema: 'natlang.skill-graded/1'; kind: GradedKind; database_root?: string; sandbox_image?: string };
export type GradedScore = { quality: number; gates: Record<string, boolean>; detail?: Record<string, unknown> };

const invalid = (gate: string, detail?: Record<string, unknown>): GradedScore => ({ quality: 0, gates: { [gate]: false }, detail });

/** Runs read-only queries in a child process so that a runaway query cannot stall the host. */
const RUNNER = `
const { DatabaseSync } = require('node:sqlite');
const [path, sql, limit] = JSON.parse(process.argv[1]);
try {
  const db = new DatabaseSync(path, { readOnly: true });
  const statement = db.prepare(sql);
  const rows = []; for (const row of statement.iterate()) { rows.push(Object.values(row)); if (rows.length > limit) break; }
  process.stdout.write(JSON.stringify({ rows }));
} catch (error) { process.stdout.write(JSON.stringify({ error: String(error && error.message || error) })); }
`;

export function runReadOnlyQuery(path: string, sql: string, options: { timeoutMs?: number; limit?: number } = {}):
  { rows?: unknown[][]; error?: string } {
  const result = spawnSync(process.execPath, ['--no-warnings', '-e', RUNNER, JSON.stringify([path, sql, options.limit ?? 5000])],
    { timeout: options.timeoutMs ?? 5000, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  if (result.error || result.status !== 0) return { error: result.error ? String(result.error.message) : `exit ${result.status}` };
  try { return JSON.parse(result.stdout); } catch { return { error: 'unreadable query output' }; }
}

/** Rows compare as multisets; integral floats and integers are the same value. */
function key(row: unknown[]): string {
  return JSON.stringify(row.map(cell => typeof cell === 'number' && Number.isInteger(cell) ? cell
    : typeof cell === 'number' ? Number(cell.toPrecision(12)) : cell));
}

export function multisetF1(predicted: unknown[][], gold: unknown[][]): { f1: number; precision: number; recall: number } {
  if (!predicted.length && !gold.length) return { f1: 1, precision: 1, recall: 1 };
  const counts = new Map<string, number>();
  for (const row of gold) counts.set(key(row), (counts.get(key(row)) ?? 0) + 1);
  let hit = 0;
  for (const row of predicted) { const k = key(row), n = counts.get(k) ?? 0; if (n > 0) { hit++; counts.set(k, n - 1); } }
  const precision = predicted.length ? hit / predicted.length : 0, recall = gold.length ? hit / gold.length : 0;
  return { f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0, precision, recall };
}

function extractSql(value: unknown): string | undefined {
  let text = typeof value === 'string' ? value : value && typeof value === 'object' && typeof (value as any).sql === 'string' ? (value as any).sql : undefined;
  if (text === undefined) return undefined;
  const fenced = /```(?:sql)?\s*([\s\S]*?)```/i.exec(text);
  if (fenced) text = fenced[1]!;
  return text.trim().replace(/;\s*$/, '');
}

/** Pinned, network-less sandbox for model-written code. */
export const PYTHON_SANDBOX_IMAGE = 'python@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016';
const PYTHON_RUNNER = `
import importlib, json, sys, traceback
sys.path.insert(0, '/work')
result = {"passed": 0, "failed": 0, "errors": []}
try:
    tests = importlib.import_module('test_solution')
except BaseException as error:
    print(json.dumps({"import_error": type(error).__name__ + ": " + str(error)[:300]})); sys.exit(0)
for name in sorted(n for n in dir(tests) if n.startswith('test')):
    fn = getattr(tests, name)
    if not callable(fn): continue
    try:
        fn(); result["passed"] += 1
    except BaseException as error:
        result["failed"] += 1; result["errors"].append(name + ": " + type(error).__name__)
print(json.dumps(result))
`;

function extractCode(value: unknown): string | undefined {
  let text = typeof value === 'string' ? value : value && typeof value === 'object' && typeof (value as any).code === 'string' ? (value as any).code : undefined;
  if (text === undefined) return undefined;
  const fenced = /```(?:python|py)?\s*([\s\S]*?)```/i.exec(text);
  return (fenced ? fenced[1]! : text).trim();
}

/** Fraction of the reference unit tests that the returned Python module passes, run in the sandbox. */
export function pythonTestScore(code: string, tests: string, options: { image?: string; timeoutMs?: number } = {}): GradedScore {
  const work = mkdtempSync(join(tmpdir(), 'natlang-python-tests-'));
  try {
    writeFileSync(join(work, 'solution.py'), code); writeFileSync(join(work, 'test_solution.py'), tests);
    writeFileSync(join(work, 'runner.py'), PYTHON_RUNNER);
    chmodSync(work, 0o755);
    const run = spawnSync('docker', ['run', '--rm', '--network', 'none', '--memory', '512m', '--cpus', '1', '--pids-limit', '64',
      '--read-only', '--tmpfs', '/tmp:size=64m', '--user', '65534:65534', '-v', `${work}:/work:ro`,
      options.image ?? PYTHON_SANDBOX_IMAGE, 'timeout', '20', 'python', '-B', '/work/runner.py'],
      { encoding: 'utf8', timeout: options.timeoutMs ?? 60000, maxBuffer: 4 * 1024 * 1024 });
    if (run.error) return invalid('sandbox', { error: String(run.error.message) });
    const last = run.stdout.trim().split('\n').pop() ?? '';
    let parsed: any; try { parsed = JSON.parse(last); } catch { return invalid('runs', { status: run.status, stderr: run.stderr.slice(-300) }); }
    if (parsed.import_error) return invalid('imports', { error: parsed.import_error });
    const total = parsed.passed + parsed.failed;
    if (!total) return invalid('has_tests');
    return { quality: parsed.passed / total, gates: { runs: true, all_pass: parsed.failed === 0 }, detail: { passed: parsed.passed, total, errors: parsed.errors.slice(0, 10) } };
  } finally { rmSync(work, { recursive: true, force: true }); }
}

/** SQuAD-style answer normalisation: lower case, no punctuation or articles, single spaces. */
function answerTokens(text: string): string[] {
  return text.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\b(a|an|the)\b/g, ' ').split(/\s+/).filter(Boolean);
}

export function tokenF1(predicted: string, gold: string): number {
  const p = answerTokens(predicted), g = answerTokens(gold);
  if (!p.length || !g.length) return p.length === g.length ? 1 : 0;
  return multisetF1(p.map(token => [token]), g.map(token => [token])).f1;
}

/** Binary-relevance NDCG of a ranked list; duplicates count once, at their first rank. */
export function ndcg(ranking: string[], relevant: string[]): number {
  const wanted = new Set(relevant), seen = new Set<string>();
  let dcg = 0;
  ranking.forEach((item, index) => { if (wanted.has(item) && !seen.has(item)) dcg += 1 / Math.log2(index + 2); seen.add(item); });
  let ideal = 0;
  for (let index = 0; index < wanted.size; index++) ideal += 1 / Math.log2(index + 2);
  return ideal ? dcg / ideal : 0;
}

/** A returned value that may arrive as JSON text, optionally fenced. */
function structured(value: unknown): unknown {
  if (typeof value !== 'string') return value;
  const fenced = /```(?:json)?\s*([\s\S]*?)```/i.exec(value);
  try { return JSON.parse(fenced ? fenced[1]! : value); } catch { return undefined; }
}

/** One atom per call name and per argument binding, compared as multisets. */
function callAtoms(calls: unknown): string[][] | undefined {
  if (!Array.isArray(calls)) return undefined;
  const atoms: string[][] = [];
  for (const call of calls) {
    if (!call || typeof call !== 'object' || typeof (call as any).name !== 'string') return undefined;
    const name = (call as any).name as string, args = (call as any).arguments ?? {};
    if (!args || typeof args !== 'object' || Array.isArray(args)) return undefined;
    atoms.push([name]);
    for (const key of Object.keys(args).sort()) atoms.push([name, key, JSON.stringify(args[key])]);
  }
  return atoms;
}

/** Probabilities over the options: an object of label weights (optionally under `probabilities`), or one label as certainty. */
function choiceDistribution(value: unknown, options: string[]): Record<string, number> | undefined {
  const parsed = typeof value === 'string' && options.includes(value.trim()) ? value.trim() : structured(value);
  if (typeof parsed === 'string') return options.includes(parsed) ? Object.fromEntries(options.map(o => [o, o === parsed ? 1 : 0])) : undefined;
  const weights = parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? ((parsed as any).probabilities ?? parsed) : undefined;
  if (!weights || typeof weights !== 'object' || Object.keys(weights).some(key => !options.includes(key))) return undefined;
  const raw = options.map(o => weights[o] ?? 0);
  if (!raw.every(w => typeof w === 'number' && Number.isFinite(w) && w >= 0)) return undefined;
  const total = raw.reduce((a, b) => a + b, 0);
  return total > 0 ? Object.fromEntries(options.map((o, i) => [o, raw[i]! / total])) : undefined;
}

/** A probability in [0, 1]: a number, a numeric string, `{ probability }`, or a boolean as certainty. */
function probabilityOf(value: unknown): number | undefined {
  const parsed = structured(value) ?? value;
  const raw = typeof parsed === 'boolean' ? (parsed ? 1 : 0) : typeof parsed === 'number' ? parsed
    : typeof parsed === 'string' && /^\s*[01]?(\.\d+)?\s*$/.test(parsed) ? Number(parsed)
    : parsed && typeof parsed === 'object' ? (parsed as any).probability ?? (parsed as any).noul : undefined;
  return typeof raw === 'number' && Number.isFinite(raw) && raw >= 0 && raw <= 1 ? raw : undefined;
}

/** A distribution over `k` ordered levels from probabilities (by level name or index), a level, or a fractional score. */
export function ordinalDistribution(value: unknown, levels: string[]): number[] | undefined {
  const k = levels.length;
  // A level's own name wins over parsing, so a level named "4" is that level, not index 4.
  const named = typeof value === 'string' ? levels.indexOf(value.trim()) : -1;
  const parsed = named >= 0 ? levels[named] : structured(value) ?? value;
  const point = (x: number) => {  // a fractional position splits its mass between the two neighbouring levels
    if (!Number.isFinite(x) || x < 0 || x > k - 1) return undefined;
    const out = new Array(k).fill(0), low = Math.floor(x), frac = x - low;
    out[low] += 1 - frac; if (frac > 0) out[low + 1] += frac;
    return out;
  };
  if (typeof parsed === 'number') return point(parsed);
  if (typeof parsed === 'string') return levels.includes(parsed.trim()) ? point(levels.indexOf(parsed.trim())) : undefined;
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return undefined;
  const weights = (parsed as any).probabilities;
  if (weights && typeof weights === 'object') {
    const raw = levels.map((name, i) => weights[name] ?? weights[String(i)] ?? 0);
    if (!raw.every(w => typeof w === 'number' && Number.isFinite(w) && w >= 0)) return undefined;
    const total = raw.reduce((a, b) => a + b, 0);
    return total > 0 ? raw.map(w => w / total) : undefined;
  }
  return typeof (parsed as any).score === 'number' ? point((parsed as any).score) : undefined;
}

/** Ranked probability score of a predicted against a target distribution, normalised to [0, 1] (0 is perfect). */
export function rankedProbabilityScore(predicted: number[], target: number[]): number {
  let p = 0, t = 0, sum = 0;
  for (let i = 0; i < predicted.length - 1; i++) { p += predicted[i]!; t += target[i]!; sum += (p - t) ** 2; }
  return predicted.length > 1 ? sum / (predicted.length - 1) : 0;
}

function scoreSimple(metric: GradedMetric, value: unknown, expected: unknown): GradedScore {
  const reference = expected as Record<string, unknown> | null;
  if (metric.kind === 'compaction-utility') {
    // Keep what the task needs and little else: recall of the needed items minus `cost` times the share kept.
    const needed = reference?.kind === 'keep-set' && Array.isArray(reference.needed) ? reference.needed as string[] : undefined;
    const total = typeof reference?.total === 'number' ? reference.total : undefined, cost = typeof reference?.cost === 'number' ? reference.cost : 0.5;
    if (!needed?.length || !total) return invalid('valid_reference');
    const parsed = structured(value);
    const kept = Array.isArray(parsed) ? parsed : parsed && typeof parsed === 'object' ? (parsed as any).keep : undefined;
    if (!Array.isArray(kept) || !kept.every(item => typeof item === 'string')) return invalid('returned_selection');
    const unique = new Set(kept as string[]), recall = needed.filter(item => unique.has(item)).length / needed.length;
    const share = Math.min(1, unique.size / total);
    return { quality: Math.max(0, recall - cost * share), gates: { selection: true, all_needed: recall === 1 },
      detail: { recall, kept_share: share } };
  }
  if (metric.kind === 'binary-brier') {
    // `answer` is a label (true/false) or a target frequency in [0, 1] (soft labels such as annotator agreement).
    const target = reference?.kind === 'binary' ? (typeof reference.answer === 'boolean' ? (reference.answer ? 1 : 0) : reference.answer) : undefined;
    if (typeof target !== 'number' || target < 0 || target > 1) return invalid('valid_reference');
    const p = probabilityOf(value);
    if (p === undefined) return invalid('returned_probability');
    return { quality: 1 - (p - target) ** 2, gates: { probability: true, correct_side: (p >= 0.5) === (target >= 0.5) },
      detail: { log_score: target * Math.log(Math.max(p, 1e-6)) + (1 - target) * Math.log(Math.max(1 - p, 1e-6)) } };
  }
  if (metric.kind === 'ordinal-rps') {
    const levels = reference?.kind === 'ordinal' && Array.isArray(reference.levels) ? reference.levels as string[] : undefined;
    const answer = reference?.answer;
    if (!levels || levels.length < 2 || typeof answer !== 'number') return invalid('valid_reference');
    const target = ordinalDistribution(answer, levels), predicted = ordinalDistribution(value, levels);
    if (!target) return invalid('valid_reference');
    if (!predicted) return invalid('returned_distribution');
    const expectedLevel = predicted.reduce((sum, w, i) => sum + w * i, 0);
    return { quality: 1 - rankedProbabilityScore(predicted, target), gates: { distribution: true,
      within_half_level: Math.abs(expectedLevel - answer) <= 0.5 }, detail: { expected_level: expectedLevel } };
  }
  if (metric.kind === 'choice-brier') {
    const options = reference?.kind === 'choice' && Array.isArray(reference.options) ? reference.options as string[] : undefined;
    if (!options?.length || !options.includes(reference!.answer as string)) return invalid('valid_reference');
    const distribution = choiceDistribution(value, options);
    if (!distribution) return invalid('returned_distribution');
    const brier = options.reduce((sum, o) => sum + (distribution[o]! - (o === reference!.answer ? 1 : 0)) ** 2, 0);
    const top = options.reduce((best, o) => distribution[o]! > distribution[best]! ? o : best, options[0]!);
    return { quality: 1 - brier / 2, gates: { distribution: true, top_correct: top === reference!.answer },
      detail: { brier, log_score: Math.log(Math.max(distribution[reference!.answer as string]!, 1e-6)) } };
  }
  if (metric.kind === 'answer-token-f1') {
    const answers = reference?.kind === 'gold-answer' ? [reference.value].flat() : [];
    if (!answers.length || !answers.every(item => typeof item === 'string')) return invalid('valid_reference');
    const text = typeof value === 'string' ? value : value && typeof value === 'object' && typeof (value as any).answer === 'string' ? (value as any).answer : undefined;
    if (text === undefined) return invalid('returned_answer');
    const f1 = Math.max(...(answers as string[]).map(answer => tokenF1(text, answer)));
    return { quality: f1, gates: { answered: true, exact: f1 === 1 } };
  }
  if (metric.kind === 'ranking-ndcg') {
    const relevant = reference?.kind === 'relevant-set' && Array.isArray(reference.items) ? reference.items as string[] : undefined;
    if (!relevant?.length) return invalid('valid_reference');
    const parsed = structured(value);
    const ranking = Array.isArray(parsed) ? parsed : parsed && typeof parsed === 'object' ? (parsed as any).ranking : undefined;
    if (!Array.isArray(ranking) || !ranking.every(item => typeof item === 'string')) return invalid('returned_ranking');
    const score = ndcg(ranking, relevant);
    return { quality: score, gates: { ranked: true, relevant_first: ranking.slice(0, relevant.length).every(item => relevant.includes(item)) } };
  }
  if (metric.kind === 'assignment-accuracy') {
    const gold = reference?.kind === 'assignment' && reference.value && typeof reference.value === 'object' ? reference.value as Record<string, unknown> : undefined;
    if (!gold || !Object.keys(gold).length) return invalid('valid_reference');
    const parsed = structured(value);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return invalid('returned_assignment');
    const keys = Object.keys(gold), right = keys.filter(key => JSON.stringify((parsed as any)[key]) === JSON.stringify(gold[key])).length;
    return { quality: right / keys.length, gates: { assigned: true, all_correct: right === keys.length } };
  }
  const gold = reference?.kind === 'function-calls' ? callAtoms(reference.calls) : undefined;
  if (!gold) return invalid('valid_reference');
  const parsed = structured(value);
  const predicted = callAtoms(Array.isArray(parsed) ? parsed : parsed && typeof parsed === 'object' ? (parsed as any).calls : undefined);
  if (!predicted) return invalid('returned_calls');
  const { f1, precision, recall } = multisetF1(predicted, gold);
  return { quality: f1, gates: { well_formed: true, exact_calls: f1 === 1 }, detail: { precision, recall } };
}

/** Host-only graded score of a returned value against the episode reference. */
export function scoreGraded(metric: GradedMetric, value: unknown, expected: unknown): GradedScore {
  if (['answer-token-f1', 'ranking-ndcg', 'assignment-accuracy', 'call-f1', 'choice-brier', 'binary-brier', 'ordinal-rps', 'compaction-utility'].includes(metric.kind)) return scoreSimple(metric, value, expected);
  if (metric.kind === 'python-tests') {
    const reference = expected as { kind?: string; tests?: string } | null;
    if (!reference || reference.kind !== 'python-tests' || typeof reference.tests !== 'string') return invalid('valid_reference');
    const code = extractCode(value);
    if (!code) return invalid('returned_code');
    return pythonTestScore(code, reference.tests, { image: metric.sandbox_image });
  }
  if (metric.kind !== 'sql-result-f1') return invalid('graded_kind');
  const gold = expected as { kind?: string; db?: string; sql?: string } | null;
  if (!gold || gold.kind !== 'sql-gold' || typeof gold.db !== 'string' || typeof gold.sql !== 'string') return invalid('valid_reference');
  if (!metric.database_root || isAbsolute(gold.db) || normalize(gold.db).startsWith('..')) return invalid('database_path');
  const path = join(metric.database_root, gold.db);
  const sql = extractSql(value);
  if (!sql) return invalid('returned_sql');
  if (/;\s*\S/.test(sql)) return invalid('single_statement');
  const reference = runReadOnlyQuery(path, gold.sql);
  if (reference.error || !reference.rows) return invalid('reference_executes', { error: reference.error });
  const predicted = runReadOnlyQuery(path, sql);
  if (predicted.error || !predicted.rows) return invalid('executes', { error: predicted.error });
  const { f1, precision, recall } = multisetF1(predicted.rows, reference.rows);
  return { quality: f1, gates: { executes: true, exact_result: f1 === 1 }, detail: { precision, recall, rows: predicted.rows.length, gold_rows: reference.rows.length } };
}

/**
 * The output the reference itself implies, for checks that the scorer gives the gold answer its best score (the
 * episode gate) and for supervision (soft-skill baselines). Undefined when the reference has no output form
 * (python-tests carry tests, not a solution).
 */
export function goldOutput(kind: GradedKind, expected: unknown): unknown {
  const reference = expected as Record<string, any> | null;
  if (!reference || typeof reference !== 'object') return undefined;
  switch (kind) {
    case 'sql-result-f1': return reference.kind === 'sql-gold' ? reference.sql : undefined;
    case 'answer-token-f1': return reference.kind === 'gold-answer' ? [reference.value].flat()[0] : undefined;
    case 'ranking-ndcg': return reference.kind === 'relevant-set' ? reference.items : undefined;
    case 'assignment-accuracy': return reference.kind === 'assignment' ? reference.value : undefined;
    case 'call-f1': return reference.kind === 'function-calls' ? reference.calls : undefined;
    case 'choice-brier': return reference.kind === 'choice' && Array.isArray(reference.options)
      ? { probabilities: Object.fromEntries(reference.options.map((o: string) => [o, o === reference.answer ? 1 : 0])) } : undefined;
    case 'binary-brier': return reference.kind === 'binary' ? (typeof reference.answer === 'boolean' ? Number(reference.answer) : reference.answer) : undefined;
    case 'ordinal-rps': return reference.kind === 'ordinal' ? { score: reference.answer } : undefined;
    case 'compaction-utility': return reference.kind === 'keep-set' ? reference.needed : undefined;
    default: return undefined;
  }
}

/** The best quality a gold output can reach: 1, except where the metric charges for the output itself. */
export function goldQualityBound(kind: GradedKind, expected: unknown): number {
  const reference = expected as Record<string, any> | null;
  if (kind === 'compaction-utility' && reference?.kind === 'keep-set')
    return Math.max(0, 1 - (reference.cost ?? 0.5) * Math.min(1, new Set(reference.needed).size / reference.total));
  return 1;
}
