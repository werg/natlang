import { createHash } from 'node:crypto';

export const SYNTHETIC_IR_VERSION = 'natlang.program/1';
export const NATIVE_SYNTHETIC_GENERATOR_VERSION = 'natlang.synthetic_generator.native/1';
export const NATIVE_SYNTHETIC_FAMILIES = ['array_kernel', 'staged_ranking', 'algorithm_pipeline'] as const;
export type NativeSyntheticFamily = typeof NATIVE_SYNTHETIC_FAMILIES[number];
type Dict = Record<string, unknown>;

/** A small deterministic PRNG. Each program is independently derived from (seed,index). */
class Random {
  private state: number;
  constructor(seed: number, index: number) {
    const digest = createHash('sha256').update(`${seed}:${index}:native-synthetic/1`).digest();
    this.state = digest.readUInt32LE(0) || 0x9e3779b9;
  }
  next(): number {
    let x = this.state;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    this.state = x >>> 0;
    return this.state / 0x1_0000_0000;
  }
  int(min: number, max: number): number { return min + Math.floor(this.next() * (max - min + 1)); }
  pick<T>(values: readonly T[]): T { return values[this.int(0, values.length - 1)]!; }
}

function record(id: string, family: string, kind: 'lambda_source' | 'lambda_graph', semantics: Dict): Dict {
  const record = { version: SYNTHETIC_IR_VERSION, id, kind, family: `algo_${family}`,
    source: 'natlang-synthetic-native', split: 'train', source_ids: [id], source_groups: [id],
    source_revisions: [NATIVE_SYNTHETIC_GENERATOR_VERSION], license: 'project-generated',
    gold_sources: ['typescript-synthetic-generator'], generation: { generator: NATIVE_SYNTHETIC_GENERATOR_VERSION },
    semantics };
  validateSyntheticRecord(record);
  return record;
}

function lambda(instructions: string, type: string, functionName: string, codebase?: Dict, types?: Dict): Dict {
  const value: Dict = { type, instructions, function: functionName };
  if (codebase) value.codebase = codebase;
  if (types) value.types = types;
  return { $lambda: value };
}

function arrayKernel(rng: Random, id: string): Dict {
  const nums = Array.from({ length: rng.int(4, 14) }, () => rng.int(-40, 90));
  const items = Array.from({ length: rng.int(4, 14) }, () => rng.pick(['alpha', 'beta', 'gamma', 'delta']));
  const matrix = Array.from({ length: rng.int(2, 7) }, () => Array.from({ length: rng.int(1, 6) }, () => rng.int(-20, 40)));
  const flags = Array.from({ length: rng.int(3, 15) }, () => rng.next() < 0.5);
  const intervals = Array.from({ length: rng.int(3, 9) }, () => {
    const start = rng.int(0, 30); return { start, end: start + rng.int(0, 9) };
  });
  const cases: Array<{ name: string; inputs: Dict; params: string; returns: string; text: string; code: string; expected: unknown }> = [
    { name: 'prefix_sums', inputs: { numbers: nums }, params: 'numbers: number[]', returns: 'number[]',
      text: 'Return every running prefix sum of `numbers` in order.',
      code: 'numbers.reduce((out, x) => out.concat([(out.length ? out[out.length - 1] : 0) + x]), [])',
      expected: nums.reduce<number[]>((out, x) => [...out, (out.at(-1) ?? 0) + x], []) },
    { name: 'window_sums', inputs: { numbers: nums, width: rng.int(1, Math.min(5, nums.length)) }, params: 'numbers: number[], width: number', returns: 'number[]',
      text: 'Compute the sum of every contiguous window of `width` numbers.',
      code: 'numbers.slice(0, numbers.length - width + 1).map((_, i) => sum(numbers.slice(i, i + width)))',
      expected: [] },
    { name: 'top_k', inputs: { numbers: nums, k: rng.int(0, nums.length) }, params: 'numbers: number[], k: number', returns: 'number[]',
      text: 'Return the largest `k` values from `numbers`, greatest first.', code: 'numbers.slice().sort((a, b) => b - a).slice(0, k)', expected: [] },
    { name: 'stable_unique', inputs: { items }, params: 'items: string[]', returns: 'string[]',
      text: 'Remove duplicate values from `items` while preserving first occurrence order.',
      code: 'items.filter((x, i) => items.indexOf(x) === i)', expected: [...new Set(items)] },
    { name: 'weighted_checksum', inputs: { numbers: nums }, params: 'numbers: number[]', returns: 'number',
      text: 'Multiply each number by its one-based position and add the products.',
      code: 'sum(numbers.map((x, i) => x * (i + 1)))', expected: nums.reduce((sum, x, i) => sum + (i + 1) * x, 0) },
    { name: 'adjacent_changes', inputs: { items }, params: 'items: string[]', returns: 'number',
      text: 'Count positions after the first where the value in `items` differs from the preceding value.',
      code: 'items.slice(1).filter((x, i) => x !== items[i]).length',
      expected: items.slice(1).filter((x, i) => x !== items[i]).length },
    { name: 'row_sums', inputs: { matrix }, params: 'matrix: number[][]', returns: 'number[]',
      text: 'Return the sum of each row in `matrix`, preserving row order.', code: 'matrix.map(row => sum(row))',
      expected: matrix.map(row => row.reduce((a, b) => a + b, 0)) },
    { name: 'longest_true_run', inputs: { flags }, params: 'flags: boolean[]', returns: 'number',
      text: 'Return the length of the longest contiguous run of true values in `flags`.',
      code: 'flags.reduce((s, x) => ({ run: x ? s.run + 1 : 0, best: Math.max(s.best, x ? s.run + 1 : 0) }), { run: 0, best: 0 }).best',
      expected: flags.reduce((s, x) => ({ run: x ? s.run + 1 : 0, best: Math.max(s.best, x ? s.run + 1 : 0) }), { run: 0, best: 0 }).best },
    { name: 'merge_intervals', inputs: { intervals }, params: 'intervals: Interval[]', returns: 'Interval[]',
      text: 'Merge all overlapping intervals in `intervals` and return them ordered by start.',
      code: 'intervals.slice().sort((a, b) => a.start - b.start || a.end - b.end).reduce((out, x) => { const last = out[out.length - 1]; return last && x.start <= last.end ? out.slice(0, -1).concat([{ start: last.start, end: Math.max(last.end, x.end) }]) : out.concat([{ start: x.start, end: x.end }]) }, [])',
      expected: mergeIntervals(intervals) },
  ];
  const selected = rng.pick(cases);
  if (selected.name === 'window_sums') {
    const { width } = selected.inputs as { width: number };
    selected.expected = nums.slice(0, nums.length - width + 1).map((_, i) => nums.slice(i, i + width).reduce((a, b) => a + b, 0));
  }
  if (selected.name === 'top_k') {
    const { k } = selected.inputs as { k: number };
    selected.expected = [...nums].sort((a, b) => b - a).slice(0, k);
  }
  const root = lambda(selected.text, `(${selected.params}) => ${selected.returns}`, selected.name,
    undefined, selected.name === 'merge_intervals' ? { Interval: '{ start: number, end: number }' } : undefined);
  return record(id, selected.name, 'lambda_source', { root, inputs: selected.inputs, expected: selected.expected,
    operation: 'algorithm', expression: selected.code, algorithm: selected.name });
}

function mergeIntervals(intervals: Array<{ start: number; end: number }>): Array<{ start: number; end: number }> {
  const result: Array<{ start: number; end: number }> = [];
  for (const interval of [...intervals].sort((a, b) => a.start - b.start || a.end - b.end)) {
    const previous = result.at(-1);
    if (previous && interval.start <= previous.end) previous.end = Math.max(previous.end, interval.end);
    else result.push({ ...interval });
  }
  return result;
}

function stagedRanking(rng: Random, id: string): Dict {
  const candidates = Array.from({ length: rng.int(4, 12) }, (_, i) => ({ id: `c${i}`, quality: rng.int(0, 20), cost: rng.int(0, 100) }));
  const k = rng.int(0, candidates.length);
  const expected = candidates.map(item => ({ ...item, score: item.quality * 100 - item.cost }))
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).slice(0, k);
  const candidate = '{ id: string, quality: number, cost: number }', ranked = '{ id: string, quality: number, cost: number, score: number }';
  const instructions = 'Score every candidate, sort by descending score with id as the tie breaker, then return the first `k`.';
  const codebase = {
    score_candidates: { description: 'Attach the exact quality/cost score to every candidate.', args: { items: 'Candidate[]' }, returns: 'Ranked[]', code: 'return items.map(x => ({ ...x, score: x.quality * 100 - x.cost }))' },
    sort_ranked: { description: 'Sort ranked candidates by score descending and id ascending.', args: { items: 'Ranked[]' }, returns: 'Ranked[]', code: 'return items.slice().sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))' },
    take_ranked: { description: 'Take the requested prefix of a ranked list.', args: { items: 'Ranked[]', k: 'number' }, returns: 'Ranked[]', code: 'return items.slice(0, k)' },
  };
  const operations = [
    { op: 'eval', code: 'const scored = await score_candidates(candidates)' },
    { op: 'eval', code: 'const sorted = await sort_ranked(scored)' },
    { op: 'eval', code: 'await take_ranked(sorted, k)' },
  ];
  return record(id, 'staged_ranking', 'lambda_graph', { root: lambda(instructions,
    '(candidates: Candidate[], k: number) => Ranked[]', 'rank_candidates', codebase,
    { Candidate: candidate, Ranked: ranked }), inputs: { candidates, k }, expected, operations, source_lines: [], leaf_oracles: {} });
}

function algorithmPipeline(rng: Random, id: string): Dict {
  const accounts = Array.from({ length: rng.int(2, 6) }, (_, i) => `acct-${i}`);
  const transactions = Array.from({ length: rng.int(8, 24) }, () => ({ account: rng.pick(accounts), amount: rng.int(-50, 180), active: rng.next() < 0.8 }));
  const active = transactions.filter(item => item.active);
  const totals = accounts.map(account => active.filter(item => item.account === account).reduce((sum, item) => sum + item.amount, 0));
  const expected = accounts.map((account, i) => ({ account, total: totals[i]! })).sort((a, b) => b.total - a.total || a.account.localeCompare(b.account));
  const instructions = 'Discard inactive transactions, total the remaining amounts per account, and rank accounts by total descending then account id.';
  const codebase = {
    active_only: { description: 'Keep only active transactions.', args: { items: 'Transaction[]' }, returns: 'Transaction[]', code: 'return items.filter(x => x.active)' },
    rank_accounts: { description: 'Pair accounts with totals and sort the leaderboard.', args: { accounts: 'string[]', totals: 'number[]' }, returns: 'Summary[]', code: 'return accounts.map((account, i) => ({ account, total: totals[i] })).sort((a, b) => b.total - a.total || a.account.localeCompare(b.account))' },
  };
  const operations = [
    { op: 'eval', code: 'const active = await active_only(transactions)' },
    { op: 'eval', code: 'const totals = accounts.map(a => active.filter(x => x.account === a).reduce((sum, x) => sum + x.amount, 0))' },
    { op: 'eval', code: 'await rank_accounts(accounts, totals)' },
  ];
  return record(id, 'algorithm_pipeline', 'lambda_graph', { root: lambda(instructions,
    '(transactions: Transaction[], accounts: string[]) => Summary[]', 'account_leaderboard', codebase,
    { Transaction: '{ account: string, amount: number, active: boolean }', Summary: '{ account: string, total: number }' }),
    inputs: { transactions, accounts }, expected, operations, source_lines: [], leaf_oracles: {} });
}

export function validateSyntheticRecord(value: unknown): asserts value is Dict {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError('program IR must be an object');
  const item = value as Dict;
  if (item.version !== SYNTHETIC_IR_VERSION || typeof item.id !== 'string' || !item.id ||
      !['lambda_source', 'lambda_graph'].includes(String(item.kind)) || !String(item.family).startsWith('algo_'))
    throw new Error('invalid native synthetic program identity');
  const semantics = item.semantics as Dict;
  if (!semantics || typeof semantics !== 'object' || !semantics.root || !semantics.inputs || !('expected' in semantics))
    throw new Error(`${item.id}: missing source, inputs, or expected value`);
  if (item.kind === 'lambda_graph' && (!Array.isArray(semantics.operations) || !(semantics.operations as unknown[]).length ||
      !Array.isArray(semantics.source_lines) || !semantics.leaf_oracles)) throw new Error(`${item.id}: malformed lambda graph`);
  if (item.kind === 'lambda_source' && semantics.operation !== 'algorithm') throw new Error(`${item.id}: unsupported lambda source operation`);
}

export function generateSyntheticRecord(seed: number, index: number, family?: NativeSyntheticFamily): Dict {
  if (!Number.isSafeInteger(seed) || !Number.isSafeInteger(index) || index < 0) throw new RangeError('seed and index must be safe integers; index must be nonnegative');
  const rng = new Random(seed, index);
  const selected = family ?? NATIVE_SYNTHETIC_FAMILIES[index % NATIVE_SYNTHETIC_FAMILIES.length]!;
  if (!(NATIVE_SYNTHETIC_FAMILIES as readonly string[]).includes(selected)) throw new Error(`unsupported native family: ${selected}`);
  const id = `${seed}:algo_${selected}:${index}`;
  switch (selected) {
    case 'array_kernel': return arrayKernel(rng, id);
    case 'staged_ranking': return stagedRanking(rng, id);
    case 'algorithm_pipeline': return algorithmPipeline(rng, id);
  }
}

export function syntheticRecords(seed: number, start: number, count: number, families: readonly NativeSyntheticFamily[] = NATIVE_SYNTHETIC_FAMILIES): Dict[] {
  if (!Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(count) || count < 0) throw new RangeError('start and count must be nonnegative safe integers');
  if (!families.length) throw new Error('at least one family is required');
  return Array.from({ length: count }, (_, offset) => generateSyntheticRecord(seed, start + offset, families[(start + offset) % families.length]));
}
