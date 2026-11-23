/**
 * Collection combinators (plans/semantic-fp/03_SEMANTIC_STDLIB_AND_PATTERNS.md S1; ts-host/std/DECOMPOSITION.md):
 * `semanticMap`, `semanticFilter`, `semanticPartition`, `semanticReduce`, `semanticMatch` and `semanticRank`.
 * Each combinator is TypeScript mechanism (bounded concurrency, exact grouping, blocking, Copeland ranking) around
 * typed natural-language judgments. Values pass through by reference so `Neuralese<T>` states flow untouched.
 */
import { currentFrame } from '../runtime/context.js';
import { decideInFrame } from '../runtime/runtime.js';
import { isNeuraleseRef, NeuraleseUnsupportedError } from '../native/neuralese.js';
import { readNeuraleseForCurrentTask } from '../neuralese/combinators.js';
import { beats as shippedBeats, meets as shippedMeets, relates as shippedRelates } from './judgments.js';
import { StdSpan, stdError } from './trace.js';

type Maybe<T> = T | Promise<T>;

/** How intermediate state or mapped items are carried between calls: plain `'text'` (default) or `'neuralese'` soft state. */
export type CarryMode = 'text' | 'neuralese';

export function carryOf(value: unknown, combinator: string): CarryMode {
  if (value === undefined || value === 'text') return 'text';
  if (value === 'neuralese') return 'neuralese';
  throw stdError('std-arguments', `${combinator} carry must be "text" or "neuralese".`);
}

const asViewText = (value: unknown): string => typeof value === 'string' ? value : JSON.stringify(value) ?? String(value);

/**
 * Default soft view for `carry: 'neuralese'`: the Neuralese instance of the builtin `view(value, instructions?)`
 * (neuralese/view.ts), the one compaction operator the runtime serves and trains.
 */
export async function defaultSoftView(value: unknown, instructions?: string): Promise<unknown> {
  if (isNeuraleseRef(value)) return value;
  const view = currentFrame()?.task.runtime.options.neuralese?.view;
  const viewed = view ? await view({ value: asViewText(value), ...(instructions ? { instructions } : {}) }) : undefined;
  if (viewed !== undefined) return viewed;
  throw new NeuraleseUnsupportedError('carry: "neuralese" needs a runtime with a Neuralese server that serves view; ' +
    'leave carry out to pass ordinary values, or pass your own view option.');
}

/** Default final readout for `carry: 'neuralese'`: reads a Neuralese reference back to an ordinary value. */
export async function defaultSoftRead(value: unknown): Promise<unknown> {
  return isNeuraleseRef(value) ? readNeuraleseForCurrentTask(value) : value;
}

function concurrencyOf(value: unknown, combinator: string): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isSafeInteger(value) || (value as number) < 1)
    throw stdError('std-arguments', `${combinator} concurrency must be a whole number of at least 1.`);
  return value as number;
}

function floorOf(value: unknown, fallback: number, combinator: string): number {
  const resolved = value === undefined ? fallback : value;
  if (typeof resolved !== 'number' || !(resolved >= 0 && resolved <= 1))
    throw stdError('std-arguments', `${combinator} floor is a probability between 0 and 1.`);
  return resolved;
}

/** Map `items` with optional bounded concurrency, preserving order. */
async function mapBounded<T, R>(items: readonly T[], concurrency: number | undefined, fn: (item: T, index: number) => Promise<R>): Promise<R[]> {
  if (!items.length) return [];
  if (concurrency === undefined || concurrency >= items.length) return Promise.all(items.map((item, index) => fn(item, index)));
  const out = new Array<R>(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const index = next++;
      out[index] = await fn(items[index]!, index);
    }
  };
  await Promise.all(Array.from({ length: concurrency }, worker));
  return out;
}

// --- semanticMap ----------------------------------------------------------------------------------------------------

export type MapOptions = {
  /** Maximum concurrent calls. Omit to run every item concurrently. */
  concurrency?: number;
  /** How items reach `fn`: `'text'` (default) passes items untouched; `'neuralese'` runs `view` once per item first. */
  carry?: CarryMode;
  /** Optional soft-view hook used when `carry: 'neuralese'` (default: the Neuralese instance of the builtin `view`). */
  view?: (value: unknown, instructions?: string) => Maybe<unknown>;
  /** Optional instructions forwarded to `view` when `carry: 'neuralese'`. */
  viewInstructions?: string;
};

/** Apply a semantic function to every item in `items` concurrently (or with bounded concurrency), preserving order. */
export async function semanticMap<T, R>(items: readonly T[], fn: (item: T, index: number) => Maybe<R>, options: MapOptions = {}): Promise<R[]> {
  if (!Array.isArray(items)) throw stdError('std-arguments', 'semanticMap(items, fn) needs an array of items.');
  if (typeof fn !== 'function') throw stdError('std-arguments', 'semanticMap(items, fn) needs a function.');
  const concurrency = concurrencyOf(options.concurrency, 'semanticMap');
  const carry = carryOf(options.carry, 'semanticMap');
  const view = options.view ?? defaultSoftView;
  const span = new StdSpan('semanticMap');
  return span.around(async () => {
    const results = await mapBounded(items, concurrency, async (item, index) => {
      const input = carry === 'neuralese' ? await span.run(() => view(item, options.viewInstructions)) as T : item;
      return await span.run(() => fn(input, index));
    });
    span.stage('map', { items: items.length, ...(carry === 'neuralese' ? { carry } : {}) });
    return results;
  }, results => `mapped ${results.length} items`);
}

// --- semanticFilter -------------------------------------------------------------------------------------------------

export type FilterOptions<T> = {
  /** Minimum probability of `true` required to keep an item (default 0.5 when scoring a decision or criterion). */
  floor?: number;
  /** Maximum concurrent calls. */
  concurrency?: number;
  /** Replacement for the shipped `meets(criterion, item)` judgment, such as `meets.in(context)` or `meets.on({ model })`. */
  meets?: (criterion: string, item: T) => Promise<boolean>;
};

/**
 * Keep every item in `items` that satisfies `predicate` (either a semantic function `(item) => Promise<boolean>` or a
 * plain-language criterion judged by the shipped `meets.nl` decision readout).
 */
export async function semanticFilter<T>(items: readonly T[], predicate: string | ((item: T, index: number) => Maybe<boolean>),
    options: FilterOptions<T> = {}): Promise<T[]> {
  if (!Array.isArray(items)) throw stdError('std-arguments', 'semanticFilter(items, predicate) needs an array of items.');
  if (typeof predicate !== 'string' && typeof predicate !== 'function')
    throw stdError('std-arguments', 'semanticFilter(items, predicate) needs a criterion string or predicate function.');
  const concurrency = concurrencyOf(options.concurrency, 'semanticFilter');
  const floor = options.floor === undefined ? undefined : floorOf(options.floor, 0.5, 'semanticFilter');
  const span = new StdSpan('semanticFilter');
  return span.around(async () => {
    const keptFlags = await mapBounded(items, concurrency, async (item, index) => {
      if (typeof predicate === 'string') {
        const judge = options.meets ?? shippedMeets;
        const decision = await decideInFrame(span.frame, judge, [predicate, item]);
        const prob = decision.probabilities.find(entry => entry.value === true)?.probability ?? (decision.value === true ? decision.confidence : 0);
        return prob >= (floor ?? 0.5);
      }
      if (floor !== undefined) {
        const decision = await decideInFrame(span.frame, async (x: T, i: number) => predicate(x, i), [item, index]);
        const prob = decision.probabilities.find(entry => entry.value === true)?.probability ?? (decision.value === true ? decision.confidence : 0);
        return prob >= floor;
      }
      return Boolean(await span.run(() => predicate(item, index)));
    });
    const kept = items.filter((_, index) => keptFlags[index]);
    span.stage('filter', { items: items.length, kept: kept.length });
    return kept;
  }, kept => `kept ${kept.length} of ${items.length}`);
}

// --- semanticPartition ----------------------------------------------------------------------------------------------

export type PartitionOptions = {
  /** Minimum decision confidence to place an item in its group; items below the floor go into `uncertain`. */
  floor?: number;
  /** Maximum concurrent calls. */
  concurrency?: number;
};

export type Partition<T, K extends string = string> = {
  /** Items grouped by the label `classifier` returned, preserving input order within each group. */
  groups: Record<K, T[]>;
  /** Items whose classification confidence was below `floor` (empty when `floor` is omitted). */
  uncertain: { item: T; label: K; confidence: number }[];
};

/** Classify every item concurrently and partition them by label, separating low-confidence items into `uncertain`. */
export async function semanticPartition<T, K extends string = string>(items: readonly T[], classifier: (item: T) => Maybe<K>,
    options: PartitionOptions = {}): Promise<Partition<T, K>> {
  if (!Array.isArray(items)) throw stdError('std-arguments', 'semanticPartition(items, classifier) needs an array of items.');
  if (typeof classifier !== 'function') throw stdError('std-arguments', 'semanticPartition(items, classifier) needs a classifier function.');
  const concurrency = concurrencyOf(options.concurrency, 'semanticPartition');
  const floor = options.floor === undefined ? undefined : floorOf(options.floor, 0, 'semanticPartition');
  const span = new StdSpan('semanticPartition');
  return span.around(async () => {
    const decisions = await mapBounded(items, concurrency, async item => {
      if (floor !== undefined) {
        const decision = await decideInFrame(span.frame, async (x: T) => classifier(x), [item]);
        return { label: decision.value, confidence: decision.confidence };
      }
      const label = await span.run(() => classifier(item));
      return { label, confidence: 1 };
    });
    const groups = {} as Record<K, T[]>;
    const uncertain: { item: T; label: K; confidence: number }[] = [];
    decisions.forEach(({ label, confidence }, index) => {
      const item = items[index]!;
      if (floor !== undefined && confidence < floor) {
        uncertain.push({ item, label, confidence });
        return;
      }
      (groups[label] ??= []).push(item);
    });
    span.stage('partition', { items: items.length, groups: Object.keys(groups).length, uncertain: uncertain.length });
    return { groups, uncertain };
  }, result => `${Object.keys(result.groups).length} groups, ${result.uncertain.length} uncertain`);
}

// --- semanticReduce -------------------------------------------------------------------------------------------------

export type ReduceOptions<S> = {
  /** `'linear'` (default) folds left to right; `'tree'` maps leaves concurrently and combines pairs bottom-up. */
  mode?: 'linear' | 'tree';
  /** Associative combiner for `mode: 'tree'`. */
  combine?: (left: S, right: S) => Maybe<S>;
  /** Maximum concurrent calls in `mode: 'tree'`. */
  concurrency?: number;
  /** How state is carried between steps: `'text'` (default) or `'neuralese'` (`view` between steps, `read` once at the end). */
  carry?: CarryMode;
  /** Optional soft-view hook used when `carry: 'neuralese'` (default: the Neuralese instance of the builtin `view`). */
  view?: (value: unknown, instructions?: string) => Maybe<unknown>;
  /** Optional readout hook used once at the end when `carry: 'neuralese'` (default: Neuralese `read`). */
  read?: (value: unknown) => Maybe<unknown>;
  /** Optional instructions forwarded to `view` when `carry: 'neuralese'`. */
  viewInstructions?: string;
};

/**
 * Fold `items` from `initial` using `step(state, item, index)`. State passes by reference, so `Neuralese<S>` values
 * carry directly from step to step (or across tree levels in `mode: 'tree'`). With `carry: 'neuralese'`, intermediate
 * states pass through `view` and are read once at the end.
 */
export async function semanticReduce<T, S>(items: readonly T[], initial: S, step: (state: S, item: T, index: number) => Maybe<S>,
    options: ReduceOptions<S> = {}): Promise<S> {
  if (!Array.isArray(items)) throw stdError('std-arguments', 'semanticReduce(items, initial, step) needs an array of items.');
  if (typeof step !== 'function') throw stdError('std-arguments', 'semanticReduce(items, initial, step) needs a step function.');
  const mode = options.mode ?? 'linear';
  if (mode !== 'linear' && mode !== 'tree') throw stdError('std-arguments', 'semanticReduce mode must be "linear" or "tree".');
  if (mode === 'tree' && typeof options.combine !== 'function')
    throw stdError('std-arguments', 'semanticReduce with mode "tree" needs a combine(left, right) function.');
  const concurrency = concurrencyOf(options.concurrency, 'semanticReduce');
  const carry = carryOf(options.carry, 'semanticReduce');
  const view = options.view ?? defaultSoftView;
  const read = options.read ?? defaultSoftRead;
  const span = new StdSpan('semanticReduce');
  return span.around(async () => {
    if (!items.length) return initial;
    if (mode === 'linear') {
      let state: unknown = carry === 'neuralese' ? await span.run(() => view(initial, options.viewInstructions)) : initial;
      for (let index = 0; index < items.length; index++) {
        state = await span.run(() => step(state as S, items[index]!, index));
        if (carry === 'neuralese' && index + 1 < items.length)
          state = await span.run(() => view(state, options.viewInstructions));
        span.stage('step', { index, ...(carry === 'neuralese' ? { carry } : {}) });
      }
      return (carry === 'neuralese' ? await span.run(() => read(state)) : state) as S;
    }
    const initState = (carry === 'neuralese' ? await span.run(() => view(initial, options.viewInstructions)) : initial) as S;
    let level = await mapBounded(items, concurrency, async (item, index) => {
      const out = await span.run(() => step(initState, item, index));
      return (carry === 'neuralese' && items.length > 1 ? await span.run(() => view(out, options.viewInstructions)) : out) as S;
    });
    span.stage('leaves', { count: level.length, ...(carry === 'neuralese' ? { carry } : {}) });
    let round = 0;
    while (level.length > 1) {
      const pairs: [S, S][] = [];
      const odd = level.length % 2 === 1 ? level[level.length - 1] : undefined;
      for (let i = 0; i + 1 < level.length; i += 2) pairs.push([level[i]!, level[i + 1]!]);
      const nextCount = pairs.length + (odd !== undefined ? 1 : 0);
      const combined = await mapBounded(pairs, concurrency, async ([left, right]) => {
        const out = await span.run(() => options.combine!(left, right));
        return (carry === 'neuralese' && nextCount > 1 ? await span.run(() => view(out, options.viewInstructions)) : out) as S;
      });
      if (odd !== undefined) combined.push(odd);
      level = combined;
      span.stage('combine', { round: ++round, count: level.length, ...(carry === 'neuralese' ? { carry } : {}) });
    }
    return (carry === 'neuralese' ? await span.run(() => read(level[0]!)) : level[0]!) as S;
  }, () => `reduced ${items.length} items (${mode})`);
}

// --- semanticMatch --------------------------------------------------------------------------------------------------

export type MatchOptions<L, R> = {
  /** Plain-language relation judged by the shipped `relates.nl` when `match` is not supplied. */
  relation?: string;
  /** Caller's own pair judgment `(left, right) => Promise<boolean>` instead of `relation`. */
  match?: (left: L, right: R) => Maybe<boolean>;
  /** Optional exact pre-filter: only pairs where `block(left, right)` is true reach the model. */
  block?: (left: L, right: R) => boolean;
  /** Minimum probability of `true` for a match (default 0.5). */
  floor?: number;
  /** Maximum concurrent pair judgments. */
  concurrency?: number;
  /** Replacement for the shipped `relates(relation, left, right)`. */
  relates?: (relation: string, left: L, right: R) => Promise<boolean>;
};

export type MatchPair<L, R> = { left: L; right: R; leftIndex: number; rightIndex: number; confidence: number };
export type MatchResult<L, R> = { pairs: MatchPair<L, R>[]; unmatchedLeft: L[]; unmatchedRight: R[] };

/** Match elements across `left` and `right`, using exact `block`ing to prune candidates before semantic pair decisions. */
export async function semanticMatch<L, R>(left: readonly L[], right: readonly R[], options: MatchOptions<L, R>): Promise<MatchResult<L, R>> {
  if (!Array.isArray(left) || !Array.isArray(right)) throw stdError('std-arguments', 'semanticMatch(left, right, options) needs two arrays.');
  if (typeof options?.match !== 'function' && (!options?.relation || typeof options.relation !== 'string'))
    throw stdError('std-arguments', 'semanticMatch needs either a relation string or a match(left, right) function.');
  const concurrency = concurrencyOf(options.concurrency, 'semanticMatch');
  const floor = floorOf(options.floor, 0.5, 'semanticMatch');
  const span = new StdSpan('semanticMatch');
  return span.around(async () => {
    const candidates: { left: L; right: R; leftIndex: number; rightIndex: number }[] = [];
    for (let li = 0; li < left.length; li++) {
      for (let ri = 0; ri < right.length; ri++) {
        if (options.block && !options.block(left[li]!, right[ri]!)) continue;
        candidates.push({ left: left[li]!, right: right[ri]!, leftIndex: li, rightIndex: ri });
      }
    }
    span.stage('block', { candidates: candidates.length, totalPairs: left.length * right.length });
    const judged = await mapBounded(candidates, concurrency, async candidate => {
      if (options.match) {
        const decision = await decideInFrame(span.frame, async (l: L, r: R) => options.match!(l, r), [candidate.left, candidate.right]);
        const confidence = decision.probabilities.find(entry => entry.value === true)?.probability ?? (decision.value === true ? decision.confidence : 0);
        return { ...candidate, confidence };
      }
      const judge = options.relates ?? shippedRelates;
      const decision = await decideInFrame(span.frame, judge, [options.relation!, candidate.left, candidate.right]);
      const confidence = decision.probabilities.find(entry => entry.value === true)?.probability ?? (decision.value === true ? decision.confidence : 0);
      return { ...candidate, confidence };
    });
    const pairs = judged.filter(item => item.confidence >= floor);
    const matchedL = new Set(pairs.map(pair => pair.leftIndex));
    const matchedR = new Set(pairs.map(pair => pair.rightIndex));
    const unmatchedLeft = left.filter((_, index) => !matchedL.has(index));
    const unmatchedRight = right.filter((_, index) => !matchedR.has(index));
    span.stage('match', { matched: pairs.length });
    return { pairs, unmatchedLeft, unmatchedRight };
  }, result => `${result.pairs.length} matches`);
}

// --- semanticRank ---------------------------------------------------------------------------------------------------

export type RankOptions<T> = {
  /** Replacement for the shipped `beats(goal, first, second)` (`readout: decision`), such as `beats.on({ model })`. */
  beats?: (goal: string, first: T, second: T) => Promise<boolean>;
  /** Compare every pair in both orders `(a, b)` and `(b, a)` to cancel position bias (default `true`). */
  debiased?: boolean;
  /** Maximum concurrent pairwise comparisons. */
  concurrency?: number;
};

export type RankedEntry<T> = { item: T; index: number; wins: number; score: number };
export type Ranking<T> = { ranked: T[]; scores: RankedEntry<T>[] };

/**
 * Rank `items` against `goal` by concurrent pairwise `beats` decisions and exact Copeland / expected-win scoring.
 * Debiased by default: each pair is judged in both directions `(a, b)` and `(b, a)` and averaged to eliminate
 * position bias.
 */
export async function semanticRank<T>(items: readonly T[], goal: string, options: RankOptions<T> = {}): Promise<Ranking<T>> {
  if (!Array.isArray(items)) throw stdError('std-arguments', 'semanticRank(items, goal) needs an array of items.');
  if (typeof goal !== 'string' || !goal.trim()) throw stdError('std-arguments', 'semanticRank(items, goal) needs a nonempty goal string.');
  const concurrency = concurrencyOf(options.concurrency, 'semanticRank');
  const debiased = options.debiased ?? true;
  const beats = options.beats ?? shippedBeats;
  const span = new StdSpan('semanticRank');
  return span.around(async () => {
    if (items.length <= 1) {
      const scores = items.map((item, index) => ({ item, index, wins: 0, score: 0 }));
      return { ranked: [...items], scores };
    }
    const comparisons: { first: number; second: number }[] = [];
    for (let i = 0; i < items.length; i++) {
      for (let j = i + 1; j < items.length; j++) {
        comparisons.push({ first: i, second: j });
        if (debiased) comparisons.push({ first: j, second: i });
      }
    }
    const probMap = new Map<string, number>();
    await mapBounded(comparisons, concurrency, async ({ first, second }) => {
      const decision = await decideInFrame(span.frame, beats, [goal, items[first]!, items[second]!]);
      const pFirst = decision.probabilities.find(entry => entry.value === true)?.probability ?? (decision.value === true ? decision.confidence : 1 - decision.confidence);
      probMap.set(`${first}:${second}`, pFirst);
    });
    span.stage('compare', { items: items.length, comparisons: comparisons.length, debiased });
    const scores: RankedEntry<T>[] = items.map((item, index) => ({ item, index, wins: 0, score: 0 }));
    for (let i = 0; i < items.length; i++) {
      for (let j = i + 1; j < items.length; j++) {
        const pij = probMap.get(`${i}:${j}`) ?? 0.5;
        const p = debiased ? (pij + (1 - (probMap.get(`${j}:${i}`) ?? 0.5))) / 2 : pij;
        scores[i]!.score += p;
        scores[j]!.score += 1 - p;
        if (p > 0.5) scores[i]!.wins += 1;
        else if (p < 0.5) scores[j]!.wins += 1;
        else { scores[i]!.wins += 0.5; scores[j]!.wins += 0.5; }
      }
    }
    scores.sort((a, b) => (b.score - a.score) || (b.wins - a.wins) || (a.index - b.index));
    span.stage('rank', { topIndex: scores[0]!.index });
    return { ranked: scores.map(entry => entry.item), scores };
  }, result => `ranked ${result.ranked.length} items`);
}
