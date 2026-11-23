/**
 * CI Doctor (WS6 Tier 2 flagship A5):
 * - Crisp `pluggable` known-flake signature detector handles well-known transient network/runner flakes with 0 LLM calls.
 * - `semanticPartition` classifies remaining traces by `FailureKind` with a confidence floor, and `semanticMatch` clusters
 *   each kind by root cause.
 * - `iterateOn` with `withMeasure` runs a bounded binary-search commit bisect, stopped by `bisectSettled.nl` when the
 *   tested commits show going on cannot find this failure's cause.
 * - `branchAndSelect` evaluates candidate fixes in isolated `Folder` copies against an exact test verifier
 *   plus a strict assertion-preservation guard (`assert`/`expect` count check) that rejects any test-weakening patch.
 */
import {
  Folder,
  iterateOn,
  pluggable,
  untrusted,
  type FolderProposal,
  type FolderSnapshot,
  type PluggableSetting,
} from '@natlang/node';
import { branchAndSelect, semanticMatch, semanticPartition, type Branch } from '@natlang/node/std';
import bisectSettled from './bisectSettled.nl';
import classifyFailure from './classifyFailure.nl';
import proposeFix from './proposeFix.nl';
import type {
  BisectObservation,
  BisectState,
  CiReport,
  Cluster,
  FailureKind,
  FailureTrace,
  FixProposal,
  RejectedCandidate,
} from './types.js';

/** What testing one commit gave: whether it failed, optionally with the test output. */
export type CommitOutcome = boolean | { fails: boolean; log?: string };

export type DoctorOptions = {
  /** Pluggable mode for known-flake detection (default `'crisp'`). */
  flakeMode?: PluggableSetting;
  /** Minimum decision confidence for `semanticPartition` (default `0.75`). */
  partitionFloor?: number;
  /** Maximum concurrent classification calls (default `6`). */
  concurrency?: number;
  /**
   * Optional commit tester for bisection: true when the commit reproduces a failure, or `{ fails, log }` with the test
   * output, which the natural-language stop reads to tell this failure from another.
   */
  failsAtCommit?: (commit: string, trace: FailureTrace) => Promise<CommitOutcome> | CommitOutcome;
  /**
   * When bisection stops (`pluggable`, default `'nl'`): `crisp` stops only when the range is one commit; `nl` also stops
   * when `bisectSettled.nl` judges that going on cannot find this failure's cause; `shadow` runs both and serves crisp.
   */
  bisectStop?: PluggableSetting;
  /**
   * How failures of one kind are grouped: `'semantic'` (default) puts traces with the same root cause in one cluster
   * (std `semanticMatch` pair decisions, joined transitively); `'signature'` groups by the exact first error line.
   */
  clustering?: 'semantic' | 'signature';
  /** Optional workspace folder to patch when a regression is identified. */
  repoFolder?: Folder;
  /** Optional custom candidate branches (defaults to two `proposeFix` strategy branches). */
  fixBranches?: Branch[];
  /** Optional candidate fix strategies passed to `proposeFix` when `fixBranches` is omitted. */
  strategies?: string[];
  /** Exact test-runner verifier callback over a candidate `FolderSnapshot`: returns failure messages, empty on green. */
  runTests?: (snapshot: FolderSnapshot) => Promise<readonly string[]> | readonly string[];
};

const KNOWN_FLAKE_PATTERNS: ReadonlyArray<{ signature: string; pattern: RegExp }> = [
  { signature: 'flake:econnreset', pattern: /\bECONNRESET\b/i },
  { signature: 'flake:etimedout', pattern: /\bETIMEDOUT\b/i },
  { signature: 'flake:socket-hang-up', pattern: /socket hang up/i },
  { signature: 'flake:rate-limit', pattern: /rate limit exceeded|HTTP 429/i },
  { signature: 'flake:service-unavailable', pattern: /503 Service Unavailable|502 Bad Gateway/i },
  { signature: 'flake:runner-lost', pattern: /runner lost communication|agent disconnected unexpectedly/i },
];

const DETERMINISTIC_FAILURE = /\bAssertionError\b|\berror TS\d+\b|\bSyntaxError\b|\bReferenceError\b|\bTypeError\b/;

/** Exact crisp detector for well-known transient CI flake signatures (zero model calls). */
export function detectKnownFlakeSignature(log: string): string | null {
  // A log that also shows a deterministic failure is not settled here; the semantic classifier weighs both.
  if (DETERMINISTIC_FAILURE.test(log)) return null;
  for (const entry of KNOWN_FLAKE_PATTERNS) {
    if (entry.pattern.test(log)) return entry.signature;
  }
  return null;
}

/** Fence an untrusted CI trace before crossing any model boundary. */
export function fenceTrace(trace: FailureTrace): FailureTrace {
  return {
    id: trace.id,
    job: trace.job,
    log: untrusted(String(trace.log), `ci:${trace.id}:log`),
    commits: [...trace.commits],
  };
}

/** Normalize a failure log line into a compact cluster signature. */
export function extractFailureSignature(trace: FailureTrace, kind: FailureKind): string {
  const crisp = detectKnownFlakeSignature(String(trace.log));
  if (crisp) return crisp;
  const firstError = String(trace.log)
    .split('\n')
    .map(line => line.trim())
    .find(line => /(?:Error|FAIL|AssertionError|TypeError|ReferenceError|Exit code)/i.test(line));
  if (!firstError) return `${kind}:${trace.job}`;
  return `${kind}:${firstError.replace(/\b0x[0-9a-f]+\b|\b\d{3,}\b/gi, '#').slice(0, 80)}`;
}

/**
 * Count active `assert(...)` / `assert.*(...)` and `expect(...)` calls in source text,
 * ignoring single-line `//` comments so commented-out assertions do not count.
 */
export function countAssertions(source: string): number {
  const withoutLineComments = source
    .split('\n')
    .map(line => line.replace(/\/\/.*$/, ''))
    .join('\n')
    .replace(/\/\*[\s\S]*?\*\//g, '');
  const assertMatches = withoutLineComments.match(/\bassert(?:\.\w+)?\s*\(/g) ?? [];
  const expectMatches = withoutLineComments.match(/\bexpect\s*\(/g) ?? [];
  return assertMatches.length + expectMatches.length;
}

const textDecoder = new TextDecoder('utf-8');

/**
 * Strict exact guard ensuring no candidate patch deletes, skips, or weakens existing test assertions.
 */
export function checkAssertionsPreserved(proposal: FolderProposal<FixProposal>): string[] {
  const problems: string[] = [];
  for (const change of proposal.diff.changes) {
    const beforeText = change.before ? textDecoder.decode(change.before) : '';
    const afterText = change.after ? textDecoder.decode(change.after) : '';
    const beforeCount = countAssertions(beforeText);
    const afterCount = countAssertions(afterText);
    if (change.kind === 'deleted' && beforeCount > 0) {
      problems.push(`Patch deleted ${change.path} which contained ${beforeCount} assertions.`);
      continue;
    }
    if (afterCount < beforeCount) {
      problems.push(
        `Assertion count decreased in ${change.path} (${beforeCount} -> ${afterCount}); fixes may not remove or weaken assertions.`,
      );
    }
    // An edited assertion is a weakened one until a human says otherwise: every existing assertion line must survive verbatim.
    const afterLines = new Set(assertionLines(afterText));
    const edited = assertionLines(beforeText).filter(line => !afterLines.has(line));
    if (edited.length > 0) {
      problems.push(`Patch changed existing assertion(s) in ${change.path} (${edited[0]}); fix the source code instead.`);
    }
    const beforeSkips = (beforeText.match(/\b(?:test|it|describe)\.skip\s*\(|\bskip\s*:\s*true\b/g) ?? []).length;
    const afterSkips = (afterText.match(/\b(?:test|it|describe)\.skip\s*\(|\bskip\s*:\s*true\b/g) ?? []).length;
    if (afterSkips > beforeSkips) {
      problems.push(`Patch introduced skipped tests in ${change.path}; fixes may not skip failing tests.`);
    }
  }
  return problems;
}

/** Trimmed source lines that contain an active `assert(...)` / `assert.*(...)` / `expect(...)` call. */
function assertionLines(source: string): string[] {
  return source
    .split('\n')
    .map(line => line.replace(/\/\/.*$/, '').trim())
    .filter(line => /\bassert(?:\.\w+)?\s*\(|\bexpect\s*\(/.test(line));
}

/**
 * Commit bisection with a natural-language stop. The step is an exact binary search over `commits` (oldest first; the
 * newest is taken as failing), bounded by a strictly decreasing `withMeasure` interval width. The stop is policy: the
 * range narrowed to one commit, or (`nl`) `bisectSettled.nl` judging from the tested commits' outputs that going on
 * cannot find this failure's cause. Returns the earliest failing commit when the range narrowed to it.
 */
export async function bisectCommits(
  commits: readonly string[],
  failsAtCommit: (commit: string) => Promise<CommitOutcome> | CommitOutcome,
  options: { trace?: FailureTrace; stop?: PluggableSetting } = {},
): Promise<{ culprit?: string; tested: string[]; stopped?: 'judged' }> {
  const observe = async (commit: string): Promise<BisectObservation> => {
    const outcome = await failsAtCommit(commit);
    return typeof outcome === 'boolean' ? { commit, fails: outcome } :
      { commit, fails: outcome.fails, ...(outcome.log !== undefined ? { log: untrusted(outcome.log, `ci:bisect:${commit}`) } : {}) };
  };
  if (commits.length === 0) return { tested: [] };
  if (commits.length === 1) {
    const observed = await observe(commits[0]!);
    return { ...(observed.fails ? { culprit: commits[0]! } : {}), tested: [commits[0]!] };
  }

  const trace = options.trace ?? { id: 'bisect', job: 'bisect', log: untrusted('', 'ci:bisect'), commits: [...commits] };
  const initial: BisectState = {
    trace,
    commits: [...commits],
    low: 0,
    high: commits.length - 1,
    tested: [],
    observations: [],
    done: false,
  };
  const stop = pluggable(
    {
      crisp: (state: BisectState): boolean => state.done,
      nl: async (state: BisectState): Promise<boolean> => state.done || (state.observations.length > 0 && await bisectSettled(state)),
    },
    options.stop ?? 'nl',
    { default: 'nl', serve: 'crisp', name: 'ci-bisect-stop' },
  );

  const maxSteps = Math.ceil(Math.log2(commits.length)) + 2;
  const finalState = await iterateOn(
    async (state: BisectState): Promise<BisectState> => {
      if (state.low >= state.high) {
        return { ...state, culprit: commits[state.low], done: true };
      }
      const mid = Math.floor((state.low + state.high) / 2);
      const observed = await observe(commits[mid]!);
      const nextLow = observed.fails ? state.low : mid + 1;
      const nextHigh = observed.fails ? mid : state.high;
      const done = nextLow >= nextHigh;
      return {
        ...state,
        low: nextLow,
        high: nextHigh,
        tested: [...state.tested, observed.commit],
        observations: [...state.observations, observed],
        ...(done ? { culprit: commits[nextLow] } : {}),
        done,
      };
    },
    initial,
  )
    .withMeasure(state => (state.done ? 0 : Math.max(0, state.high - state.low)))
    .withLimit({ maxSteps })
    .until(stop);

  return {
    ...(finalState.culprit ? { culprit: finalState.culprit } : {}),
    tested: finalState.tested,
    ...(!finalState.done ? { stopped: 'judged' as const } : {}),
  };
}

/**
 * Group traces of one failure kind into root-cause clusters: `semantic` asks, for each pair, whether the two failures
 * have the same root cause (std `semanticMatch` with the shipped `relates.nl`) and joins the pairs transitively;
 * `signature` groups by the exact normalized first error line. Each cluster is labelled with its first trace's signature.
 */
export async function clusterFailures(
  kind: FailureKind,
  traces: readonly FailureTrace[],
  options: { clustering?: 'semantic' | 'signature'; concurrency?: number } = {},
): Promise<Cluster[]> {
  const parent = traces.map((_, index) => index);
  const find = (index: number): number => {
    let root = index;
    for (const _ of traces) { if (parent[root] === root) break; root = parent[root]!; }
    return root;
  };
  const join = (a: number, b: number) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb); };
  if ((options.clustering ?? 'semantic') === 'semantic' && traces.length > 1) {
    const matched = await semanticMatch(traces, traces, {
      relation: 'the two CI failures have the same root cause: the same defect or fault, even when their logs word it differently',
      block: (left, right) => traces.indexOf(left) < traces.indexOf(right),
      ...(options.concurrency !== undefined ? { concurrency: options.concurrency } : {}),
    });
    for (const pair of matched.pairs) join(pair.leftIndex, pair.rightIndex);
  } else {
    const signatures = traces.map(trace => extractFailureSignature(trace, kind));
    signatures.forEach((signature, index) => { const first = signatures.indexOf(signature); if (first !== index) join(first, index); });
  }
  const clusters = new Map<number, Cluster>();
  traces.forEach((trace, index) => {
    const root = find(index);
    const cluster = clusters.get(root);
    if (cluster) cluster.traces.push(trace);
    else clusters.set(root, { kind, signature: extractFailureSignature(trace, kind), source: 'nl', traces: [trace] });
  });
  return [...clusters.values()];
}

/** Diagnose a batch of CI failures, bisect regressions, and verify candidate fixes via `branchAndSelect`. */
export async function diagnoseAndFixCi(
  rawTraces: FailureTrace[],
  options: DoctorOptions = {},
): Promise<CiReport> {
  const traces = rawTraces.map(fenceTrace);

  // Pluggable known-flake detector: crisp matches known signatures (0 model calls); nl asks classifyFailure.nl;
  // shadow runs both, records agreement, and serves crisp. Traces it does not settle go to semanticPartition.
  const flakeMode = options.flakeMode ?? 'crisp';
  const detectFlake = pluggable(
    {
      crisp: (trace: FailureTrace): boolean => detectKnownFlakeSignature(String(trace.log)) !== null,
      nl: async (trace: FailureTrace): Promise<boolean> => (await classifyFailure(trace)) === 'flake',
    },
    flakeMode,
    { default: 'crisp', serve: 'crisp', name: 'ci-known-flake-detector' },
  );
  const crispFlakes: FailureTrace[] = [];
  const remaining: FailureTrace[] = [];
  for (const trace of traces) {
    // In nl mode every trace is classified once by semanticPartition below; the fast path is crisp's alone.
    if (flakeMode !== 'nl' && await detectFlake(trace)) crispFlakes.push(trace);
    else remaining.push(trace);
  }
  const partition = remaining.length > 0
    ? await semanticPartition<FailureTrace, FailureKind>(
        remaining,
        trace => classifyFailure(trace),
        { floor: options.partitionFloor ?? 0.75, concurrency: options.concurrency ?? 6 },
      )
    : { groups: {} as Record<FailureKind, FailureTrace[]>, uncertain: [] };

  // Known flake signatures are exact clusters; the classified traces cluster by root cause within their kind.
  const clusters: Cluster[] = [];
  const crispClusters = new Map<string, Cluster>();
  for (const trace of crispFlakes) {
    const signature = extractFailureSignature(trace, 'flake');
    const existing = crispClusters.get(signature);
    if (existing) existing.traces.push(trace);
    else crispClusters.set(signature, { kind: 'flake', signature, source: 'crisp', traces: [trace] });
  }
  clusters.push(...crispClusters.values());
  for (const [kind, items] of Object.entries(partition.groups) as Array<[FailureKind, FailureTrace[]]>) {
    clusters.push(...await clusterFailures(kind, items, { clustering: options.clustering, concurrency: options.concurrency ?? 6 }));
  }

  const regressions = partition.groups.regression ?? [];
  let culpritCommit: string | undefined;
  let bisectedCommits: string[] = [];
  let bisectStopped: 'judged' | undefined;

  const primaryRegression = regressions[0];
  if (primaryRegression && primaryRegression.commits.length > 0 && options.failsAtCommit) {
    const bisect = await bisectCommits(
      primaryRegression.commits,
      commit => options.failsAtCommit!(commit, primaryRegression),
      { trace: primaryRegression, stop: options.bisectStop },
    );
    culpritCommit = bisect.culprit;
    bisectedCommits = bisect.tested;
    bisectStopped = bisect.stopped;
  }

  let fixAccepted = false;
  let chosenBranch: number | null = null;
  let chosenFix: FixProposal | undefined;
  let rejectedCandidates: RejectedCandidate[] = [];

  if (primaryRegression && options.repoFolder) {
    const strategies = options.strategies ?? [
      'minimal root-cause fix preserving existing API contracts and all test assertions',
      'defensive boundary-check fix with full test assertion preservation',
    ];
    const branches: Branch[] = options.fixBranches ?? strategies.map(strategy => ({
      reducer: proposeFix,
      args: [primaryRegression, culpritCommit ?? primaryRegression.commits.at(-1) ?? 'HEAD', strategy],
    }));

    const verify = async (proposal: FolderProposal<FixProposal>): Promise<string[]> => {
      const problems = checkAssertionsPreserved(proposal);
      if (options.runTests) {
        const testProblems = await options.runTests(proposal.folder);
        problems.push(...testProblems);
      }
      return problems;
    };

    const selected = await branchAndSelect<FixProposal>(options.repoFolder, branches, {
      verify,
      goal: 'minimal, clean regression fix that passes all tests without removing or weakening assertions',
    });

    chosenBranch = selected.chosen;
    fixAccepted = selected.chosen !== null;
    chosenFix = selected.value;
    rejectedCandidates = selected.candidates
      .filter(candidate => candidate.problems.length > 0)
      .map(candidate => ({ index: candidate.index, problems: candidate.problems }));
  }

  return {
    crispFlakes,
    clusters,
    uncertain: partition.uncertain.map(entry => ({
      trace: entry.item,
      kind: entry.label,
      confidence: entry.confidence,
    })),
    ...(culpritCommit ? { culpritCommit } : {}),
    bisectedCommits,
    ...(bisectStopped ? { bisectStopped } : {}),
    fixAccepted,
    chosenBranch,
    ...(chosenFix ? { chosenFix } : {}),
    rejectedCandidates,
  };
}
