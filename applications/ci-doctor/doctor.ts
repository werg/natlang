/**
 * CI Doctor (WS6 Tier 2 flagship A5):
 * - Crisp `pluggable` known-flake signature detector handles well-known transient network/runner flakes with 0 LLM calls.
 * - `semanticPartition` classifies and groups remaining traces by `FailureKind` with a confidence floor.
 * - `iterateOn` with `withMeasure` runs a bounded binary-search commit bisect to pinpoint the culprit commit.
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
import { branchAndSelect, semanticPartition, type Branch } from '@natlang/node/std';
import classifyFailure from './classifyFailure.nl';
import proposeFix from './proposeFix.nl';
import type {
  BisectState,
  CiReport,
  Cluster,
  FailureKind,
  FailureTrace,
  FixProposal,
  RejectedCandidate,
} from './types.js';

export type DoctorOptions = {
  /** Pluggable mode for known-flake detection (default `'crisp'`). */
  flakeMode?: PluggableSetting;
  /** Minimum decision confidence for `semanticPartition` (default `0.75`). */
  partitionFloor?: number;
  /** Maximum concurrent classification calls (default `6`). */
  concurrency?: number;
  /** Optional commit tester for bisection: returns true when the commit reproduces the failure. */
  failsAtCommit?: (commit: string, trace: FailureTrace) => Promise<boolean> | boolean;
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
 * Bounded commit bisection using `iterateOn` with a strictly decreasing `withMeasure` interval width.
 * `commits` is ordered oldest-to-newest and the newest is taken as failing (the trace failed there); returns the earliest
 * failing commit in `commits`.
 */
export async function bisectCommits(
  commits: readonly string[],
  failsAtCommit: (commit: string) => Promise<boolean> | boolean,
): Promise<{ culprit?: string; tested: string[] }> {
  if (commits.length === 0) return { tested: [] };
  if (commits.length === 1) {
    const fails = await failsAtCommit(commits[0]!);
    return { ...(fails ? { culprit: commits[0]! } : {}), tested: [commits[0]!] };
  }

  const initial: BisectState = {
    low: 0,
    high: commits.length - 1,
    tested: [],
    done: false,
  };

  const maxSteps = Math.ceil(Math.log2(commits.length)) + 2;
  const finalState = await iterateOn(
    async (state: BisectState): Promise<BisectState> => {
      if (state.low >= state.high) {
        return { ...state, culprit: commits[state.low], done: true };
      }
      const mid = Math.floor((state.low + state.high) / 2);
      const commit = commits[mid]!;
      const failed = await failsAtCommit(commit);
      const tested = [...state.tested, commit];
      const nextLow = failed ? state.low : mid + 1;
      const nextHigh = failed ? mid : state.high;
      const done = nextLow >= nextHigh;
      return {
        low: nextLow,
        high: nextHigh,
        tested,
        ...(done ? { culprit: commits[nextLow] } : {}),
        done,
      };
    },
    initial,
  )
    .withMeasure(state => (state.done ? 0 : Math.max(0, state.high - state.low)))
    .withLimit({ maxSteps })
    .until(state => state.done);

  return {
    ...(finalState.culprit ? { culprit: finalState.culprit } : {}),
    tested: finalState.tested,
  };
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

  const clusterMap = new Map<string, Cluster>();
  const addToCluster = (kind: FailureKind, source: 'crisp' | 'nl', trace: FailureTrace) => {
    const signature = extractFailureSignature(trace, kind);
    const key = `${source}:${signature}`;
    const existing = clusterMap.get(key);
    if (existing) {
      existing.traces.push(trace);
    } else {
      clusterMap.set(key, { kind, signature, source, traces: [trace] });
    }
  };

  for (const trace of crispFlakes) addToCluster('flake', 'crisp', trace);
  for (const [kind, items] of Object.entries(partition.groups) as Array<[FailureKind, FailureTrace[]]>) {
    for (const trace of items) addToCluster(kind, 'nl', trace);
  }

  const regressions = partition.groups.regression ?? [];
  let culpritCommit: string | undefined;
  let bisectedCommits: string[] = [];

  const primaryRegression = regressions[0];
  if (primaryRegression && primaryRegression.commits.length > 0 && options.failsAtCommit) {
    const bisect = await bisectCommits(
      primaryRegression.commits,
      commit => options.failsAtCommit!(commit, primaryRegression),
    );
    culpritCommit = bisect.culprit;
    bisectedCommits = bisect.tested;
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
    clusters: [...clusterMap.values()],
    uncertain: partition.uncertain.map(entry => ({
      trace: entry.item,
      kind: entry.label,
      confidence: entry.confidence,
    })),
    ...(culpritCommit ? { culpritCommit } : {}),
    bisectedCommits,
    fixAccepted,
    chosenBranch,
    ...(chosenFix ? { chosenFix } : {}),
    rejectedCandidates,
  };
}
