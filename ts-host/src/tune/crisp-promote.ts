/**
 * Automatic `pluggable` crisp fast-path promotion (`plans/NATLANG_COMPILER.md` §6.3, W2.2).
 *
 * Verifies synthesized TypeScript fast-paths (`{ hit: true, value } | { hit: false }`)
 * against separate development and confirmation trace partitions with a strict 0-FP rule,
 * and wraps promoted fast-paths so misses, errors, or `ensures:` violations fall through
 * seamlessly to the compiled neural implementation.
 */

export type CrispFastPathResult<R> = { hit: true; value: R } | { hit: false };
export type CrispFastPathFn<A extends unknown[], R> = (...args: A) => CrispFastPathResult<R> | Promise<CrispFastPathResult<R>>;

export interface CrispTraceCase<A extends unknown[], R> {
  id: string;
  args: A;
  expected: R;
}

export interface CrispPartitionStats {
  total: number;
  hits: number;
  correctHits: number;
  falsePositives: number;
  errors: number;
  coverage: number;
}

export interface CrispVerificationReport {
  promoted: boolean;
  reason: string;
  dev: CrispPartitionStats;
  confirmation: CrispPartitionStats;
}

function defaultEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

async function evaluatePartition<A extends unknown[], R>(
  candidate: CrispFastPathFn<A, R>,
  cases: readonly CrispTraceCase<A, R>[],
  same: (a: R, b: R) => boolean,
  ensures?: (value: R, args: A) => boolean | Promise<boolean>,
): Promise<CrispPartitionStats> {
  let hits = 0;
  let correctHits = 0;
  let falsePositives = 0;
  let errors = 0;

  for (const item of cases) {
    try {
      const res = await candidate(...item.args);
      if (!res.hit) continue;
      hits += 1;
      const contractOk = ensures ? await ensures(res.value, item.args) : true;
      if (contractOk && same(res.value, item.expected)) {
        correctHits += 1;
      } else {
        falsePositives += 1;
      }
    } catch {
      errors += 1;
    }
  }

  return {
    total: cases.length,
    hits,
    correctHits,
    falsePositives,
    errors,
    coverage: cases.length > 0 ? hits / cases.length : 0,
  };
}

/**
 * Verify a synthesized crisp fast-path on disjoint dev and confirmation partitions.
 * Promotion requires: 0 false positives on BOTH partitions, 0 errors, and >= minCoverage (default 15%).
 */
export async function verifyCrispCandidate<A extends unknown[], R>(
  candidate: CrispFastPathFn<A, R>,
  partitions: {
    dev: readonly CrispTraceCase<A, R>[];
    confirmation: readonly CrispTraceCase<A, R>[];
  },
  options: {
    minCoverage?: number;
    same?: (a: R, b: R) => boolean;
    ensures?: (value: R, args: A) => boolean | Promise<boolean>;
  } = {},
): Promise<CrispVerificationReport> {
  const minCoverage = options.minCoverage ?? 0.15;
  const same = options.same ?? defaultEqual;

  if (!partitions.dev.length || !partitions.confirmation.length) {
    const empty: CrispPartitionStats = { total: 0, hits: 0, correctHits: 0, falsePositives: 0, errors: 0, coverage: 0 };
    return { promoted: false, reason: 'both dev and confirmation partitions must be nonempty', dev: empty, confirmation: empty };
  }

  const dev = await evaluatePartition(candidate, partitions.dev, same, options.ensures);
  const confirmation = await evaluatePartition(candidate, partitions.confirmation, same, options.ensures);

  if (dev.falsePositives > 0 || confirmation.falsePositives > 0) {
    return {
      promoted: false,
      reason: `nonzero false positives (dev=${dev.falsePositives}, confirmation=${confirmation.falsePositives})`,
      dev,
      confirmation,
    };
  }
  if (dev.errors > 0 || confirmation.errors > 0) {
    return {
      promoted: false,
      reason: `crisp fast-path threw errors (dev=${dev.errors}, confirmation=${confirmation.errors})`,
      dev,
      confirmation,
    };
  }
  if (dev.coverage < minCoverage || confirmation.coverage < minCoverage) {
    return {
      promoted: false,
      reason: `coverage below floor ${minCoverage} (dev=${dev.coverage.toFixed(3)}, confirmation=${confirmation.coverage.toFixed(3)})`,
      dev,
      confirmation,
    };
  }
  return { promoted: true, reason: 'verified: 0 false positives and coverage above floor', dev, confirmation };
}

export interface CrispWrapperMetrics {
  crispHits: number;
  neuralFallbacks: number;
  contractRejections: number;
  crispErrors: number;
}

/**
 * Wrap a verified crisp fast-path in front of a compiled neural fallback.
 * Any miss (`hit: false`), thrown error, or failed `ensures` check transparently runs `nlFallback`.
 */
export function wrapCrispFastPath<A extends unknown[], R>(
  crispFn: CrispFastPathFn<A, R>,
  nlFallback: (...args: A) => Promise<R> | R,
  options: {
    enabled?: boolean;
    ensures?: (value: R, args: A) => boolean | Promise<boolean>;
  } = {},
): {
  call: (...args: A) => Promise<R>;
  metrics: CrispWrapperMetrics;
} {
  const metrics: CrispWrapperMetrics = {
    crispHits: 0,
    neuralFallbacks: 0,
    contractRejections: 0,
    crispErrors: 0,
  };
  const enabled = options.enabled ?? true;

  const call = async (...args: A): Promise<R> => {
    if (enabled) {
      try {
        const res = await crispFn(...args);
        if (res.hit) {
          const ok = options.ensures ? await options.ensures(res.value, args) : true;
          if (ok) {
            metrics.crispHits += 1;
            return res.value;
          }
          metrics.contractRejections += 1;
        }
      } catch {
        metrics.crispErrors += 1;
      }
    }
    metrics.neuralFallbacks += 1;
    return await nlFallback(...args);
  };

  return { call, metrics };
}

export interface PromotedCaseSource {
  /** Guard expression or function `(args) => boolean` matching the fast-path domain. */
  whenSource: string;
  /** Body expression or function `(args) => value` producing the crisp result. */
  runSource: string;
  /** Optional service import lines from `natlang:services`. */
  imports?: readonly string[];
}

/**
 * Emit a standard `cases.ts` compilation module from promoted crisp fast-path sources so they execute
 * through the kernel's `runCrispCase` pipeline (preserving type checks, refinements, effects, capture, and audits).
 */
export function buildCompiledCasesModule(
  promotedCases: readonly PromotedCaseSource[],
  report?: CrispVerificationReport,
): string {
  if (report && !report.promoted) {
    throw new Error(`cannot emit compiled cases.ts for unverified crisp candidate: ${report.reason}`);
  }
  const importBlock = [...new Set(promotedCases.flatMap(c => c.imports ?? []))].join('\n');
  const entries = promotedCases.map(c => `  {\n    when: ${c.whenSource.trim()},\n    run: ${c.runSource.trim()},\n  }`);
  return `${importBlock ? importBlock + '\n\n' : ''}export const cases = [\n${entries.join(',\n')}\n];\n`;
}
