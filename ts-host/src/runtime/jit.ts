/**
 * Multi-tier JIT compiler controller & runtime closure cache (`plans/NATLANG_COMPILER.md` §5, §6, W2.3).
 *
 * Portable across Node, Workerd, and Browser (`@noble/hashes/sha256`, no Node built-ins).
 * Tier choice is a `pluggable` policy (crisp default + natural-language override).
 */
import { sha256 } from '@noble/hashes/sha2.js';
import { bytesToHex } from '@noble/hashes/utils.js';
import { pluggable, suppliedNl, type PluggableSetting } from './pluggable.js';

export type JitTier = '0' | '0.5' | '1' | '1.5' | '2' | 'crisp';

export interface SiteStats {
  siteId: string;
  callCount: number;
  verifiedExampleCount: number;
  refinementFailureRate: number;
  meanDecisionConfidence: number | null;
  repairRate: number;
  isSynthesizedLiteral: boolean;
  hasCrispFastPath: boolean;
  hasCompiledWeights: boolean;
  hasTier15Weights: boolean;
  hasTier2Weights: boolean;
  compilerAvailable: boolean;
}

export interface TierThresholds {
  /** Minimum calls before promoting from Tier 0 -> Tier 0.5 primed state. */
  primedMinCalls?: number;
  /** Minimum calls before promoting to Tier 1 compiled weights (synthesized literals compile at 0). */
  compileMinCalls?: number;
  /** Minimum verified examples to recommend Tier 1.5 latent-manifold TTT. */
  refineMinExamples?: number;
  /** Minimum verified examples to recommend Tier 2 full tune. */
  tuneMinExamples?: number;
  /** Refinement failure rate above which a site escalates toward refinement/tuning. */
  failureRateThreshold?: number;
  /** Decision confidence floor below which a site escalates toward refinement/tuning. */
  confidenceFloor?: number;
}

export interface CompiledProvenance {
  siteId: string;
  headerSha: string;
  compilerSha: string;
  policySha: string;
  baseGgufSha256?: string;
}

export interface CompiledSiteEntry {
  cacheKey: string;
  tier: '0.5' | '1' | '1.5' | '2';
  provenance: CompiledProvenance;
  adapterId: string | null;
  nativeStateId: string | null;
  createdAtMs: number;
  hits: number;
}

const textEncoder = new TextEncoder();

/** Deterministic SHA-256 hex digest of UTF-8 text (browser/workerd/Node portable). */
export function sha256Hex(text: string): string {
  return bytesToHex(sha256(textEncoder.encode(text)));
}

/** Exact provenance cache key binding site, header, compiler, policy, and base GGUF identity. */
export function compiledCacheKey(p: CompiledProvenance): string {
  const canonical = JSON.stringify({
    baseGgufSha256: p.baseGgufSha256 ?? '',
    compilerSha: p.compilerSha,
    headerSha: p.headerSha,
    policySha: p.policySha,
    siteId: p.siteId,
  });
  return sha256Hex(canonical);
}

/** Crisp tier-selection policy (hot-path twin of `chooseTier.nl`). */
export function chooseTierCrisp(stats: SiteStats, thresholds: TierThresholds = {}): JitTier {
  const primedMin = thresholds.primedMinCalls ?? 2;
  const compileMin = thresholds.compileMinCalls ?? 3;
  const refineMin = thresholds.refineMinExamples ?? 5;
  const tuneMin = thresholds.tuneMinExamples ?? 100;
  const failThresh = thresholds.failureRateThreshold ?? 0.15;
  const confFloor = thresholds.confidenceFloor ?? 0.75;

  if (stats.hasCrispFastPath) return 'crisp';
  if (stats.hasTier2Weights) return '2';
  if (stats.hasTier15Weights) return '1.5';

  const struggling =
    stats.refinementFailureRate >= failThresh ||
    stats.repairRate >= failThresh ||
    (stats.meanDecisionConfidence !== null && stats.meanDecisionConfidence < confFloor);

  if (stats.verifiedExampleCount >= tuneMin && struggling) return '2';
  if (stats.compilerAvailable && stats.verifiedExampleCount >= refineMin && (struggling || !stats.hasCompiledWeights)) {
    return '1.5';
  }
  if (stats.hasCompiledWeights) return '1';
  if (stats.compilerAvailable && (stats.isSynthesizedLiteral || stats.callCount >= compileMin)) {
    return '1';
  }
  if (stats.callCount >= primedMin) return '0.5';
  return '0';
}

export interface ChooseTierPolicyOptions {
  mode?: PluggableSetting;
  nl?: (stats: SiteStats, thresholds?: TierThresholds) => Promise<JitTier> | JitTier;
  thresholds?: TierThresholds;
}

/** Create the pluggable `chooseTier` policy (crisp default; supports `nl` and `shadow` modes). */
export function createChooseTierPolicy(options: ChooseTierPolicyOptions = {}): (stats: SiteStats) => Promise<JitTier> {
  const thresholds = options.thresholds ?? {};
  const crisp = (stats: SiteStats): JitTier => chooseTierCrisp(stats, thresholds);
  const nl = suppliedNl(options.nl, options.mode, 'crisp', 'jit:chooseTier');
  return pluggable({ crisp, nl }, options.mode, { default: 'crisp', name: 'jit:chooseTier' });
}

/** Bounded LRU cache for compiled site/closure artifacts keyed by exact provenance. */
export class CompiledSiteCache {
  private readonly entries = new Map<string, CompiledSiteEntry>();
  private readonly maxEntries: number;

  constructor(maxEntries = 1024) {
    if (!Number.isInteger(maxEntries) || maxEntries < 1) {
      throw new RangeError(`maxEntries must be a positive integer, got ${maxEntries}`);
    }
    this.maxEntries = maxEntries;
  }

  get size(): number {
    return this.entries.size;
  }

  get(provenance: CompiledProvenance): CompiledSiteEntry | undefined {
    const key = compiledCacheKey(provenance);
    const hit = this.entries.get(key);
    if (!hit) return undefined;
    // Refresh LRU order
    this.entries.delete(key);
    hit.hits += 1;
    this.entries.set(key, hit);
    return hit;
  }

  put(entry: Omit<CompiledSiteEntry, 'cacheKey' | 'hits'>): CompiledSiteEntry {
    const key = compiledCacheKey(entry.provenance);
    const full: CompiledSiteEntry = { ...entry, cacheKey: key, hits: 0 };
    if (this.entries.has(key)) this.entries.delete(key);
    this.entries.set(key, full);
    while (this.entries.size > this.maxEntries) {
      const oldest = this.entries.keys().next().value;
      if (oldest !== undefined) this.entries.delete(oldest);
    }
    return full;
  }

  invalidatePolicy(policySha: string): number {
    let removed = 0;
    for (const [key, entry] of this.entries.entries()) {
      if (entry.provenance.policySha === policySha) {
        this.entries.delete(key);
        removed += 1;
      }
    }
    return removed;
  }
}

export interface CompileClosureRequest {
  siteId: string;
  resolvedHeaderText: string;
  partialArgsJson?: string;
  compilerSha: string;
  policySha: string;
  baseGgufSha256?: string;
}

/**
 * Compile-on-creation hook for runtime-synthesized NL literals (`nl\`...\`` with captures)
 * and static partial applications (`.partial(arg)`, `.in(ctx)`).
 */
export async function compileOnCreateClosure(
  cache: CompiledSiteCache,
  req: CompileClosureRequest,
  compileFn: (req: CompileClosureRequest, headerSha: string) => Promise<{
    adapterId: string | null;
    nativeStateId: string | null;
  }>,
): Promise<CompiledSiteEntry> {
  const combined = req.partialArgsJson
    ? `${req.resolvedHeaderText}\n__NATLANG_PARTIAL__:${req.partialArgsJson}`
    : req.resolvedHeaderText;
  const headerSha = sha256Hex(combined);
  const provenance: CompiledProvenance = {
    siteId: req.siteId,
    headerSha,
    compilerSha: req.compilerSha,
    policySha: req.policySha,
    baseGgufSha256: req.baseGgufSha256,
  };
  const cached = cache.get(provenance);
  if (cached) return cached;

  const built = await compileFn(req, headerSha);
  return cache.put({
    tier: '1',
    provenance,
    adapterId: built.adapterId,
    nativeStateId: built.nativeStateId,
    createdAtMs: Date.now(),
  });
}

/**
 * Create a live Mode-P T3 HTTP JIT compiler callback backed by `natlang_neuralese.compiler.t3_server`
 * (`POST /v1/natlang/compile-site`). Plugs directly into `NatlangRuntimeOptions.compiler.jit.compile`.
 */
export function createHttpJitCompiler(
  endpoint: string,
  options: {
    scopes?: readonly string[];
    excludeTasks?: readonly string[];
    timeoutMs?: number;
    fetchImpl?: typeof fetch;
  } = {},
): (args: {
  siteId: string;
  resolvedHeaderText: string;
  partialArgsJson?: string;
}) => Promise<{ adapterId: string | null; nativeStateId: string | null }> {
  const baseUrl = endpoint.replace(/\/+$/, '');
  const doFetch = options.fetchImpl ?? globalThis.fetch;
  return async args => {
    const controller = options.timeoutMs ? new AbortController() : undefined;
    const timer = controller ? setTimeout(() => controller.abort(), options.timeoutMs) : undefined;
    try {
      const response = await doFetch(`${baseUrl}/v1/natlang/compile-site`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          site_id: args.siteId,
          header: args.resolvedHeaderText,
          ...(args.partialArgsJson ? { partial_args_json: args.partialArgsJson } : {}),
          ...(options.scopes ? { scopes: options.scopes } : {}),
          ...(options.excludeTasks ? { exclude_tasks: options.excludeTasks } : {}),
        }),
        ...(controller ? { signal: controller.signal } : {}),
      });
      if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw new Error(`T3 compile-site failed (${response.status}): ${text.slice(0, 200)}`);
      }
      const payload = await response.json() as {
        adapterId?: string | null;
        nativeStateId?: string | null;
      };
      return {
        adapterId: payload.adapterId ?? null,
        nativeStateId: payload.nativeStateId ?? null,
      };
    } finally {
      if (timer) clearTimeout(timer);
    }
  };
}
