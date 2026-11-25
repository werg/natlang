/**
 * Four-Tier Scoped Experience KB & Stage A/B Serving Bridge (`plans/NATLANG_COMPILER.md` §7, W4.1).
 *
 * Scopes: `natlang:global` | `group:<domain>` | `repo:<sha>` | `fn:<siteId>`.
 * Enforces pre-top-k `excludeTasks` filtering (matching ~/hindsight-kb canonical retrieval rules)
 * and uses `pluggable` policies for `whichScopes` and `worthRemembering`.
 */
import { pluggable, type PluggableSetting } from '../runtime/pluggable.js';

export type ExperienceScope =
  | 'natlang:global'
  | `group:${string}`
  | `repo:${string}`
  | `fn:${string}`;

export interface ExperienceRecord {
  recordId: string;
  scope: ExperienceScope;
  /** Normalized lookup key (128-d or hidden-state projection). */
  key: readonly number[];
  /** Human/model-readable demonstration or repair lesson (Stage A) or compacted summary (Stage B). */
  text: string;
  /** Optional Neuralese compacted state ID (Stage B) or 256-d value vector (Stage C). */
  nativeStateId?: string | null;
  valueVector?: readonly number[] | null;
  /** Task lineage IDs for strict pre-top-k exclusion (never leak held/source task into itself). */
  taskIds: readonly string[];
  strength: number;
  kind: 'success' | 'repair' | 'doc' | 'guidance';
}

export interface ScopeSelectionContext {
  siteId?: string;
  repoSha?: string;
  taskGroup?: string;
  includeGlobal?: boolean;
}

export interface RetrievalRequest {
  queryKey: readonly number[];
  scopes: readonly ExperienceScope[];
  excludeTasks?: readonly string[];
  topK?: number;
  threshold?: number;
  nullMass?: number;
}

export interface RetrievedExperience {
  record: ExperienceRecord;
  similarity: number;
  weight: number;
}

export function whichScopesCrisp(ctx: ScopeSelectionContext): ExperienceScope[] {
  const scopes: ExperienceScope[] = [];
  if (ctx.includeGlobal ?? true) scopes.push('natlang:global');
  if (ctx.taskGroup) scopes.push(`group:${ctx.taskGroup}`);
  if (ctx.repoSha) scopes.push(`repo:${ctx.repoSha}`);
  if (ctx.siteId) scopes.push(`fn:${ctx.siteId}`);
  return scopes;
}

export function createWhichScopesPolicy(
  options: { mode?: PluggableSetting; nl?: (ctx: ScopeSelectionContext) => Promise<ExperienceScope[]> | ExperienceScope[] } = {},
): (ctx: ScopeSelectionContext) => Promise<ExperienceScope[]> {
  return pluggable({ crisp: whichScopesCrisp, nl: options.nl ?? whichScopesCrisp }, options.mode, {
    default: 'crisp',
    name: 'kb:whichScopes',
  });
}

export interface AdmissionCandidate {
  taskId: string;
  verifiedSuccess: boolean;
  wasRepairWin: boolean;
  heldExcluded: boolean;
  textLength: number;
}

export function worthRememberingCrisp(cand: AdmissionCandidate): boolean {
  if (cand.heldExcluded) return false;
  if (!cand.verifiedSuccess && !cand.wasRepairWin) return false;
  return cand.textLength > 0;
}

export function createWorthRememberingPolicy(
  options: { mode?: PluggableSetting; nl?: (cand: AdmissionCandidate) => Promise<boolean> | boolean } = {},
): (cand: AdmissionCandidate) => Promise<boolean> {
  return pluggable({ crisp: worthRememberingCrisp, nl: options.nl ?? worthRememberingCrisp }, options.mode, {
    default: 'crisp',
    name: 'kb:worthRemembering',
  });
}

function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length || a.length === 0) return -1;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = b[i]!;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const denom = Math.sqrt(na * nb);
  return denom > 0 ? dot / denom : -1;
}

/**
 * In-memory / Stage-A/B reference Experience KB store mirroring ~/hindsight-kb's
 * thresholded quadratic retrieval with explicit null mass and pre-top-k `excludeTasks` filtering.
 */
export class ScopedExperienceStore {
  private readonly records = new Map<string, ExperienceRecord>();
  private readonly excludedHeldIds: ReadonlySet<string>;

  constructor(excludedHeldIds: Iterable<string> = []) {
    this.excludedHeldIds = new Set(excludedHeldIds);
  }

  get size(): number {
    return this.records.size;
  }

  add(record: ExperienceRecord): boolean {
    if (record.strength <= 0) return false;
    for (const tid of record.taskIds) {
      if (this.excludedHeldIds.has(tid)) {
        throw new Error(`Refusing to admit held-excluded task ${tid} into Experience KB`);
      }
    }
    this.records.set(record.recordId, record);
    return true;
  }

  /** Retrieve top-k records after strict pre-top-k scope and `excludeTasks` filtering. */
  retrieve(req: RetrievalRequest): { items: RetrievedExperience[]; evidenceGate: number } {
    const scopeSet = new Set(req.scopes);
    const excludeSet = new Set(req.excludeTasks ?? []);
    const topK = req.topK ?? 3;
    const threshold = req.threshold ?? 0.2;
    const nullMass = req.nullMass ?? 0.1;

    const scored: { record: ExperienceRecord; sim: number; rawWeight: number }[] = [];
    for (const rec of this.records.values()) {
      if (!scopeSet.has(rec.scope)) continue;
      if (rec.strength <= 0) continue;
      // Pre-top-k task exclusion (matches ~/hindsight-kb canonical policy)
      let excluded = false;
      for (const tid of rec.taskIds) {
        if (excludeSet.has(tid)) {
          excluded = true;
          break;
        }
      }
      if (excluded) continue;

      const sim = cosineSimilarity(req.queryKey, rec.key);
      if (sim <= threshold) continue;
      const diff = sim - threshold;
      const rawWeight = rec.strength * diff * diff;
      scored.push({ record: rec, sim, rawWeight });
    }

    scored.sort((a, b) => b.rawWeight - a.rawWeight);
    const kept = scored.slice(0, topK);
    const totalWeight = kept.reduce((acc, item) => acc + item.rawWeight, 0);
    const denom = nullMass + totalWeight;
    const evidenceGate = denom > 0 ? totalWeight / denom : 0;

    return {
      items: kept.map(item => ({
        record: item.record,
        similarity: item.sim,
        weight: denom > 0 ? item.rawWeight / denom : 0,
      })),
      evidenceGate,
    };
  }
}

/** Format Stage-A retrieved demonstrations (or Stage-B compacted references) for prompt injection. */
export function formatStageADemonstrations(items: readonly RetrievedExperience[], maxChars = 4000): string {
  if (!items.length) return '';
  const blocks: string[] = ['[RETRIEVED VERIFIED EXPERIENCE]'];
  let used = blocks[0]!.length;
  for (let i = 0; i < items.length; i++) {
    const entry = items[i]!;
    const line = `\n(${i + 1}) [${entry.record.scope} | ${entry.record.kind} | w=${entry.weight.toFixed(2)}] ${entry.record.text}`;
    if (used + line.length > maxChars && i > 0) break;
    blocks.push(line);
    used += line.length;
  }
  return blocks.join('');
}
