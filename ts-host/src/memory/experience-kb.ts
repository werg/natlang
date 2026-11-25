/**
 * Experience-KB policies and Stage-A formatting (`plans/NATLANG_COMPILER.md` §7, W4.1).
 *
 * Scopes: `natlang:global` | `group:<domain>` | `repo:<sha>` | `fn:<siteId>`.
 * The store, its index, held admission and retrieval (with `exclude_tasks` applied before top-k) have one
 * implementation: `ScopedExperienceKB` in `training/neuralese/natlang_neuralese/compiler/hkb_bridge.py` over
 * hindsight-kb. This module holds only the two policies, `whichScopes` and `worthRemembering`, as `pluggable` parts,
 * and the prompt formatting of what the store returned.
 */
import { pluggable, suppliedNl, type PluggableSetting } from '../runtime/pluggable.js';

export type ExperienceScope =
  | 'natlang:global'
  | `group:${string}`
  | `repo:${string}`
  | `fn:${string}`;

/** One hit of `ScopedExperienceKB.retrieve` (`RetrievedExperience` there), in camelCase. */
export interface RetrievedExperience {
  recordId: string;
  scope: ExperienceScope;
  kind: 'success' | 'repair' | 'doc' | 'guidance';
  /** Demonstration or repair lesson (Stage A) or compacted summary (Stage B). */
  text: string;
  taskIds: readonly string[];
  compactedStateId?: string | null;
  similarity: number;
  /** Evidence weight normalized against the null mass. */
  weight: number;
}

export interface ScopeSelectionContext {
  siteId?: string;
  repoSha?: string;
  taskGroup?: string;
  includeGlobal?: boolean;
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
  return pluggable({ crisp: whichScopesCrisp, nl: suppliedNl(options.nl, options.mode, 'crisp', 'kb:whichScopes') }, options.mode, {
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
  return pluggable({ crisp: worthRememberingCrisp, nl: suppliedNl(options.nl, options.mode, 'crisp', 'kb:worthRemembering') }, options.mode, {
    default: 'crisp',
    name: 'kb:worthRemembering',
  });
}

/** Format Stage-A retrieved demonstrations (or Stage-B compacted references) for prompt injection. */
export function formatStageADemonstrations(items: readonly RetrievedExperience[], maxChars = 4000): string {
  if (!items.length) return '';
  const blocks: string[] = ['[RETRIEVED VERIFIED EXPERIENCE]'];
  let used = blocks[0]!.length;
  for (let i = 0; i < items.length; i++) {
    const entry = items[i]!;
    const line = `\n(${i + 1}) [${entry.scope} | ${entry.kind} | w=${entry.weight.toFixed(2)}] ${entry.text}`;
    if (used + line.length > maxChars && i > 0) break;
    blocks.push(line);
    used += line.length;
  }
  return blocks.join('');
}
