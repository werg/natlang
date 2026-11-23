/**
 * Automated pull-request review on a small local model:
 * - Binds one reusable `check_hunk.nl` per team rule with `withContext` (`@natlang/node/std`), keeping its
 *   `ensures:` quote-grounding helper intact so the model can never hallucinate code quotes.
 * - Fans out over `(hunks x rules)` concurrently with `semanticMap`.
 * - Gates the merge decision with `escalate` using an asymmetric risk floor (`risk: { block: 0.25 }`): even if the
 *   local model leans approve/request-changes, a >= 25% chance of "block" escalates to the stronger model.
 * - Ranks findings by urgency to fix with debiased pairwise `semanticRank`.
 */
import { escalate, semanticMap, semanticRank, withContext } from '@natlang/node/std';
import checkHunk from './check_hunk.nl';
import mergeVerdict from './merge_verdict.nl';
import type { Finding, Hunk, ReviewReport } from './types.js';

export type Rule = { name: string; guidance: string };

export async function reviewPullRequest(hunks: Hunk[], rules: Rule[]): Promise<ReviewReport> {
  // Bind check_hunk once per rule while keeping its protected `quotesMatch` verifier in context.
  const checks = rules.map(rule => ({
    rule,
    run: withContext(checkHunk, { rule: `${rule.name}: ${rule.guidance}` }),
  }));
  const pairs = hunks.flatMap(hunk => checks.map(check => ({ hunk, check })));
  const reviews = await semanticMap(pairs, ({ hunk, check }) => check.run(hunk), { concurrency: 6 });
  const rawFindings: Finding[] = reviews.flatMap(review => review.findings);
  const findings = rawFindings.length > 1
    ? (await semanticRank(rawFindings, 'most critical engineering risk to fix before merging')).ranked
    : rawFindings;
  // Asymmetric safety gate: escalate when unsure (< 0.85) OR when P(block) >= 0.25.
  const decideMerge = escalate(mergeVerdict, { to: 'teacher', floor: 0.85, risk: { block: 0.25 } });
  const decision = await decideMerge(findings);
  return {
    verdict: decision.value,
    confidence: decision.confidence,
    escalated: decision.escalated,
    ...(decision.reason ? { escalationReason: decision.reason } : {}),
    findings,
  };
}
