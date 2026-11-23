/**
 * Technical & Vendor Diligence (WS6 §3 A10):
 * - Marks every dossier section's text as `Untrusted<string>` at the boundary.
 * - Recursively digests each dossier bottom-up via `digestSection.nl` (`recursive: structural`), enforcing
 *   verbatim quote grounding through its protected `ensures: citationsMatch(result, section)` postcondition.
 * - Scores every rubric criterion with calibrated `readout: decision` and asymmetric `escalate`
 *   (`floor: 0.82, risk: { dealbreaker: 0.2 }`) so even a 20% probability of a dealbreaker escalates to `teacher`.
 * - Ranks candidate vendors/libraries via debiased pairwise Copeland `semanticRank`.
 */
import { untrusted } from '@natlang/node';
import { escalate, semanticMap, semanticRank } from '@natlang/node/std';
import digestSection from './digestSection.nl';
import scoreCriterion from './scoreCriterion.nl';
import type {
  CriterionAssessment,
  DiligenceReport,
  Dossier,
  DossierScore,
  RubricCriterion,
  Section,
} from './types.js';

export type DiligenceOptions = {
  /** Decision goal used for final debiased pairwise ranking of candidates. */
  goal?: string;
  /** Target model name for asymmetric escalation (default `'teacher'`). */
  escalateTo?: string;
  /** Confidence floor below which criterion scoring escalates (default `0.82`). */
  floor?: number;
  /** Asymmetric risk floor for `'dealbreaker'` verdicts (default `0.2`). */
  dealbreakerRisk?: number;
  /** Maximum concurrent semantic calls (default `6`). */
  concurrency?: number;
};

/** Recursively mark every section's text in a section tree as `Untrusted<string>`. */
export function markSectionUntrusted(section: Section, dossierName: string): Section {
  return {
    ...section,
    text: untrusted(section.text, `dossier ${dossierName} / ${section.title}`),
    sections: section.sections.map(child => markSectionUntrusted(child, dossierName)),
  };
}

/** Mark every section of a dossier as `Untrusted<string>` at the ingestion boundary. */
export function markDossierUntrusted(dossier: Dossier): Dossier {
  return {
    ...dossier,
    root: markSectionUntrusted(dossier.root, dossier.name),
  };
}

/**
 * Digest each candidate dossier recursively, score every rubric criterion with asymmetric dealbreaker escalation,
 * and rank all candidates via debiased pairwise `semanticRank`.
 */
export async function evaluateDossiers(
  dossiers: readonly Dossier[],
  rubric: readonly RubricCriterion[],
  options: DiligenceOptions = {},
): Promise<DiligenceReport> {
  const goal = options.goal ?? 'safest, best-evidenced candidate with zero dealbreakers and strongest rubric coverage';
  const to = options.escalateTo ?? 'teacher';
  const floor = options.floor ?? 0.82;
  const dealbreakerRisk = options.dealbreakerRisk ?? 0.2;
  const concurrency = options.concurrency ?? 6;

  // Step 1: Recursive bottom-up digestion with verbatim citation enforcement on every node.
  const digested = await semanticMap(
    dossiers,
    async dossier => ({
      dossier,
      digest: await digestSection(dossier.root),
    }),
    { concurrency },
  );

  // Step 2: Score every (dossier, criterion) pair with asymmetric dealbreaker risk escalation.
  const judgeCriterion = escalate(scoreCriterion, {
    to,
    floor,
    risk: { dealbreaker: dealbreakerRisk },
  });

  const pairs = digested.flatMap(entry =>
    rubric.map(criterion => ({ dossier: entry.dossier, digest: entry.digest, criterion })),
  );

  const scoredPairs = await semanticMap(
    pairs,
    async ({ dossier, digest, criterion }) => {
      const decision = await judgeCriterion(dossier.name, digest, criterion);
      const assessment: CriterionAssessment = {
        criterionId: criterion.id,
        criterionName: criterion.name,
        verdict: decision.value,
        confidence: decision.confidence,
        escalated: decision.escalated,
        ...(decision.reason ? { escalationReason: decision.reason } : {}),
        ...(decision.initialValue !== undefined ? { initialVerdict: decision.initialValue } : {}),
      };
      return { dossierId: dossier.id, assessment };
    },
    { concurrency },
  );

  // Exact TypeScript aggregation of criterion verdicts per dossier.
  const assessmentsByDossier = new Map<string, CriterionAssessment[]>();
  for (const { dossierId, assessment } of scoredPairs) {
    const bucket = assessmentsByDossier.get(dossierId);
    if (bucket) bucket.push(assessment);
    else assessmentsByDossier.set(dossierId, [assessment]);
  }

  const dossierScores: DossierScore[] = digested.map(({ dossier, digest }) => {
    const assessments = assessmentsByDossier.get(dossier.id) ?? [];
    return {
      dossierId: dossier.id,
      dossierName: dossier.name,
      digest,
      assessments,
      dealbreakerCount: assessments.filter(a => a.verdict === 'dealbreaker').length,
      concernCount: assessments.filter(a => a.verdict === 'concern').length,
      strongCount: assessments.filter(a => a.verdict === 'strong').length,
      acceptableCount: assessments.filter(a => a.verdict === 'acceptable').length,
      escalatedCount: assessments.filter(a => a.escalated).length,
    };
  });

  // Step 3: Debiased pairwise Copeland ranking via semanticRank.
  const ranking = dossierScores.length > 1
    ? await semanticRank(dossierScores, goal, { debiased: true, concurrency })
    : {
        ranked: dossierScores,
        scores: dossierScores.map((item, index) => ({ item, index, wins: 0, score: 0 })),
      };

  const totalEscalations = dossierScores.reduce((sum, item) => sum + item.escalatedCount, 0);

  return {
    goal,
    ranked: ranking.ranked,
    rankings: ranking.scores.map(entry => ({
      dossierId: entry.item.dossierId,
      dossierName: entry.item.dossierName,
      wins: entry.wins,
      score: entry.score,
    })),
    totalEscalations,
  };
}
