/**
 * Technical & Vendor Diligence (WS6 §3 A10):
 * - Marks every dossier section's text as `Untrusted<string>` at the boundary automatically in `evaluateDossiers`.
 * - Recursively digests each dossier bottom-up via `digestSection.nl` (`recursive: structural`), enforcing
 *   verbatim quote grounding through its protected `ensures: citationsMatch(result, section)` postcondition.
 * - Supports optional `Neuralese<string>` views (`neuraleseView` hook or `carry: 'neuralese'` via `semanticReduce`)
 *   for compacting large dossier sections, with crisp fallback to `digestSection` when no view is attached.
 * - Scores every rubric criterion with calibrated `readout: decision` and asymmetric `escalate`
 *   (`floor: 0.82, risk: { dealbreaker: 0.2 }`), applies `RubricCriterion.weight` (default `1`) to compute
 *   `weightedScore` and `recommendation`, and ranks candidates via debiased pairwise Copeland `semanticRank`.
 */
import { untrusted } from '@natlang/node';
import { escalate, semanticMap, semanticRank, semanticReduce, type CarryMode } from '@natlang/node/std';
import digestSection from './digestSection.nl';
import scoreCriterion from './scoreCriterion.nl';
import type {
  CriterionAssessment,
  CriterionVerdict,
  DiligenceRecommendation,
  DiligenceReport,
  Dossier,
  DossierScore,
  RubricCriterion,
  Section,
  SectionDigest,
} from './types.js';

/** Crisp carrier holding an opaque `Neuralese<T>` view reference for large dossier sections. */
export type NeuraleseViewSlot<T = string> = {
  readonly view: Neuralese<T>;
};

/** Optional Neuralese<T> view hook: lets callers supply compact section views and a view-native digester. */
export type DiligenceNeuraleseView = {
  getView?: (section: Section) => NeuraleseViewSlot<string> | undefined | Promise<NeuraleseViewSlot<string> | undefined>;
  digestView?: (section: Section, view: Neuralese<string>) => Promise<SectionDigest>;
};

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
  /** How intermediate dossier digests are carried when compacting large sections (`'text'` default or `'neuralese'`). */
  carry?: CarryMode;
  /** Optional soft-view hook forwarded to `semanticReduce` when `carry: 'neuralese'`. */
  view?: (value: unknown, instructions?: string) => unknown | Promise<unknown>;
  /** Optional readout hook forwarded to `semanticReduce` when `carry: 'neuralese'`. */
  read?: (value: unknown) => unknown | Promise<unknown>;
  /** Optional view compaction instructions forwarded when `carry: 'neuralese'`. */
  viewInstructions?: string;
  /** Optional section-level `Neuralese<string>` view provider / consumer hook. */
  neuraleseView?: DiligenceNeuraleseView;
};

const VERDICT_SCORE: Record<CriterionVerdict, number> = {
  strong: 1.0,
  acceptable: 0.7,
  concern: 0.3,
  dealbreaker: 0.0,
};

/** Effective positive weight for a rubric criterion (defaults to `1`). */
export function effectiveCriterionWeight(criterion: RubricCriterion): number {
  return typeof criterion.weight === 'number' && Number.isFinite(criterion.weight) && criterion.weight > 0
    ? criterion.weight
    : 1;
}

/** Exact weighted rubric score in [0, 1] and recommendation derived from assessments. */
export function summarizeAssessments(assessments: readonly CriterionAssessment[]): {
  weightedScore: number;
  totalWeight: number;
  recommendation: DiligenceRecommendation;
  dealbreakerCount: number;
  concernCount: number;
  strongCount: number;
  acceptableCount: number;
  escalatedCount: number;
} {
  let weightedSum = 0;
  let totalWeight = 0;
  let dealbreakerCount = 0;
  let concernCount = 0;
  let strongCount = 0;
  let acceptableCount = 0;
  let escalatedCount = 0;

  for (const assessment of assessments) {
    const w = typeof assessment.weight === 'number' && Number.isFinite(assessment.weight) && assessment.weight > 0
      ? assessment.weight
      : 1;
    totalWeight += w;
    weightedSum += w * VERDICT_SCORE[assessment.verdict];
    if (assessment.verdict === 'dealbreaker') dealbreakerCount++;
    else if (assessment.verdict === 'concern') concernCount++;
    else if (assessment.verdict === 'strong') strongCount++;
    else if (assessment.verdict === 'acceptable') acceptableCount++;
    if (assessment.escalated) escalatedCount++;
  }

  const weightedScore = totalWeight > 0 ? Math.round((weightedSum / totalWeight) * 10000) / 10000 : 0;
  let recommendation: DiligenceRecommendation;
  if (dealbreakerCount > 0) {
    recommendation = 'reject';
  } else if (weightedScore >= 0.75 && concernCount === 0) {
    recommendation = 'adopt';
  } else if (weightedScore >= 0.45) {
    recommendation = 'adopt-with-mitigations';
  } else {
    recommendation = 'reject';
  }

  return {
    weightedScore,
    totalWeight,
    recommendation,
    dealbreakerCount,
    concernCount,
    strongCount,
    acceptableCount,
    escalatedCount,
  };
}

/** Recursively mark every section's text in a section tree as `Untrusted<string>`. */
export function markSectionUntrusted(section: Section, dossierName: string): Section {
  return {
    ...section,
    text: untrusted(String(section.text), `dossier ${dossierName} / ${section.title}`),
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

async function digestWithOptionalView(
  section: Section,
  options: DiligenceOptions,
  onViewUsed: () => void,
): Promise<SectionDigest> {
  const viewHook = options.neuraleseView;
  const slot = section.viewSlot ?? (viewHook?.getView ? await viewHook.getView(section) : undefined);
  if (slot && viewHook?.digestView) {
    onViewUsed();
    return viewHook.digestView(section, slot.view);
  }

  if (options.carry === 'neuralese' && section.sections.length > 0 && (options.view || viewHook)) {
    const childDigests = await semanticMap(
      section.sections,
      child => digestWithOptionalView(child, options, onViewUsed),
      { concurrency: options.concurrency ?? 6 },
    );
    const rootLeaf: Section = { ...section, sections: [] };
    const rootDigest = await digestSection(rootLeaf);
    const compacted = await semanticReduce<SectionDigest, SectionDigest>(
      childDigests,
      rootDigest,
      (acc, child) => ({
        sectionId: section.id,
        title: section.title,
        summary: [acc.summary, child.summary].filter(Boolean).join(' '),
        keyClaims: [...acc.keyClaims, ...child.keyClaims],
        citations: [...acc.citations, ...child.citations],
        children: [...acc.children, child],
      }),
      {
        carry: 'neuralese',
        ...(options.view ? { view: options.view } : {}),
        ...(options.read ? { read: options.read } : {}),
        viewInstructions: options.viewInstructions ?? 'Compact dossier section digest while preserving all verbatim citations and technical claims.',
      },
    );
    onViewUsed();
    return compacted;
  }

  return digestSection(section);
}

/**
 * Digest each candidate dossier recursively, score every rubric criterion with asymmetric dealbreaker escalation,
 * apply rubric criterion weights to compute overall scores/recommendations, and rank candidates via `semanticRank`.
 */
export async function evaluateDossiers(
  rawDossiers: readonly Dossier[],
  rubric: readonly RubricCriterion[],
  options: DiligenceOptions = {},
): Promise<DiligenceReport> {
  const goal = options.goal ?? 'safest, best-evidenced candidate with zero dealbreakers and strongest rubric coverage';
  const to = options.escalateTo ?? 'teacher';
  const floor = options.floor ?? 0.82;
  const dealbreakerRisk = options.dealbreakerRisk ?? 0.2;
  const concurrency = options.concurrency ?? 6;
  const carryMode: 'text' | 'neuralese' = options.carry ?? 'text';

  // Always fence every dossier section at the boundary before any model sees it.
  const dossiers = rawDossiers.map(markDossierUntrusted);
  let viewsUsed = 0;

  // Step 1: Recursive bottom-up digestion with verbatim citation enforcement (and optional Neuralese views/carry).
  const digested = await semanticMap(
    dossiers,
    async dossier => ({
      dossier,
      digest: await digestWithOptionalView(dossier.root, options, () => {
        viewsUsed++;
      }),
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
        weight: effectiveCriterionWeight(criterion),
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

  // Exact TypeScript aggregation of weighted criterion verdicts per dossier.
  const assessmentsByDossier = new Map<string, CriterionAssessment[]>();
  for (const { dossierId, assessment } of scoredPairs) {
    const bucket = assessmentsByDossier.get(dossierId);
    if (bucket) bucket.push(assessment);
    else assessmentsByDossier.set(dossierId, [assessment]);
  }

  const dossierScores: DossierScore[] = digested.map(({ dossier, digest }) => {
    const assessments = assessmentsByDossier.get(dossier.id) ?? [];
    const summary = summarizeAssessments(assessments);
    return {
      dossierId: dossier.id,
      dossierName: dossier.name,
      digest,
      assessments,
      ...summary,
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
    carryMode,
    viewsUsed,
    ranked: ranking.ranked,
    rankings: ranking.scores.map(entry => ({
      dossierId: entry.item.dossierId,
      dossierName: entry.item.dossierName,
      wins: entry.wins,
      score: entry.score,
      weightedScore: entry.item.weightedScore,
      recommendation: entry.item.recommendation,
    })),
    totalEscalations,
  };
}
