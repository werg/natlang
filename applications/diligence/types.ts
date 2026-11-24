/** A section of an untrusted vendor, paper, or library dossier: its id, title, untrusted text, and subsections. */
export type Section = {
  id: string;
  title: string;
  text: Untrusted<string>;
  sections: Section[];
  /** Optional precomputed Neuralese compact view slot for large section payloads. */
  viewSlot?: { view: Neuralese<string> };
};

/** One candidate vendor, library, or technical paper under diligence review. */
export type Dossier = {
  id: string;
  name: string;
  root: Section;
};

/** A verbatim quote grounded in a section of the dossier. */
export type Citation = {
  sectionId: string;
  quote: string;
};

/** Bottom-up grounded digest of a section and its subsections. */
export type SectionDigest = {
  sectionId: string;
  title: string;
  summary: string;
  keyClaims: string[];
  citations: Citation[];
  children: SectionDigest[];
};

/** One rubric criterion against which every dossier is scored. */
export type RubricCriterion = {
  id: string;
  name: string;
  guidance: string;
  weight?: number;
};

/** Calibrated rubric verdict for a single criterion. */
export type CriterionVerdict = 'strong' | 'acceptable' | 'concern' | 'dealbreaker';

/** Overall diligence recommendation derived from weighted rubric assessments and dealbreaker checks. */
export type DiligenceRecommendation = 'adopt' | 'adopt-with-mitigations' | 'reject';

/** Scored and optionally escalated assessment for one criterion on one dossier. */
export type CriterionAssessment = {
  criterionId: string;
  criterionName: string;
  weight: number;
  verdict: CriterionVerdict;
  confidence: number;
  escalated: boolean;
  escalationReason?: string;
  initialVerdict?: CriterionVerdict;
};

/** Complete grounded evaluation of one candidate dossier across the rubric. */
export type DossierScore = {
  dossierId: string;
  dossierName: string;
  digest: SectionDigest;
  assessments: CriterionAssessment[];
  dealbreakerCount: number;
  concernCount: number;
  strongCount: number;
  acceptableCount: number;
  escalatedCount: number;
  /** Weighted rubric score normalized to [0, 1] using RubricCriterion.weight (default 1). */
  weightedScore: number;
  /** Sum of effective criterion weights applied to this dossier. */
  totalWeight: number;
  /** Overall recommendation derived from weightedScore and dealbreaker/concern counts. */
  recommendation: DiligenceRecommendation;
};

/** Summary ranking entry from debiased pairwise Copeland comparison. */
export type RankedDossierSummary = {
  dossierId: string;
  dossierName: string;
  wins: number;
  score: number;
  weightedScore: number;
  recommendation: DiligenceRecommendation;
};

/** Final diligence report ranking all evaluated candidates. */
export type DiligenceReport = {
  goal: string;
  carryMode: 'text' | 'neuralese';
  viewsUsed: number;
  ranked: DossierScore[];
  rankings: RankedDossierSummary[];
  totalEscalations: number;
};
