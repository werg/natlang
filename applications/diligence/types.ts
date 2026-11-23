/** A section of an untrusted vendor, paper, or library dossier: its id, title, untrusted text, and subsections. */
export type Section = {
  id: string;
  title: string;
  text: Untrusted<string>;
  sections: Section[];
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

/** Scored and optionally escalated assessment for one criterion on one dossier. */
export type CriterionAssessment = {
  criterionId: string;
  criterionName: string;
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
};

/** Summary ranking entry from debiased pairwise Copeland comparison. */
export type RankedDossierSummary = {
  dossierId: string;
  dossierName: string;
  wins: number;
  score: number;
};

/** Final diligence report ranking all evaluated candidates. */
export type DiligenceReport = {
  goal: string;
  ranked: DossierScore[];
  rankings: RankedDossierSummary[];
  totalEscalations: number;
};
