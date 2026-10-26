export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type BudgetRound = RecordMeta & WorkRecord & {
  communityId: string;
  proposalIds: string[];
  availableBudget: Money;
  ruleSetId: string;
  facilitatorId: string;
};
export type CivicProposal = RecordMeta & WorkRecord & {
  caseId: string;
  proposerId: string;
  intendedBenefit: string;
  requiredWork: string;
  beneficiaryScope: string;
};
export type ProposalRelationship = RecordMeta & WorkRecord & {
  caseId: string;
  proposalIds: string[];
  relation: 'duplicate'|'complementary'|'conflicting'|'independent';
  preservedIntent: string;
};
export type ProposalFeasibility = RecordMeta & WorkRecord & {
  caseId: string;
  proposalId: string;
  costEstimate: Money;
  constraints: string[];
  missingAssessment: string;
};
export type CivicPortfolio = RecordMeta & WorkRecord & {
  caseId: string;
  proposalIds: string[];
  totalCost: Money;
  sharedWorkIds: string[];
  tradeoff: string;
};
export type BudgetDecisionPacket = RecordMeta & WorkRecord & {
  caseId: string;
  portfolioIds: string[];
  explanation: string;
  unresolvedQuestions: string[];
  decisionRule: string;
};
export type DraftCivicProposal = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    proposerId?: string;
    intendedBenefit?: string;
    requiredWork?: string;
    beneficiaryScope?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type BudgetFacilitator = { id: string };
export type PortfolioChange = { revisedEstimates: EvidenceItem[]; proposerReviews: EvidenceItem[]; locationConstraints: EvidenceItem[]; deliveryChanges: EvidenceItem[] };
export type BudgetDossier = { round: BudgetRound; proposals: CivicProposal[]; relationships: ProposalRelationship[]; feasibility: ProposalFeasibility[]; portfolios: CivicPortfolio[]; decisionPackets: BudgetDecisionPacket[]; evidence: EvidenceItem[]; checkpoint: BudgetCheckpoint };
export type EnablingWorkNode = { id: string; work: string; cost: Money; sourceIds: string[]; enablesProposalIds: string[]; status: 'supported'|'unresolved' };
export type PortfolioCostLine = { lineId: string; proposalId?: string; cost: Money; sourceRefs: string[]; status: 'supported'|'unresolved' };
export type ExactPortfolioSum = { total: Money; withinBudget: boolean; unresolved: string[]; duplicatedLineIds: string[] };
export type PortfolioLedger = { nodes: EnablingWorkNode[]; proposalIds: string[]; costLines?: PortfolioCostLine[]; exactTotal: Money; withinBudget: boolean; unresolvedCosts: string[]; proposerReviewsNeeded: string[]; }
export type BudgetCheckpoint = { edition: string; proposals: CivicProposal[]; relationships: ProposalRelationship[]; feasibility: ProposalFeasibility[]; portfolios: CivicPortfolio[]; selectedProposalIds: string[]; deliveryChanges: EvidenceItem[]; invalidatedPortfolioIds: string[]; };
export type BudgetProposalSession = { application: 'Participatory Budgeting Facilitator'; status: 'ready-for-deliberation'|'blocked'; relationships: ProposalRelationship[]; feasibility: ProposalFeasibility[]; portfolios: CivicPortfolio[]; enablingWork: PortfolioLedger; packet: BudgetDecisionPacket; invalidatedPortfolioIds: string[]; blockers: string[]; actionBoundary: string; resumeState: BudgetCheckpoint; };
