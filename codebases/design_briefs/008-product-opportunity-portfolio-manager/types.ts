
export type ID = string; export type Hash = string; export type LocalDate = string; export type DateTime = string; export type Decimal = string; export type Ratio = { numerator: Decimal; denominator: Decimal }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number };
export type Ref = { tenantId: ID; objectId: ID; version: string; hash: Hash };
export type SpanRef = { source: Ref; locatorKind: 'text' | 'table' | 'timecode' | 'region'; locator: string; excerptHash: Hash };

export type FieldConflict = { field: string; alternatives: string[]; evidenceRefs: string[] };

export type Question = { question: string; respondent: string; evidenceNeeded: string };

export type Evidence = { id: string; source: string; version: string; observedAt: string; authority: string; scope: string; text: string };
export type EvidencePage = { evidence: Evidence[]; searchedPartition: string; complete: boolean };

export type GoalContract = { outcome: string; exclusions: string[]; entities: string[]; acceptanceEvidence: string[]; unresolvedScope: string[] };
export type PolicyCandidate = { id: string; authority: string; scope: string; effectiveFrom: LocalDate; effectiveTo: LocalDate; supersedes: string[]; clauses: string[] };
export type PolicySelection = { selected: string[]; precedenceQuestions: string[]; applicableScope: string };

export type GateVerdict = 'pass' | 'fail' | 'needs-review';
export type OpportunityPortfolio = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; strategyId: string; planningHorizon: string; budget: { minorUnits: string; currency: string; scale: number }; capacityUnits: number; sponsorId: string };
export type CustomerNeed = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; segment: string; desiredOutcome: string; obstacle: string; evidenceRefs: Ref[]; uncertainty: string };
export type Intervention = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; needIds: string[]; concept: string; mechanism: string; nonProductAlternative: string; assumptionIds: string[] };
export type OpportunityEstimate = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; interventionId: string; costRange: string; dependencyIds: string[]; confidenceBasis: string };
export type PortfolioOption = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; interventionIds: string[]; totalCost: { minorUnits: string; currency: string; scale: number }; capacityUse: number; tradeoff: string };
export type OpportunityExperiment = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; assumptionId: string; discriminatingTask: string; decisionThreshold: string; ownerId: string };
export type DraftCustomerNeed = { value: { tenantId?: string; id?: string; version?: number; schemaVersion?: string; state?: string; createdAt?: string; updatedAt?: string; createdBy?: string; sourceRefs?: Ref[]; decisionRefs?: Ref[]; sensitivity?: string[]; purposeIds?: string[]; retentionPolicyId?: string; caseId?: string; segment?: string; desiredOutcome?: string; obstacle?: string; evidenceRefs?: Ref[]; uncertainty?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidenceRefs: string[] };

export type PortfolioDecisionRequest = { caseId: string; planningQuestion: string; strategyId: string; planningHorizon: string; sponsorId: string; expectedPortfolioVersion: string; approvedBaselineVersion?: string };
export type OpportunityEvidenceState = { version: string; planningFrames: OpportunityPortfolio[]; customerSignals: EvidencePage[]; costAndCapacityEvidence: EvidencePage[]; strategyRules: PolicyCandidate[]; exactSolverResults: FeasiblePortfolioSet[]; sponsorPriorityEvents: EvidencePage[]; searchedPartitions: string[]; derivedEstimates?: OpportunityEstimate[]; constraintArtifacts?: PortfolioConstraintArtifact[] };

export type NeedMechanism = { segment: string; workflow: string; desiredOutcome: string; obstacle: string; currentWorkaround: string; implementationSuggestion: string; evidenceRefs: string[]; uncertainty: string[] };
export type InterventionMechanism = { needIds: string[]; kind: 'product'|'service'|'process'; mechanism: string; affectedUsers: string; assumptions: string[]; tradeoffs: string[]; lowChangeBaseline: boolean };
export type CostBasisReview = { packageId: string; estimateRefs: string[]; costRange: string; confidence: string; eligibleForReceipt: boolean; blockers: string[] };
export type WorkPackageCostReview = { package: WorkPackage; basis: CostBasisReview; dependencyAttributions: string[] };
export type WorkPackageDecomposition = { packages: WorkPackage[]; costReviews: WorkPackageCostReview[] };
export type SharedDependencyEvidence = { sharedWorkId: string; interventionIds: string[]; evidenceRefs: string[]; sameOperationalDependency: boolean; distinctOutcomeRisks: string[]; costCountOnce: boolean; blocker: string | null };
export type SharedWorkAnalysis = { clusters: SharedBenefitCluster[]; comparisons: SharedDependencyEvidence[] };
export type SharedBenefitCluster = { sharedWorkId: string; benefit: string; interventionIds: string[]; countOnce: boolean; evidenceRefs: string[] };
export type ExperimentDecisionRule = { assumption: string; decisionAffected: string; positiveAction: string; negativeAction: string; inconclusiveAction: string; cheapestCredibleTest: string; evidenceRefs: string[] };

export type WorkPackage = { interventionId: string; packageId: string; workKind: string; costRange: string; ownerSource: string; dependencyIds: string[]; confidence: string };
export type PortfolioStrategyAssessment = { caseId: string; baselineVersion?: string; scenarioFork?: ScenarioFork|null; strategy: GoalContract; constraints: PolicySelection; underlyingNeeds: NeedMechanism[]; interventionMechanisms: InterventionMechanism[]; workPackages: WorkPackage[]; workPackageCostReviews?: WorkPackageCostReview[]; sharedWork: SharedBenefitCluster[]; sharedWorkComparisons?: SharedDependencyEvidence[]; learningRules: ExperimentDecisionRule[]; receiptReviews: PortfolioReceiptAssessment[]; selection: PortfolioSelectionDecision; questions: Question[]; commitAllowed: boolean };
export type PortfolioConstraintArtifact = { planningFrameId: string; strategyId: string; planningHorizon: string; constraintDigest: string; budget: Money; capacity: number; estimateIds: string[]; sharedWorkIds: string[]; solverVersion: string; sourceRefs: string[] };
export type PortfolioReceiptAssessment = { accepted: boolean; receipt: FeasiblePortfolioSet; reason: string; feasibleInterventionSets: string[][]; infeasibleInterventionSets: string[][]; exactCostValues: Money[]; unresolvedEstimateIds: string[]; constraintDigest: string };
export type PortfolioSelectionDecision = { disposition: 'selected'|'review-required'|'no-feasible-option'; selectedInterventionIds: string[]; consideredSets: string[][]; reason: string; displacedWork: string[]; evidenceRefs: string[] };
export type PortfolioDecisionCheckpoint = { caseId: string; revision: number; approvedBaselineVersion: string; assessments: PortfolioStrategyAssessment[]; scenarioForks: ScenarioFork[]; currentSelection: PortfolioSelectionDecision | null; solverReceiptIds: string[]; supersededAssessmentIds: string[]; unresolvedInvestmentApprovals: string[] };
export type PortfolioDecisionRun = { assessment: PortfolioStrategyAssessment; checkpoint: PortfolioDecisionCheckpoint; forkedFromBaseline: boolean; supersededOptionIds: string[]; duplicateSolverReceiptIds: string[] };
export type FeasiblePortfolioSet = { solverVersion: string; inputConstraintDigest: string; budgetMinorUnits: string; currency: string; capacity: number; sourceEstimateIds: string[]; sharedWorkIds: string[]; feasibleInterventionSets: string[][]; infeasibleInterventionSets: string[][]; constraints: string[]; exactCostValues: Money[]; unknownEstimateIds: string[] };
export type ScenarioFork = { scenarioId: string; baselineVersion: string; changedPriorities: string[]; effectiveAt: string; evidenceIds: string[] };
