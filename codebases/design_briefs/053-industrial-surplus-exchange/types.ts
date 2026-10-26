export type ID = string;
export type Ref = string;
export type Decimal = string;
export type Ratio = { numerator: string; denominator: string };
export type Quantity = { value: Decimal; unit: string };
export type Money = { minorUnits: string; currency: string; scale: number };
export type DateTime = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type SurplusExchange = RecordMeta & {
  offerIds: ID[];
  needIds: ID[];
  coordinatorId: ID;
  exchangeRegion: string;
  allocationPolicyId: ID;
};
export type SurplusOffer = RecordMeta & {
  caseId: ID;
  ownerId: ID;
  materialFunction: string;
  quantity: Quantity;
  condition: string;
  transferRestrictions: string[];
};
export type ReuseNeed = RecordMeta & {
  caseId: ID;
  recipientId: ID;
  desiredFunction: string;
  essentialProperties: string[];
  adaptationTolerance: string;
};
export type ReuseMatch = RecordMeta & {
  caseId: ID;
  offerIds: ID[];
  needId: ID;
  reuseConcept: string;
  requiredAdaptations: string[];
};
export type ExchangePackage = RecordMeta & {
  caseId: ID;
  matchIds: ID[];
  transferLegIds: ID[];
  reservedQuantities: Quantity[];
  logisticsCost: Money;
};
export type ReuseReceipt = RecordMeta & {
  caseId: ID;
  packageId: ID;
  custodyRefs: Ref[];
  recipientAcceptance: Ref;
  residualIssue: string;
};

export type ReuseDesk = { exchangeId: string; coordinatorRequest: string; suppliedOffer: SurplusOffer; materialEvents: Evidence[]; allocationRules: string[]; authority: "authorized" | "unclear" | "unauthorized"; changedMaterialIds: string[]; audience: string; offers?: SurplusOffer[]; needs?: ReuseNeed[]; transferReceipts?: TransferObservation[] };
export type MaterialProfile = { offerId: string; ownerId: string; supportedFunctions: string[]; verifiedProperties: string[]; unknownProperties: string[]; quantity: Quantity; restrictions: string[]; maintenanceEvidenceIds: string[]; withdrawn: boolean };
export type FunctionalNeed = { needId: string; recipientId: string; essentialBehavior: string[]; preferredForm: string[]; allowedAdaptations: string[]; confirmationIds: string[]; unresolved: string[] };
export type ReuseConcept = { id: string; offerIds: string[]; needId: string; mechanism: string; essentialFitEvidence: string[]; adaptations: string[]; specialistQuestions: string[]; qualificationReceiptIds: string[]; status: "provisional" | "qualified" | "rejected" };
export type ExchangeLeg = { id: string; conceptId: string; offerId: string; recipientId: string; quantity: Quantity; prerequisites: string[]; reservationId: string | null; state: "proposed" | "reserved" | "in-custody" | "accepted" | "withdrawn" | "unknown"; receiptIds: string[] };
export type ResidualMaterial = { offerId: string; offered: Quantity; reserved: Quantity; accepted: Quantity; available: Quantity; rejectedCondition: string[] };
export type TransferObservation = { legId: string; custodyReceiptId: string | null; acceptanceReceiptId: string | null; acceptedQuantity: Quantity | null; rejectedQuantity: Quantity | null; condition: string; titleEvidenceId: string | null; evidence: Evidence[] };
export type ReuseExchangeBook = { revision: number; profiles: MaterialProfile[]; needs: FunctionalNeed[]; concepts: ReuseConcept[]; legs: ExchangeLeg[]; residuals: ResidualMaterial[]; consumedEventIds: string[] };
export type ReuseExchangeReview = { exchange: ReuseExchangeBook; qualificationTasks: string[]; proposedLegs: ExchangeLeg[]; invalidatedLegIds: string[]; residualOffers: ResidualMaterial[]; unmetNeeds: string[]; outcomeClaims: string[] };
