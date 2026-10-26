export type ID = string;
export type Decimal = string;
export type Ratio = { numerator: string; denominator: string };
export type Quantity = { value: Decimal; unit: string };
export type Money = { minorUnits: string; currency: string; scale: number };
export type DateTime = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type ShipmentDisruption = RecordMeta & {
  shipmentIds: ID[];
  disruptionDescription: string;
  coordinatorId: ID;
  planningWindow: string;
  priorityPolicyId: ID;
};
export type DisruptionImpact = RecordMeta & {
  caseId: ID;
  routeIds: ID[];
  affectedWindow: string;
  unavailableCapabilities: string[];
  uncertainty: string;
};
export type ShippingCommitment = RecordMeta & {
  caseId: ID;
  shipmentId: ID;
  promisedOutcome: string;
  deadline: DateTime;
  essentialConditions: string[];
};
export type ShippingAlternative = RecordMeta & {
  caseId: ID;
  shipmentIds: ID[];
  routeIds: ID[];
  substitutions: string[];
  changedPromises: string[];
};
export type ShippingAllocation = RecordMeta & {
  caseId: ID;
  alternativeIds: ID[];
  reservedCapacityIds: ID[];
  totalCost: Money;
  unresolvedDemand: string;
};
export type ShippingRecovery = RecordMeta & {
  caseId: ID;
  allocationId: ID;
  communicationIds: ID[];
  carrierReceiptIds: ID[];
  remainingRisk: string;
};

export type DispatchUpdate = { incidentId: string; dispatcherInstruction: string; suppliedImpact: DisruptionImpact; notices: Evidence[]; priorityRules: string[]; authority: "authorized" | "unclear" | "unauthorized"; changedNoticeIds: string[]; audience: string; routeOffers?: RouteOffer[]; orders?: ShippingOrder[]; reservationReceipts?: CarrierReservation[] };
export type ShippingOrder = { id: string; itemClass: string; quantity: Quantity; origin: string; destination: string; handlingRequirements: string[]; appointments: DeliveryPromise[] };
export type DeliveryPromise = { id: string; shipmentId: string; appointmentId: string; deadline: string; authoritySourceId: string; changedWithApprovalId: string | null; status: "supported" | "at-risk" | "retracted" | "fulfilled" };
export type RouteOffer = { id: string; carrierId: string; from: string; to: string; departure: string; arrival: string; capacity: Quantity; eligibleClasses: string[]; handlingCapabilities: string[]; cost: Money; version: string; evidenceIds: string[] };
export type CarrierReservation = { operationId: string; offerId: string; offerVersion: string; shipmentIds: string[]; quantity: Quantity; state: "requested" | "accepted" | "rejected" | "cancel-confirmed" | "unknown"; receiptIds: string[] };
export type LegAlternative = { id: string; shipmentId: string; offerIds: string[]; preservedPromiseIds: string[]; changedPromiseIds: string[]; residualRisk: string[]; reservationOperationIds: string[]; feasible: boolean };
export type RoutingDelta = { affectedOfferIds: string[]; eligibleClasses: string[]; invalidatedReservationIds: string[]; uncertainReopening: string[]; noticeIds: string[] };
export type DeliveryRecoveryBook = { revision: number; legs: LegAlternative[]; reservations: CarrierReservation[]; promises: DeliveryPromise[]; deliveredShipmentIds: string[]; consumedNoticeIds: string[] };
export type ShipmentRepair = { dispatchBook: DeliveryRecoveryBook; revisedLegs: LegAlternative[]; proposedReservations: CarrierReservation[]; customerCorrections: DeliveryPromise[]; recoveryTasks: string[]; unresolvedCapacity: string[] };
