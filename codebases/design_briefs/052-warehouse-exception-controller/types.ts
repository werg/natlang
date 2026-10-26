export type ID = string;
export type Decimal = string;
export type Ratio = { numerator: string; denominator: string };
export type Quantity = { value: Decimal; unit: string };
export type Money = { minorUnits: string; currency: string; scale: number };
export type DateTime = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type WarehouseException = RecordMeta & {
  warehouseId: ID;
  itemRefs: ID[];
  reportedDiscrepancy: string;
  operatorId: ID;
  inventorySnapshotId: ID;
};
export type StockObservation = RecordMeta & {
  caseId: ID;
  locationId: ID;
  observedDescription: string;
  serial: string;
  quantity: Quantity;
  condition: string;
};
export type StockIdentityFinding = RecordMeta & {
  caseId: ID;
  observationId: ID;
  candidateItemIds: ID[];
  verdict: string;
  missingEvidence: string;
};
export type WarehouseFinding = RecordMeta & {
  caseId: ID;
  itemId: ID;
  kind: 'misplaced'|'mislabeled'|'damaged'|'missing'|'pending';
  affectedQuantity: Quantity;
};
export type WarehouseWorkOrder = RecordMeta & {
  caseId: ID;
  findingId: ID;
  task: string;
  targetLocationId: ID;
  verificationRequired: string;
};
export type InventoryResolution = RecordMeta & {
  caseId: ID;
  workOrderIds: ID[];
  movementIds: ID[];
  adjustmentIds: ID[];
  residualDiscrepancy: string;
};

export type WarehouseInbox = { exceptionId: string; complaint: string; suppliedObservation: StockObservation; eventEvidence: Evidence[]; procedures: string[]; authority: "authorized" | "unclear" | "unauthorized"; changedEventIds: string[]; operatorView: string; deviceEntries?: DeviceObservation[]; verificationReceipts?: ScanVerification[] };
export type DeviceObservation = { deviceId: string; sequence: number; observationId: string; payloadHash: string; operatorId: string; locationId: string; itemId: string | null; lotId: string | null; serial: string | null; quantity: Quantity; condition: string; signatureReceiptId: string | null };
export type ScanVerification = { receiptId: string; deviceId: string; sequence: number; payloadHash: string; verifierIdentity: string; signatureValid: boolean };
export type DeviceCursor = { deviceId: string; acknowledgedThrough: number; acceptedObservationIds: string[]; missingRanges: string[]; quarantinedIds: string[] };
export type StockIdentityDecision = { observationId: string; itemId: string | null; lotId: string | null; serial: string | null; alternatives: string[]; missingScan: string | null; evidenceIds: string[] };
export type StockBin = { itemId: string; lotId: string; locationId: string; physicallyPresent: Quantity; available: Quantity; ledgerVersion: string; custodyIds: string[] };
export type InventoryWork = { id: string; observationIds: string[]; kind: "search" | "recount" | "relabel" | "inspect" | "transfer" | "adjust"; prerequisites: string[]; operator: string; independentVerifier: string | null; expectedLedgerVersion: string; quantityDelta: Quantity | null; availabilityOnly: boolean; blockers: string[] };
export type StockAdjustment = { operationId: string; workId: string; itemId: string; lotId: string; locationId: string; expectedLedgerVersion: string; movementReceiptId: string | null; adjustmentReceiptId: string | null; state: "proposed" | "applied" | "unknown" | "reversed" };
export type WarehouseJournal = { revision: number; cursors: DeviceCursor[]; observations: DeviceObservation[]; identities: StockIdentityDecision[]; bins: StockBin[]; work: InventoryWork[]; adjustments: StockAdjustment[]; consumedEventIds: string[] };
export type WarehouseReview = { journal: WarehouseJournal; pendingWork: InventoryWork[]; syncRequests: string[]; uncertainIdentities: string[]; availabilityHolds: string[]; conservationFindings: string[] };
export type InventoryReceipt = { operationId: string; ledgerVersion: string; binsAfter: StockBin[]; movementReceiptId: string | null; adjustmentReceiptId: string | null; observedBy: string; sourceEvidence: Evidence[] };
export type InventoryReceiptReview = { journal: WarehouseJournal; reconciledOperationIds: string[]; residualDifferences: string[] };

export type ScanIngress = { cursors: DeviceCursor[]; admitted: DeviceObservation[]; quarantined: DeviceObservation[]; syncRequests: string[] };
