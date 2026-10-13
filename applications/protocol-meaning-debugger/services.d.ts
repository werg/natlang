import type {TrialData, ExecutablePlan, NumericReceipt, ProtocolRecord, Witness, ExactCheck, RepairEdit} from './types.js';
/** Exact arithmetic and text mechanics; these methods do not interpret protocol policy. */
export type MeasurementLab = {
  evaluate(data: TrialData, plan: ExecutablePlan): NumericReceipt;
  checkWitnesses(documents: ProtocolRecord[], witnesses: Witness[]): ExactCheck[];
  checkRepairs(documents: ProtocolRecord[], edits: RepairEdit[], allowedPaths: string[]): ExactCheck[];
  applyDefinitions(documents: ProtocolRecord[], edits: RepairEdit[], allowedPaths: string[]): ProtocolRecord[];
};
declare module 'natlang:services' { export const measurementLab: MeasurementLab; }
