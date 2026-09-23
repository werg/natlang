import type { Target, Obligation, Signature, Witness, Context, CallClaim, Diagnostic, CheckReport, Candidate, FitReport, Assessment } from "../types.js";
import { effects as fx } from "natlang:runtime";

export default function check_candidate(target: Target, context: Context, candidate: Candidate): FitReport {
return fx.types.check(target, context, candidate);
}
