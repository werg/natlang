import type { Target, Obligation, Signature, Witness, Context, CallClaim, Diagnostic, CheckReport, Candidate, FitReport, Assessment } from "../types.js";
import { effects as fx } from "natlang:runtime";

export default function check_calls(context: Context, claims: CallClaim[]): Diagnostic[] {
return fx.types.calls(context, claims);
}
