import type { Target, Obligation, Signature, Witness, Context, CallClaim, Diagnostic, CheckReport, Candidate, FitReport, Assessment } from "../types.js";
export default function assemble(claims: CallClaim[], diagnostics: Diagnostic[]): CheckReport {
return { claims: claims, diagnostics: diagnostics };
}
