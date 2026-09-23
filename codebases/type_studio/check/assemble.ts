export default function assemble(claims: CallClaim[], diagnostics: Diagnostic[]): CheckReport {
return { claims: claims, diagnostics: diagnostics };
}
