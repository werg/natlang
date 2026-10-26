---
args: { instruction: SiteInstruction, confirmation: InstructionConfirmation, events: WorkEvent[], evidence: Evidence[] }
returns: ConfirmationScopeAudit
---
1. Bind confirmsExactInstruction only when confirmation.instructionRef identifies this instruction version and its scope expressly covers the requestedChange and authorityScope; a project-wide relationship or a signer name alone is insufficient.
2. Bind signerAuthorized only from a current supplied authorityRef whose scope permits this instruction class and site; null or unverified references fail closed.
3. Bind authorityEvidenceCurrent only when the evidence is current for the instruction issue date and is not contradicted or superseded by supplied evidence. Retain the exact contradicting reference as missing/contested follow-up.
4. Bind preApprovalEventIds to events for the affected work package observed before a valid confirmation was received. Those events remain actual work, never retroactive approval.
5. Return all four booleans, the exact pre-approval event IDs, confirmation.disputed, and each missing authority, scope, instruction-version, or evidence condition.
