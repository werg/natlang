---
args: { instruction: SiteInstruction, confirmations: InstructionConfirmation[], events: WorkEvent[], evidence: Evidence[] }
returns: AuthorityFinding
---
For each written confirmation bind audit = auditConfirmationScope(instruction, confirmation, events, evidence). Count authority only when audit.signerAuthorized, audit.scopeCoversInstruction, audit.confirmsExactInstruction, and audit.authorityEvidenceCurrent are all true. Preserve every disputed confirmation and audit.preApprovalEventIds as factual work without permission. Return confirmed only when a current exact-scope confirmation exists and no competing dispute remains; return every audit missingEvidence and disputed instruction reference.
