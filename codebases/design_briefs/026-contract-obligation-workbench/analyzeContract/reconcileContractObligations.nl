---
args: { prior: ContractObligationCheckpoint|null, request: ObligationReviewRequest, dossier: ContractEvidenceDossier, duties: DutyFacetLedger, relations: ClauseRelationGraph|null, triggers: TriggerLedger|null, tasks: ObligationTaskMap|null, statuses: ObligationDischargeLedger|null }
returns: ContractObligationCheckpoint
---
Preserve the exact contract edition and source locators for each duty facet. Keep clause relations directional: superseded clauses do not erase completed historical acts. A newly established trigger activates only its matching duty; an ineffective notice does not start the contractual clock. Reopen performance tasks only for changed duties or prerequisites, and keep partially performed, disputed, waived and discharged obligations distinct. Duplicate source and event receipt replay does not create another state revision.
