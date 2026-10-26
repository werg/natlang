---
args: { request: ObligationReviewRequest; duties: DutyFacetLedger; relations: ClauseRelationGraph; triggers: TriggerLedger; calendar: ObligationGuardrail; identity: ObligationGuardrail }
returns: ObligationTaskMap
---
Create tasks only for active duties and name the contract party supported by current identity evidence. Carry prerequisite obligations from relation edges. Compute a deadline only when the incorporated calendar resolution supports the exact trigger date and counting rule; otherwise make the task unscheduled with a calendar blocker. A draft notice is an instruction candidate, never dispatch proof. Return task proposals and withheld obligations with reason.