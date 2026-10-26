---
args: { duties: DutyFacetLedger; relations: ClauseRelationGraph; evidence: EvidencePage[]; notice: ObligationGuardrail }
returns: TriggerLedger
---
For each obligation, compare each observed event with the trigger's named actors, event, date, condition and required notice effect. Mark active only when every required fact is evidenced; mark inactive only when an evidenced exception defeats the trigger; otherwise preserve unknown or disputed. Do not infer discharge from sending a notice. Return event IDs, evidence refs, status and the precise missing trigger fact for every unresolved row.