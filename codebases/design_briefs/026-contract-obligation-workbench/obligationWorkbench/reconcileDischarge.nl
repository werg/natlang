---
args: { duties: DutyFacetLedger; triggers: TriggerLedger; tasks: ObligationTaskMap; evidence: EvidencePage[] }
returns: ObligationDischargeLedger
---
Match claimed performance evidence to the obligation actor, scope, trigger and required completion criterion. Preserve partial completion where some but not all duties are evidenced. Record a waiver only with authorized waiver evidence and a disputed status whenever competing readings remain material. Do not infer satisfaction from task closure alone. Return status proposals and the linked source refs.