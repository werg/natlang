---
description: Inspect actual actor deltas against original privacy and access evidence.
args:
  source: DialogueSource
  scoped: ScopedConversation
  episodePlan: EpisodePlan
  evolution: PerspectiveEvolution
returns: LedgerFinding[]
---
1. For each actual participant belief/commitment delta locate its episode and the actor's previous view.
   Inspect actual new availability rows, not all documents currently visible to the application.
2. Judge whether its source act was authored by this actor or actually available by hearing/receipt.
   Report a failed finding for an actor-owned atom whose only source is somebody else's private note.
3. For verifiedFactIds on belief atoms inspect actual earlier attachment receipt and actor endorsement.
   A correct physical proposition can still fail knowledge provenance if the actor never received its log.
4. Inspect correction defeats against explicit actor adoption and exact superseded scope. Delayed
   recipient arrival cannot defeat an earlier snapshot; sender-private cancellation cannot retract an audience.
5. Check every actual history time follows original episode chronology and each delta is applied once.
   Exact temporal/ID arithmetic validates the relation; source role interpretation is the NL audit judgment.
6. Return witnessed LedgerFindings with actual original IDs and nullable pass where premises are unknown.
   Do not fabricate a fixed ledger or remove an anomalous actual atom from the report during this audit.
