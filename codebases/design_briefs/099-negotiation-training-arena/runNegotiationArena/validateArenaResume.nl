---
description: Verify scenario lineage and caller history against the durable negotiation checkpoint.
args:{checkpoint:ArenaCheckpoint;scenario:ArenaScenario;turns:ArenaTurn[]}
returns:{compatible:boolean;turns:ArenaTurn[];questions:string[]}
---
The checkpoint is the durable source of episode turns. Verify both scenarioId and revision match. Verify the caller-supplied turns equal an exact prefix of checkpoint.turns; a shorter matching prefix may be stale but can safely resume from the full checkpoint ledger, while a differing turn at any shared position is a lineage conflict. Return checkpoint.turns as the only continuation history. Mark compatible false for a scenario mismatch or divergent supplied history and state the exact conflict as a question; never merge conflicting turns or infer which version is authoritative.
