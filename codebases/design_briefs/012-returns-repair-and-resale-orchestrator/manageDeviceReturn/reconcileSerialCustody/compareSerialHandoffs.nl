---
description: Compare observed serials, custodians and condition across handoff receipts.
args:{scope:GoalContract;serialLedger:EvidencePage[];receipts:EvidencePage[]}
returns:HandoffComparison[]
---
Match each receipt to the original item serial and expected custodian transition. Record condition difference between adjacent observations without attributing fault absent evidence. Missing handoff creates a custody gap; a different serial blocks the chain. Never treat a generated label as proof of collection or delivery.
