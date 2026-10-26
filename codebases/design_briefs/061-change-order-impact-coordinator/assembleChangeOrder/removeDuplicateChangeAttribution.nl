---
args: { items: ChangeCostItem[], events: WorkEvent[] }
returns: CostDeduplication
---
Bind matches = items.map(item => matchCostToFieldEvents(item, events)). Group only matches with the same work package, quantity-rate basis, overlap group, and event evidence. Keep one supported attribution per evidenced work item, classify later duplicate claims as duplicates, and leave conflicting or unsubstantiated claims unresolved. Disputed authorization does not erase actual work evidence.
