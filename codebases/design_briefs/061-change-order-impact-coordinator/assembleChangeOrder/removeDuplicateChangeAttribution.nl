---
args: { items: ChangeCostItem[], events: WorkEvent[] }
returns: CostDeduplication
---
Compare work-package, quantity, rate, overlap group, and contemporaneous event evidence. Accept one supported attribution for the same work and list duplicate cost IDs; do not erase actual work solely because the authorization is disputed.
