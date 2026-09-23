---
args:
  customers: Customer[]
  events: Event[]
returns: Report
description: Reconcile two collections with deduplication, semantic triage and
  exact aggregation.
---
function reconcile(customers, events) -> Report
  joined = join_events(customers, events)   # deduplicate IDs before joining; first event wins
  labels = for each row in joined.rows: assess(row)
  return summarize(joined, labels)         # preserve row alignment; exact cents and counts
