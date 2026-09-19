---
args:
  customers: Customer[]
  events: Event[]
returns: Report
types:
  Customer: '{ id: Text, tier: Text }'
  Event: '{ id: Text, customer: Text, message: Text, cents: Num }'
  Joined: '{ id: Text, customer: Text, message: Text, cents: Num, matched: Bool, tier:
    Text }'
  Join: '{ rows: Joined[], duplicates: Num }'
  Label: '"urgent" | "normal"'
  Report: '{ totals: Dict<Num>, urgent: Text[], unmatched: Text[], duplicates: Num
    }'
description: Reconcile two collections with deduplication, semantic triage and exact
  aggregation.
---
function reconcile(customers, events) -> Report
  joined = join_events(customers, events)   # deduplicate IDs before joining; first event wins
  labels = for each row in joined.rows: assess(row)
  return summarize(joined, labels)         # preserve row alignment; exact cents and counts
