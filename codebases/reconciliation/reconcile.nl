import { assess } from "./reconcile/assess";
import { join_events } from "./reconcile/join_events";
import { summarize } from "./reconcile/summarize";
---
args:
  customers: Customer[]
  events: Event[]
returns: Report
types:
  Customer: '{ id: string, tier: string }'
  Event: '{ id: string, customer: string, message: string, cents: number }'
  Joined: '{ id: string, customer: string, message: string, cents: number, matched: boolean, tier:
    string }'
  Join: '{ rows: Joined[], duplicates: number }'
  Label: '"urgent" | "normal"'
  Report: '{ totals: Record<string, number>, urgent: string[], unmatched: string[], duplicates: number
    }'
description: Reconcile two collections with deduplication, semantic triage and exact
  aggregation.
---
function reconcile(customers, events) -> Report
  joined = join_events(customers, events)   # deduplicate IDs before joining; first event wins
  labels = for each row in joined.rows: assess(row)
  return summarize(joined, labels)         # preserve row alignment; exact cents and counts
