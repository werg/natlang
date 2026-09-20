---
description: Judge customer identity and ambiguous field conflicts in a finite import.
args:
  customers: Customer[]
  existing: Existing[]
returns: Decision[]
types:
  Customer: '{ source_key: Text, email: Text, name: Text }'
  Existing: '{ id: Text, email: Text, name: Text }'
  Decision: '{ source_key: Text, action: "new" | "merge" | "review", target_email: Text, reason: Text }'
---
Return one decision per source customer. Use `new` when there is no supported
match, `merge` only for the same normalized email and the same display name,
and `review` when identity or names conflict. A missing stable source ID or
missing email must be reviewed. Similar names alone are not identity evidence.
Consider both `args/existing` and every row in `args/customers`. For a valid
email absent from `args/existing` that appears more than once in this import,
mark one source row `new` and the other same-name rows `merge`; conflicting
names require `review`. A `merge` may refer to the matching `new` row in this
batch. Return one decision for every source row.
For `merge`, target_email is the matching email; for `new`, it is the source
email; for `review`, it is empty. Keep each source_key exactly as provided.
Each output record must have exactly `source_key`, `action`, `target_email`,
and `reason` fields. `action` is one of `new`, `merge`, or `review`; `reason`
briefly explains the decision. Do not use `decision` as a field name.
