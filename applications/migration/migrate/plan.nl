---
description: Migration planning. From the classified sites, decide which to edit and in what order, which to leave and why, and which risks the checks must watch.
args:
  intent: Intent
  classified: Classified[]
returns: CheckedPlan
---
Plan the migration of intent over classified (each a site with its usage). Compute the lists exactly in eval, then
write the risks.

1. edits: the ids of the sites whose usage.action is "edit", in this order: declaration, then export, then import,
   then type-position, property-access, call, test; within a pattern by path, then from.
2. leave: { site, reason } for every other site, with the usage's reason.
3. risks: one sentence for each of these that applies. A declaration is edited but a call site was left. A string
   literal that names the old thing was left and may be looked up at run time. A site in a file no check exercises
   (the checks are named in intent.invariants). No site is to be edited at all.

Return { edits, leave, risks }.
