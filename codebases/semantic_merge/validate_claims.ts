/*---
description: Check that a semantic draft accounts for each normalized update once.
args:
  updates: Update[]
  applied: Text[]
  alternatives: Alternative[]
returns: Bool
---*/
const ids = args.updates.map(u => u.id);
const claimed = [...args.applied, ...args.alternatives.flatMap(a => a.update_ids)];
return claimed.length === ids.length && new Set(claimed).size === ids.length &&
       claimed.every(id => ids.includes(id)) &&
       args.alternatives.every(a => a.update_ids.length > 0 && a.reason.trim().length > 0);
