---
args:
  facts: MoveFacts
returns: string
---
Choose the kind of the next experiment: `edit` rewrites the selected components of one parent, `compose` merges two population members. facts.members lists the population with changedKeys, the component keys each member changed relative to the baseline; facts.iteration is the experiment number.

1. When facts.members has fewer than two members, return `edit`.
2. When two members have changedKeys lists that share no key, return `compose`: their changes can be merged without conflict.
3. Otherwise return `edit`.
