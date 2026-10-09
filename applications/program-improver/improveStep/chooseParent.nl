---
args:
  members: ParentChoice[]
returns: string
---
Choose which population member the next experiment starts from. Each member has an id, its validation quality, wonCases (the cases where it has the highest score in the population) and timesParent (how often it has been a parent).

1. Keep the members whose wonCases list is not empty. When none remains, keep all members.
2. Among those, prefer the member with the smallest timesParent.
3. When several members tie, prefer the one with the higher quality, then the smaller id.
4. Return the id of that member.
