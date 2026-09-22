---
args:
  ready: Task[]
returns: string
---
Choose one of the ready tasks by its description: first active security remediation,
then customer-facing work, then internal housekeeping. Within the same priority,
choose the lexicographically smallest ID. Return exactly that task's ID. Do not
invent tasks or consider tasks outside the ready list.
