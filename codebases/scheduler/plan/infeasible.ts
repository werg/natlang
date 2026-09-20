/*---
engine: typescript-host
args:
  snapshot: ScheduleSnapshot
  alternatives: Alternatives
returns: ScheduleResult
---*/
return { status: 'infeasible', revision: args.snapshot.revision,
  plan: args.snapshot.plan, detail: args.alternatives.detail };
