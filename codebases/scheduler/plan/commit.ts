/*---
engine: typescript-host
args:
  chosen: Candidate
  revision: number
returns: ScheduleResult
---*/
return host.scheduler.commit(args.chosen, args.revision);
