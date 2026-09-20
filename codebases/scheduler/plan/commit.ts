/*---
engine: typescript-host
args:
  chosen: Candidate
  revision: Num
returns: ScheduleResult
---*/
return host.scheduler.commit(args.chosen, args.revision);
