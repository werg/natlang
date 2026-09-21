/*---
engine: typescript-host
args:
  head: Text
  candidate: Text
returns: CommitResult
---*/
return await host.research.activate(args.head, args.candidate);
