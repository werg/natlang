/*---
engine: typescript-host
args:
  queries: Text[]
returns: SearchResult
---*/
return host.evidence.search(args.queries);
