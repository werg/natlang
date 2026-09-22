/*---
engine: typescript-host
args:
  queries: string[]
returns: SearchResult
---*/
return host.evidence.search(args.queries);
