/*---
engine: typescript-host
args:
  query: string
  revision: string
returns: SearchResult
---*/
return host.repository.search(args.query, args.revision);
