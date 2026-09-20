/*---
engine: typescript-host
args:
  query: Text
  revision: Text
returns: SearchResult
---*/
return host.repository.search(args.query, args.revision);
