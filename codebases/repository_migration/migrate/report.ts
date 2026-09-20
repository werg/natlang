/*---
engine: typescript-host
args:
  revision: Text
  checks: Validation
returns: MigrationReport
---*/
return host.repository.report(args.revision, args.checks);
