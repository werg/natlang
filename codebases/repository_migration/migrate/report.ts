/*---
engine: typescript-host
args:
  revision: string
  checks: Validation
returns: MigrationReport
---*/
return host.repository.report(args.revision, args.checks);
