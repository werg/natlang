/*---
engine: typescript-host
args:
  lock: Lock
  target: string
returns: InstallReport
---*/
try { return await host.packages.install(args.lock, args.target); }
catch (error) { return { status: 'failed', target: args.target,
  revision: '', detail: String(error) }; }
