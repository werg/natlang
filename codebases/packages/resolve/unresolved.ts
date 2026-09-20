/*---
engine: typescript-host
args:
  request: PackageRequest
returns: Resolution
---*/
return { status: 'unresolved', lock: { id: '', root: args.request.name,
  engine: args.request.engine, packages: [] }, alternatives: 0,
  detail: 'No compatible lock satisfies the declared constraints and engine.' };
