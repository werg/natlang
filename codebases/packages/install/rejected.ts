/*---
engine: typescript-host
args:
  target: string
  checked: InstallCheck
returns: InstallReport
---*/
return { status: 'rejected', target: args.target, revision: '', detail: args.checked.detail };
