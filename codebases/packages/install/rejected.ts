/*---
engine: typescript-host
args:
  target: Text
  checked: InstallCheck
returns: InstallReport
---*/
return { status: 'rejected', target: args.target, revision: '', detail: args.checked.detail };
