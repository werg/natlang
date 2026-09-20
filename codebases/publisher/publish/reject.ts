/*---
engine: typescript-host
args:
  target: Text
  checked: PublishCheck
returns: PublishReport
---*/
return { status: 'rejected', target: args.target, revision: '',
  markdown_sha256: '', html_sha256: '', detail: args.checked.detail };
