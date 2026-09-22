/*---
engine: typescript-host
args:
  revision: string
  checked: CheckReport
returns: RunReport
---*/
return { status: 'invalid-source', run_id: '', revision: args.revision,
  source_revision: '',
  value_text: '', trace_events: 0, detail: args.checked.detail };
