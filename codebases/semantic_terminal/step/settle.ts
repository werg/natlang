/*---
engine: typescript-host
args:
  acc: Session
  item: Event
  message: Text
returns: Session
---*/
const a = args.acc, e = args.item;
if (!host.terminal.confirm(e))
  return { ...a, messages: [...a.messages, `Unverified completion ignored: ${e.job_id}`] };
const history = [...a.history, { request_id: e.request_id, job_id: e.job_id,
  status: e.status, detail: e.detail }];
if (e.request_id !== a.active_request || e.job_id !== a.active_job)
  return { ...a, history, messages: [...a.messages, `Stale result ${e.job_id}: ${args.message}`] };
return { ...a, history, revision: a.revision + 1, active_request: '', active_job: '',
  status: e.status, messages: [...a.messages, args.message] };
