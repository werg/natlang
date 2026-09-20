/*---
engine: typescript-host
args:
  acc: Session
  item: Event
  recipe: Text
returns: Session
---*/
const acc = args.acc, item = args.item;
if (!item.id || item.id === acc.active_request || acc.history.some(h => h.request_id === item.id))
  return { ...acc, messages: [...acc.messages, 'Duplicate or empty request ID.'] };
if (acc.status === 'running' || acc.status === 'cancel-requested')
  return { ...acc, messages: [...acc.messages, `Request ${item.id} was not accepted while ${acc.active_request} is active; resubmit it after completion.`] };
if (args.recipe === 'unsupported' || !host.terminal.catalog().some(r => r.id === args.recipe))
  return { ...acc, revision: acc.revision + 1, status: 'unsupported',
    messages: [...acc.messages, `No supported recipe for: ${item.text}`] };
const job = host.terminal.start(item.id, args.recipe);
return { ...acc, revision: acc.revision + 1, active_request: item.id, active_job: job.id,
  status: job.status, messages: [...acc.messages, `Started ${args.recipe} for ${item.id}.`] };
