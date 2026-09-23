import type { Event, File, Recipe, Job, Outcome, Session, ViewBlock, TerminalView } from "../types.js";
import { host } from "natlang:runtime";

export default function launch(_acc: Session, item: Event, recipe: string): Session {
const acc = _acc;
if (!item.id || item.id === acc.active_request || acc.history.some(h => h.request_id === item.id))
  return { ...acc, messages: [...acc.messages, 'Duplicate or empty request ID.'] };
if (acc.status === 'running' || acc.status === 'cancel-requested')
  return { ...acc, messages: [...acc.messages, `Request ${item.id} was not accepted while ${acc.active_request} is active; resubmit it after completion.`] };
if (recipe === 'unsupported' || !host.terminal.catalog().some(r => r.id === recipe))
  return { ...acc, revision: acc.revision + 1, status: 'unsupported',
    messages: [...acc.messages, `No supported recipe for: ${item.text}`] };
const job = host.terminal.start(item.id, recipe);
return { ...acc, revision: acc.revision + 1, active_request: item.id, active_job: job.id,
  status: job.status, messages: [...acc.messages, `Started ${recipe} for ${item.id}.`] };
}
