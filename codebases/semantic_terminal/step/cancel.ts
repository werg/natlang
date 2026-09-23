import type { Event, File, Recipe, Job, Outcome, Session, ViewBlock, TerminalView } from "../types.js";
import { host } from "natlang:runtime";

export default function cancel(acc: Session, item: Event): Session {
const a = acc, e = item;
if (!a.active_job || e.request_id !== a.active_request)
  return { ...a, messages: [...a.messages, `No active job for ${e.request_id}.`] };
host.terminal.cancel(a.active_job);
return { ...a, revision: a.revision + 1, status: 'cancel-requested',
  messages: [...a.messages, `Cancellation requested for ${a.active_job}; awaiting actual outcome.`] };
}
