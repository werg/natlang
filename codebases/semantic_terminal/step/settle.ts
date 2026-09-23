import type { Event, File, Recipe, Job, Outcome, Session, ViewBlock, TerminalView } from "../types.js";
import { host } from "natlang:runtime";

export default function settle(acc: Session, item: Event, message: string): Session {
const a = acc, e = item;
if (!host.terminal.confirm(e))
  return { ...a, messages: [...a.messages, `Unverified completion ignored: ${e.job_id}`] };
const history = [...a.history, { request_id: e.request_id, job_id: e.job_id,
  status: e.status, detail: e.detail }];
if (e.request_id !== a.active_request || e.job_id !== a.active_job)
  return { ...a, history, messages: [...a.messages, `Stale result ${e.job_id}: ${message}`] };
return { ...a, history, revision: a.revision + 1, active_request: '', active_job: '',
  status: e.status, messages: [...a.messages, message] };
}
