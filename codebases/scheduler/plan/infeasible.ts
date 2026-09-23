import type { Task, Slot, ScheduleSnapshot, Candidate, Alternatives, ScheduleResult } from "../types.js";

export default function infeasible(snapshot: ScheduleSnapshot, alternatives: Alternatives): ScheduleResult {
return { status: 'infeasible', revision: snapshot.revision,
  plan: snapshot.plan, detail: alternatives.detail };
}
