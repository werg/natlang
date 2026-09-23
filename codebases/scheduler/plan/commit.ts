import type { Task, Slot, ScheduleSnapshot, Candidate, Alternatives, ScheduleResult } from "../types.js";
import { host } from "natlang:runtime";

export default function commit(chosen: Candidate, revision: number): ScheduleResult {
return host.scheduler.commit(chosen, revision);
}
