import type { Task, Slot, ScheduleSnapshot, Candidate, Alternatives, ScheduleResult } from "../types.js";
import { host } from "natlang:runtime";

export default function enumerate(): Alternatives {
return host.scheduler.alternatives();
}
