import type { LogEvent, File, Observation, Evidence, Judgement, Alert, IncidentState, ViewBlock, TerminalView } from "../types.js";
import { host } from "natlang:runtime";

export default function observe(item: LogEvent): Observation {
return host.logs.observe(item);
}
