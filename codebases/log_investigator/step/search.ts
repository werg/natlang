import type { LogEvent, File, Observation, Evidence, Judgement, Alert, IncidentState, ViewBlock, TerminalView } from "../types.js";
import { host } from "natlang:runtime";

export default function search(observation: Observation): Evidence[] {
return host.logs.query(observation);
}
