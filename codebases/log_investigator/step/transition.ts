import type { LogEvent, File, Observation, Evidence, Judgement, Alert, IncidentState, ViewBlock, TerminalView } from "../types.js";
import { host } from "natlang:runtime";

export default async function transition(acc: IncidentState, item: LogEvent, observation: Observation, evidence: Evidence[], judgement: Judgement): Promise<IncidentState> {
const a = acc, o = observation, j = judgement;
if (item.cursor <= a.cursor || o.status === 'duplicate')
  return { ...a, status: 'duplicate' };
const cursor = item.cursor, observed = a.observed + 1;
const unknowns = j.uncertainty ? [...a.unknowns, `${item.id}: ${j.uncertainty}`] : a.unknowns;
if (j.action !== 'escalate') return { ...a, cursor, observed, unknowns,
  status: j.action === 'investigate' ? 'investigating' : 'observing' };
const receipt = await host.logs.alert(item, o, evidence, j);
return { ...a, cursor, observed, unknowns,
  alerts: receipt.status === 'insufficient' || receipt.status === 'duplicate' ? a.alerts : [...a.alerts, receipt],
  status: receipt.status === 'insufficient' ? 'investigating' : receipt.status === 'unknown' ?
    'delivery-unknown' : 'alerted' };
}
