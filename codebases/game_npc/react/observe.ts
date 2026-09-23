import type { NpcEvent, Memory, Commitment, NpcObservation, NpcPlan, NpcResult } from "../types.js";
import { host } from "natlang:runtime";

export default function observe(actor: string, event: NpcEvent): NpcObservation {
return host.npc.observe(actor, event);
}
