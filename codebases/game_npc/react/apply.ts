import type { NpcEvent, Memory, Commitment, NpcObservation, NpcPlan, NpcResult } from "../types.js";
import { host } from "natlang:runtime";

export default function apply(observation: NpcObservation, plan: NpcPlan): NpcResult {
return host.npc.apply(observation, plan);
}
