import type { Fighter, Opponent, CombatObservation, CombatPlan, CombatReceipt } from "../types.js";
import { host } from "natlang:runtime";

export default function submit(actor: string, round: number, plan: CombatPlan): CombatReceipt {
return host.combat.submit(actor, round, plan);
}
