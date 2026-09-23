import type { Fighter, Opponent, CombatObservation, CombatPlan, CombatReceipt } from "../types.js";
import { host } from "natlang:runtime";

export default function observe(actor: string): CombatObservation {
return host.combat.observe(actor);
}
