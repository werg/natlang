export default function submit(actor: string, round: number, plan: CombatPlan): CombatReceipt {
return host.combat.submit(actor, round, plan);
}
