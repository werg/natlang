import type { GameEvent, TradeOutcome } from '../../types.js';

const outcomeLine = (o: TradeOutcome) => o.status === 'traded' ? `${o.actor} bought ${o.quantity} ${o.good} from ${o.seller} for ${o.total}.`
  : o.status === 'rejected' ? `${o.actor} could not trade: ${o.reason ?? 'rejected'}.` : `${o.actor} passed.`;

/** The crisp narration: one sentence per outcome, per damaged fighter, per NPC action. */
export default function plain(kind: string, events: GameEvent[]): string {
  const lines: string[] = [];
  for (const event of events) {
    if (event.operation === 'economy.settle') lines.push(...(event.outcomes as TradeOutcome[]).map(outcomeLine));
    else if (event.operation === 'combat.resolve') {
      const damage = Object.entries(event.damage as Record<string, number>);
      lines.push(damage.length ? `Round ${event.round}: ${damage.map(([id, amount]) => `${id} took ${amount}`).join(', ')}.` : `Round ${event.round}: no one was hurt.`);
    } else if (event.operation === 'npc.act') lines.push(`${event.actor} said "${event.said}"${event.action === 'none' ? '.' : ` and chose to ${event.action}.`}`);
  }
  return lines.join(' ') || `Nothing happened in the ${kind}.`;
}
