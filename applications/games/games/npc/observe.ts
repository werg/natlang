import type { NpcEvent, NpcObservation, NpcState } from '../../types.js';

/**
 * What an NPC knows when an event reaches it: its inventory, its memory with the event entered under the id the
 * state would assign next (`event-N`), and its commitments. The information boundary: nothing of other NPCs.
 */
export default function observe(state: NpcState, actor: string, event: NpcEvent): NpcObservation {
  const npc = state.actors.find(row => row.id === actor);
  if (!npc || typeof event?.text !== 'string' || !event.text || typeof event.from !== 'string' || !event.from) throw new Error('invalid NPC event');
  const event_id = `event-${state.nextEvent}`;
  return { actor, event_id, event: { from: event.from, text: event.text }, inventory: structuredClone(npc.inventory),
    memory: [...structuredClone(npc.memory), { id: event_id, from: event.from, text: event.text }], commitments: structuredClone(npc.commitments) };
}
