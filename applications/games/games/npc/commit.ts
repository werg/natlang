/**
 * The village's commit: a pure, bounded application of an event's effects to its state, and the check of what the
 * stages produced. An event is acted on once: the effects name the version they were decided on, the event id must
 * be the next one and not yet applied. Returns the next state and the event, or the state unchanged and the problem.
 */
import observe from './observe.js';
import rules from './validate/rules.js';
import type { Committed, NpcEffects, NpcState } from '../../types.js';

export default function commit(state: NpcState, effects: NpcEffects): Committed {
  const refuse = (problem: string): Committed => ({ ok: false, state, events: [], problem });
  const { actor, event, event_id: id, plan, notes } = effects;
  if (effects.basis !== state.version) return refuse(`the effects are for version ${effects.basis}; the village is at version ${state.version}`);
  if (state.applied.includes(id)) return refuse(`${id} was acted on already`);
  if (id !== `event-${state.nextEvent}`) return refuse(`the next event is event-${state.nextEvent}, not ${id}`);
  let observation;
  try { observation = observe(state, actor, event); } catch (error) { return refuse(String((error as Error).message)); }
  const verdict = rules(observation, plan);
  if (!verdict.ok) return refuse(`${actor}: ${verdict.reason}`);
  const known = new Set(observation.memory.map(row => row.id));
  for (const note of notes)
    if (typeof note?.text !== 'string' || !note.text || !Array.isArray(note.about) || !note.about.every(about => known.has(about)))
      return refuse(`${actor}: a note has text and cites entries of its memory`);
  const next: NpcState = structuredClone(state);
  const npc = next.actors.find(row => row.id === actor)!;
  npc.memory.push({ id, from: event.from, text: event.text });
  for (const [k, note] of notes.entries()) npc.memory.push({ id: `${id}.note-${k + 1}`, from: actor, text: note.text, about: [...note.about] });
  if (plan.action === 'give') npc.inventory[plan.item!] = npc.inventory[plan.item!]! - 1;
  if (plan.action === 'promise') npc.commitments.push({ to: plan.target!, detail: plan.detail!, evidence_id: id });
  next.applied.push(id);
  next.nextEvent += 1;
  next.version += 1;
  const events = [{ operation: 'npc.act', actor, said: plan.say, action: plan.action, evidence_id: id,
    inventory: structuredClone(npc.inventory), commitments: structuredClone(npc.commitments) }];
  return { ok: true, state: next, events, problem: '' };
}
