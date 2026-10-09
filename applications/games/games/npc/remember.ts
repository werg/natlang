/**
 * The NPC's memory update for an event: the notes it adds beside the event's own entry. Pluggable hot path
 * ("remember"): `remember/notes.nl` by default; with settings 'crisp' the NPC keeps the event entry alone.
 */
import notes from './remember/notes.nl';
import type { Note, NpcObservation, Settings } from '../../types.js';

export default async function remember(observation: NpcObservation, settings: Settings): Promise<Note[]> {
  if (settings.remember === 'crisp') return [];
  return notes(observation);
}
