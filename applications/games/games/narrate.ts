/**
 * The story of a turn from its events. Pluggable hot path ("narrate"): `narrate/story.nl` by default; with settings
 * 'crisp' `narrate/plain.ts`, a plain listing. Narration reads events and never changes state.
 */
import story from './narrate/story.nl';
import plain from './narrate/plain.js';
import type { GameEvent, Settings } from '../types.js';

export default async function narrate(kind: string, events: GameEvent[], settings: Settings): Promise<string> {
  if (settings.narrate === 'crisp') return plain(kind, events);
  return story(kind, events);
}
