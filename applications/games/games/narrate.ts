/**
 * The story of a turn from its events. Pluggable hot path ("narrate"): `narrate/story.nl` by default; with settings
 * 'crisp' `narrate/plain.ts`, a plain listing; 'shadow' tells the story and records the listing beside it. Narration
 * reads events and never changes state.
 */
import { pluggable } from '@natlang/node';
import story from './narrate/story.nl';
import plain from './narrate/plain.js';
import type { GameEvent, Settings } from '../types.js';

export default async function narrate(kind: string, events: GameEvent[], settings: Settings): Promise<string> {
  // The wording differs by design, so the shadow event carries both texts and agreement only means both said something.
  return pluggable({ crisp: () => plain(kind, events), nl: () => story(kind, events) }, settings.narrate,
    { name: 'narrate', same: (exact, judged) => (exact.length > 0) === (judged.length > 0) })();
}
