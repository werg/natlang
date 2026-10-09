/**
 * Summarize one update against the block it edits. Pluggable hot path (it runs once per update): with
 * `settings.changes` "crisp" the exact comparison in `changes/exact.ts`, with "nl" (or the older "natlang")
 * `changes/summarize.nl`, and "shadow" runs both.
 */
import { pluggable } from '@natlang/node';
import exact from './changes/exact.ts';
import summarize from './changes/summarize.nl';
import type { Change, WikiBlock, WikiSettings, WikiUpdate } from '../../types.js';

export default async function changes(block: WikiBlock, update: WikiUpdate, settings: WikiSettings): Promise<Change> {
  return pluggable({ crisp: () => exact(block, update), nl: () => summarize(block, update) }, settings.changes,
    { name: 'wiki.changes' })();
}
