/**
 * How much a log line matters. Pluggable hot path (it runs on every line): with `settings.significance` "crisp" the
 * exact rule in `significance/exact.ts`, with "nl" (or the older "natlang") `significance/judge.nl`, and "shadow" runs both.
 */
import { pluggable } from '@natlang/node';
import exact from './significance/exact.ts';
import judge from './significance/judge.nl';
import type { LogEvent, LogSettings, Observation, Significance } from '../types.js';

export default async function significance(item: LogEvent, observation: Observation, settings: LogSettings): Promise<Significance> {
  return pluggable({ crisp: () => exact(item, observation), nl: () => judge(item, observation) }, settings.significance,
    { name: 'logs.significance', same: (exact, judged) => exact === judged })();
}
