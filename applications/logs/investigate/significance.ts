/**
 * How much a log line matters. Pluggable hot path (it runs on every line): with `settings.significance` "crisp" the
 * exact rule in `significance/exact.ts`, with "natlang" `significance/judge.nl`.
 */
import exact from './significance/exact.ts';
import judge from './significance/judge.nl';
import type { LogEvent, LogSettings, Observation, Significance } from '../types.js';

export default async function significance(item: LogEvent, observation: Observation, settings: LogSettings): Promise<Significance> {
  return settings.significance === 'crisp' ? exact(item, observation) : judge(item, observation);
}
