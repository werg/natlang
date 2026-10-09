/** The crisp significance rule: the level decides. It reads no message. */
import type { LogEvent, Observation, Significance } from '../../types.js';

const URGENT = ['fatal', 'critical', 'alert', 'emergency', 'panic'];
const WATCH = ['error', 'err', 'warn', 'warning'];

export default function exact(item: LogEvent, _observation: Observation): Significance {
  const level = item.level.toLowerCase();
  return URGENT.includes(level) ? 'urgent' : WATCH.includes(level) ? 'watch' : 'ignore';
}
