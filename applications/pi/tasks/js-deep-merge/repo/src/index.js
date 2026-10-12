import { deepMerge } from './merge.js';

export const defaults = { theme: { color: 'blue', size: 12 }, plugins: ['core'] };

/** User settings layered over the defaults. */
export function settings(user) {
  return deepMerge(defaults, user);
}
