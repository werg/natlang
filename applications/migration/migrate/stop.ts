/**
 * Whether the repair loop stops at a state: the pluggable `settled` point. `repository.implementation('settled')` selects
 * `settledCrisp.ts` (crisp), `settled.nl` (nl) or both (shadow, serving the crisp answer). The migration instructions call
 * `stop` and never branch on the setting.
 */
import { pluggable } from '@natlang/node';
import { repository } from 'natlang:services';
import settled from './settled.nl';
import settledCrisp from './settledCrisp.ts';
import type { RepairState } from '../types.js';

export default async function stop(state: RepairState): Promise<boolean> {
  return pluggable({ crisp: () => settledCrisp(state), nl: () => settled(state) }, await repository.implementation('settled'),
    { name: 'migration.settled', serve: 'crisp' })();
}
