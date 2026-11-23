import { pluggable, type PluggableSetting } from '@natlang/node';
import eligibleNl from './eligible.nl';
import type { Request } from './types.js';

/** The desk's 30-day return window, read from the order record alone. */
export const withinWindow = (request: Request): boolean => request.order.days_since_delivery <= 30;

/**
 * Whether a request may be settled at all: the crisp rule, or the judgment that also reads what the customer reports.
 * `shadow` runs both, serves the crisp answer and records each disagreement in the trace (`pluggable_shadow`).
 */
export function eligibility(setting: PluggableSetting) {
  return pluggable<[Request], boolean>({ crisp: withinWindow, nl: request => eligibleNl(request) }, setting,
    { name: 'refunds.eligible', default: 'crisp', serve: 'crisp' });
}
