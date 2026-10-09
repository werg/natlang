import { registerSourceContract } from '../registry.js';
import { isReviewedOklahomaAnnualEventVariant, OKLAHOMA_EVENT_SOURCE_ID } from './oklahoma-event-contract.js';

export * from './oklahoma-event-contract.js';

export function registerMusique(): void {
  registerSourceContract({ dataset: 'musique', sourceId: OKLAHOMA_EVENT_SOURCE_ID,
    isReviewedVariant: isReviewedOklahomaAnnualEventVariant });
}
