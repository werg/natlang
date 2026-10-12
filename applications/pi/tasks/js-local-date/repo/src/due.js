import { addDays, formatDate, parseDate } from './dates.js';

/** Human text for a task created on `created` ('YYYY-MM-DD') that is due `days` later. */
export function dueText(created, days) {
  return `due ${formatDate(addDays(parseDate(created), days))}`;
}
