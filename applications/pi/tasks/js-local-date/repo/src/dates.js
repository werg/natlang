const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const DAY_MS = 24 * 60 * 60 * 1000;

/** Parse 'YYYY-MM-DD' as a calendar date. */
export function parseDate(text) {
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) throw new RangeError(`invalid date: ${text}`);
  return date;
}

/** 'Mar 9, 2024' */
export function formatDate(date) {
  return `${MONTHS[date.getMonth()]} ${date.getDate()}, ${date.getFullYear()}`;
}

/** The date `days` calendar days later (negative goes back). */
export function addDays(date, days) {
  return new Date(date.getTime() + days * DAY_MS);
}
