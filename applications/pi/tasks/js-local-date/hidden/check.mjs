import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const { addDays, formatDate, parseDate } = await import(pathToFileURL(`${process.cwd()}/src/dates.js`).href);
const d = parseDate('2024-01-01');
assert.equal(formatDate(d), 'Jan 1, 2024');
assert.deepEqual([d.getHours(), d.getMinutes(), d.getDate()], [0, 0, 1]);
const cases = [
  ['2024-11-02', 1, 'Nov 3, 2024'], ['2024-11-03', 1, 'Nov 4, 2024'], ['2024-11-01', 7, 'Nov 8, 2024'],
  ['2024-03-10', 1, 'Mar 11, 2024'], ['2024-03-15', -7, 'Mar 8, 2024'], ['2024-04-07', -1, 'Apr 6, 2024'],
  ['2024-09-29', 1, 'Sep 30, 2024'], ['2024-12-31', 1, 'Jan 1, 2025'], ['2024-02-28', 1, 'Feb 29, 2024'],
  ['2023-03-01', -1, 'Feb 28, 2023'], ['2024-01-01', 366, 'Jan 1, 2025'], ['2024-06-15', 0, 'Jun 15, 2024'],
];
for (const [start, days, expected] of cases) {
  const out = addDays(parseDate(start), days);
  assert.equal(formatDate(out), expected, `${start} + ${days} in ${process.env.TZ}`);
  assert.equal(out.getHours(), 0, `${start} + ${days} midnight in ${process.env.TZ}`);
}
assert.throws(() => parseDate('not a date'), RangeError);
