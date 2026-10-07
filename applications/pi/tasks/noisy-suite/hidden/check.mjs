import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const { split, sum } = await import(pathToFileURL(`${process.cwd()}/lib/money.js`).href);
for (const [amount, n] of [[100, 3], [101, 4], [7, 5], [0, 3], [999, 7], [5, 1]]) {
  const shares = split(amount, n);
  assert.equal(shares.length, n); assert.equal(sum(shares), amount);
  assert.ok(Math.max(...shares) - Math.min(...shares) <= 1, `${amount}/${n}`);
  assert.deepEqual(shares, [...shares].sort((a, b) => b - a));
}
