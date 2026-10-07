// A chatty test runner: prints every case it checks.
import { cents, format, split, sum } from './lib/money.js';
let failed = 0, passed = 0;
const check = (name, actual, expected) => {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (ok) { passed++; console.log(`  ok    ${name}`); }
  else { failed++; console.log(`  FAIL  ${name}\n        expected ${JSON.stringify(expected)}\n        actual   ${JSON.stringify(actual)}`); }
};
console.log('money: cents');
for (let d = 0; d < 150; d++) check(`cents(${d / 4})`, cents(d / 4), Math.round(d * 25));
console.log('money: format');
for (let c = -60; c < 120; c++) check(`format(${c * 7})`, format(c * 7), `${c * 7 < 0 ? '-' : ''}$${Math.floor(Math.abs(c * 7) / 100)}.${String(Math.abs(c * 7) % 100).padStart(2, '0')}`);
console.log('money: split');
for (let n = 1; n <= 6; n++) check(`split(600, ${n}) adds up`, sum(split(600, n)), 600);
check('split(100, 3)', split(100, 3), [34, 33, 33]);
for (let n = 1; n <= 40; n++) check(`split(${n * 12}, 4)`, split(n * 12, 4), [n * 3, n * 3, n * 3, n * 3]);
console.log('money: sum');
for (let k = 0; k < 60; k++) check(`sum of ${k} ones`, sum(Array(k).fill(1)), k);
console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
