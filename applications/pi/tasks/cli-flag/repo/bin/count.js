#!/usr/bin/env node
// count [FILE...]: print the number of words in each file (or stdin), then the total.
import { readFileSync } from 'node:fs';
const files = process.argv.slice(2);
const words = (text) => text.split(/\s+/).filter(Boolean).length;
if (!files.length) {
  console.log(words(readFileSync(0, 'utf8')));
} else {
  let total = 0;
  for (const file of files) { const n = words(readFileSync(file, 'utf8')); total += n; console.log(`${n} ${file}`); }
  if (files.length > 1) console.log(`${total} total`);
}
