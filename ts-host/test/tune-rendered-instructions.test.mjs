import assert from 'node:assert/strict';
import { test } from 'node:test';
import { renderedInstructions } from '../dist/tune/pool.js';

test('an inline lambda\'s rendered instruction text is recovered from its call\'s messages', () => {
  const messages = [{ role: 'system', content: 'You run natlang.' },
    { role: 'user', content: 'Instructions:\nSummarize the report for Ada in 3 bullets.\n\nIn eval you can use report.' }];
  assert.equal(renderedInstructions(['Summarize the report for ', ' in ', ' bullets.'], messages), 'Summarize the report for Ada in 3 bullets.');
  assert.equal(renderedInstructions(['\nSummarize the report for Ada in 3 bullets.\n'], messages), 'Summarize the report for Ada in 3 bullets.',
    'a named body, trimmed of its newlines as the opening message shows it');
  assert.equal(renderedInstructions(['Translate '], messages), undefined, 'absent');
  assert.equal(renderedInstructions(['Summarize the report for ', ''], messages), undefined, 'a value at the end has no delimiter');
  assert.equal(renderedInstructions(['report'], messages), undefined, 'more than once');
});
