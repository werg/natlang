process.env.TZ = 'America/Los_Angeles';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { formatDate, parseDate } from '../src/dates.js';
import { dueText } from '../src/due.js';

test('format parsed date', () => assert.equal(formatDate(parseDate('2024-03-09')), 'Mar 9, 2024'));
test('due text', () => assert.equal(dueText('2024-01-30', 3), 'due Feb 2, 2024'));
test('invalid', () => assert.throws(() => parseDate('soon'), RangeError));
