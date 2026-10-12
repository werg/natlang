import { test } from 'node:test';
import assert from 'node:assert/strict';
import { countMatches, highlight } from '../src/highlight.js';

test('plain word', () => assert.equal(highlight('Cat and cat', 'cat'), '<mark>Cat</mark> and <mark>cat</mark>'));
test('special characters', () => assert.equal(highlight('I like C++ and c', 'c++'), 'I like <mark>C++</mark> and c'));
test('dot is literal', () => assert.equal(countMatches('a.b axb', 'a.b'), 1));
