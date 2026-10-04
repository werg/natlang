import assert from 'node:assert/strict';
import { test } from 'node:test';
import { APPROACH_PROMPT } from '../dist/native/prompt.js';

test('prompt gives a direct example for using an already-bound string input', () => {
  assert.match(APPROACH_PROMPT, /if it lists `instance: string`, use `JSON\.parse\(instance\)`/);
  assert.match(APPROACH_PROMPT, /do not redeclare `instance` or copy it from `inputs\.instance`/);
});
