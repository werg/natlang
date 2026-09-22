import test from 'node:test';
import assert from 'node:assert/strict';
import { parseArgs } from '../scripts/code-corpus/direct-pilot.mjs';

test('direct upstream pilot requires explicit execution and a new output path', () => {
  assert.deepEqual(parseArgs([]), { execute: false, output: undefined, functionName: 'ascending' });
  assert.deepEqual(parseArgs(['--output', '/tmp/pilot']), { execute: false, output: '/tmp/pilot', functionName: 'ascending' });
  assert.deepEqual(parseArgs(['--execute', '--output', '/tmp/pilot']), { execute: true, output: '/tmp/pilot', functionName: 'ascending' });
  assert.deepEqual(parseArgs(['--execute', '--output', '/tmp/pilot', '--function', 'mean']), { execute: true, output: '/tmp/pilot', functionName: 'mean' });
  assert.deepEqual(parseArgs(['--execute', '--output', '/tmp/pilot', '--function', 'transpose']), { execute: true, output: '/tmp/pilot', functionName: 'transpose' });
});
