import assert from 'node:assert/strict';
import test from 'node:test';
import {checkRequiredRootIteration} from '../dist/teacher/required-iteration.js';

test('requires one completed root iterateOn with actual steps', () => {
  const spec={kind:'one-completed-iterate-on'};
  const step={kind:'iteration_step',iteration_id:'i1',phase:'step'};
  const done={kind:'iteration_step',iteration_id:'i1',phase:'done'};
  assert.equal(checkRequiredRootIteration(spec,[]),false);
  assert.equal(checkRequiredRootIteration(spec,[step]),false);
  assert.equal(checkRequiredRootIteration(spec,[step,done]),true);
  assert.equal(checkRequiredRootIteration(spec,[step,done,{...step,iteration_id:'i2'}]),false);
  assert.equal(checkRequiredRootIteration(undefined,[]),true);
  assert.throws(()=>checkRequiredRootIteration({kind:'other'},[]));
});
