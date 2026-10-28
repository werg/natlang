import {test} from 'node:test';
import assert from 'node:assert/strict';
import {checkRequiredActions} from '../dist/teacher/required-actions.js';

test('required root read_code is exact, successful and cannot be satisfied by a child', () => {
  const spec = [{kind: 'read_code', name: 'skills.judge-against-criteria'}];
  const event = {kind: 'action', call_id: 'root', name: 'read_code', outcome: 'ok',
    arguments: {name: 'skills.judge-against-criteria'}};
  assert.equal(checkRequiredActions(spec, 'root', [event]), true);
  assert.equal(checkRequiredActions(spec, 'root', []), false);
  assert.equal(checkRequiredActions(spec, 'root', [{...event, call_id: 'child'}]), false);
  assert.equal(checkRequiredActions(spec, 'root', [{...event, outcome: 'error'}]), false);
  assert.equal(checkRequiredActions(spec, 'root', [{...event, arguments: {name: 'skills.other'}}]), false);
  assert.throws(() => checkRequiredActions([{kind: 'read_code', name: ''}], 'root', [event]));
});
