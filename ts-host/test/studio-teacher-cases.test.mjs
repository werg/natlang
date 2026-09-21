import assert from 'node:assert/strict';
import { test } from 'node:test';
import { apps } from '../studio/apps/index.mjs';
import { freeze } from '../scripts/freeze-studio-teacher-cases.mjs';

test('teacher case freezer gives every discovered Studio app varied train and eval cases', async () => {
  const rows = await freeze({ minimum: 6 });
  assert.equal(new Set(rows.map(row => row.id)).size, rows.length);
  assert.deepEqual(new Set(rows.map(row => row.target)), new Set(apps.map(spec => `studio:${spec.id}`)));
  for (const spec of apps) {
    const own = rows.filter(row => row.target === `studio:${spec.id}`);
    assert.equal(own.length, 6);
    assert.equal(own.filter(row => row.split === 'train').length, 5);
    assert.equal(own.filter(row => row.split === 'eval').length, 1);
    assert.ok(own.every(row => row.source_revision.length === 64));
    assert.ok(own.every(row => typeof row.expected.ok === 'boolean'));
  }
});
