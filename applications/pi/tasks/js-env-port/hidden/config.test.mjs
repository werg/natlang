import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const { adminPort, loadConfig } = await import(pathToFileURL(`${process.cwd()}/src/config.js`).href);
test('hidden', () => {
  assert.deepEqual(loadConfig({ PORT: '5000', HOST: 'example', DEBUG: '1' }), { host: 'example', port: 5000, debug: true });
  assert.equal(adminPort(loadConfig({ PORT: '9' })), 10);
  assert.equal(adminPort(loadConfig({})), 3001);
});
