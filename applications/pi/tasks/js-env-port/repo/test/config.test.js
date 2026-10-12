import { test } from 'node:test';
import assert from 'node:assert/strict';
import { adminPort, loadConfig } from '../src/config.js';

test('defaults', () => assert.deepEqual(loadConfig({}), { host: 'localhost', port: 3000, debug: false }));
test('admin port from env', () => assert.equal(adminPort(loadConfig({ PORT: '8080' })), 8081));
