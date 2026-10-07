import { test } from 'node:test';
import assert from 'node:assert/strict';
import { getUser, team, greet, audit } from '../src/index.js';
test('getUser finds Ada', async () => assert.equal((await getUser(1)).name, 'Ada'));
test('getUser misses', async () => assert.equal(await getUser(9), null));
test('team', async () => assert.deepEqual(await team('core'), ['Ada', 'Linus']));
test('greet', async () => assert.equal(await greet(2), 'Hello, Grace!'));
test('audit', async () => assert.match(await audit(3, 'login'), /Linus login$/));
