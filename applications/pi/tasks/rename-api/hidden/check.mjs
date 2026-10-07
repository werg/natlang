import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const index = await import(pathToFileURL(`${process.cwd()}/src/index.js`).href);
const users = await import(pathToFileURL(`${process.cwd()}/src/users.js`).href);
assert.equal((await index.fetchUser(1)).name, 'Ada');
assert.equal(typeof users.getUser, 'function', 'getUser stays as a deprecated alias');
assert.equal((await users.getUser(2)).name, 'Grace');
