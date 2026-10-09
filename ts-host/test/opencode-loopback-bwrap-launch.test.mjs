import test from 'node:test';
import assert from 'node:assert/strict';
import { realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { buildBubblewrapArgs } from '../../scripts/opencode-loopback-bwrap-launch.mjs';

test('bubblewrap command exposes only the workspace adapter, pinned package tree, and one writable run directory', () => {
  const nodeModules = '/home/werg/natlang/runs/opencode-exo-client/client/node_modules';
  const output = '/home/werg/natlang/runs/opencode-ling-smoke/run-1';
  const args = buildBubblewrapArgs({ sdkModule: `${nodeModules}/@opencode-ai/sdk/dist/v2/index.js`,
    clientBin: `${nodeModules}/.bin/opencode`, output, nodeModules,
    forwardedOptions: { '--model': 'ling-3.1-flash-free', '--max-concurrency': '1' } });
  const has = (...parts) => {
    for (let index = 0; index <= args.length - parts.length; index++)
      if (parts.every((part, offset) => args[index + offset] === part)) return true;
    return false;
  };
  assert.equal(has('--ro-bind', '/usr', '/usr'), true);
  assert.equal(has('--ro-bind', realpathSync(fileURLToPath(new URL('../../scripts', import.meta.url))), '/home/werg/natlang/scripts'), true);
  assert.equal(has('--ro-bind', nodeModules, nodeModules), true);
  assert.equal(has('--bind', output, output), true);
  assert.equal(has('--share-net'), true, 'provider egress remains enabled for official transport');
  assert.equal(has('--ro-bind', '/home/werg', '/home/werg'), false);
  assert.equal(has('--ro-bind', '/home/werg/natlang', '/home/werg/natlang'), false);
  assert.equal(args.includes('/home/werg/natlang/training'), false);
  assert.equal(args.includes(`${nodeModules}/../..`), false);
  assert.equal(args.at(-1), '1');
});
