import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, renameSync, rmSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const staged = existsSync(join(root, 'npm-packages/node/dist/cli/main.js')) && existsSync(join(root, 'npm-packages/node/skills'));

function run(command, args, options) {
  return new Promise((done, reject) => {
    const child = spawn(command, args, { ...options, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '', err = '';
    child.stdout.on('data', chunk => { out += chunk; }); child.stderr.on('data', chunk => { err += chunk; });
    child.once('error', reject); child.once('close', code => done({ code, out, err }));
  });
}

// The release path end to end: the packed @natlang/node and @natlang/cli, installed by scripts/install.sh into a fresh
// prefix and home, give a `natlang` that works without this checkout. npm fetches the packages' dependencies.
test('install.sh installs the packed CLI, which creates, checks and runs a program and ships its skills', { skip: !staged, timeout: 600000 }, async () => {
  const scratch = mkdtempSync(join(tmpdir(), 'natlang-install-'));
  const release = join(scratch, 'release'), prefix = join(scratch, 'prefix'), home = join(scratch, 'home');
  const server = createServer((request, response) => {
    request.resume(); request.on('end', () => {
      response.setHeader('content-type', 'application/json');
      if (request.url.endsWith('/models')) return response.end(JSON.stringify({ data: [{ id: 'fixture' }] }));
      response.end(JSON.stringify({ choices: [{ message: { content: '"billing"', tool_calls: [] }, finish_reason: 'stop' }], usage: { completion_tokens: 1 } }));
    });
  });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  try {
    mkdirSync(release); mkdirSync(home);
    execFileSync('npm', ['pack', '--ignore-scripts', '--workspace', '@natlang/node', '--workspace', '@natlang/cli', '--pack-destination', release],
      { cwd: root, stdio: 'ignore' });
    for (const name of readdirSync(release)) renameSync(join(release, name), join(release, name.replace(/-\d+\.\d+\.\d+.*\.tgz$/, '.tgz')));
    const env = { PATH: `${join(prefix, 'bin')}:${process.env.PATH}`, HOME: home, npm_config_cache: process.env.npm_config_cache ?? join(process.env.HOME, '.npm'),
      npm_config_prefer_offline: 'true', NATLANG_CONFIG_HOME: join(home, 'config'), NATLANG_STATE_HOME: join(home, 'state'),
      NATLANG_CACHE_HOME: join(home, 'cache'), NATLANG_HOME: join(home, 'packages'), NATLANG_CALL_STORE: 'off',
      NATLANG_SERVER: `http://127.0.0.1:${server.address().port}`, NATLANG_MODEL: 'fixture' };
    const installed = await run('sh', [join(root, 'scripts/install.sh')], { cwd: scratch, env: { ...env, NATLANG_FROM: release, NATLANG_PREFIX: prefix } });
    assert.equal(installed.code, 0, installed.err);
    assert.match(installed.out, /natlang is installed/);
    const natlang = (args, cwd = home) => run(join(prefix, 'bin', 'natlang'), args, { cwd, env });
    const program = join(home, 'my-program');
    for (const [args, cwd] of [[['init', program]], [['check'], program]]) {
      const result = await natlang(args, cwd); assert.equal(result.code, 0, `${args.join(' ')}: ${result.err}`);
    }
    const called = await natlang(['call', 'answer.nl', 'I was charged twice'], program);
    assert.equal(called.code, 0, called.err); assert.equal(called.out, 'billing\n');
    const ran = await natlang(['run'], program);
    assert.equal(ran.code, 0, ran.err); assert.equal(ran.out, 'billing\n');
    const skills = await natlang(['skills', '--json']);
    assert.equal(skills.code, 0, skills.err);
    assert.ok(JSON.parse(skills.out).root.startsWith(prefix), 'the skills come from the installed package, not the checkout');
    const copied = await natlang(['skills', 'install', '--dir', join(home, 'agent-skills')]);
    assert.equal(copied.code, 0, copied.err);
    assert.ok(existsSync(join(home, 'agent-skills', 'natlang-authoring', 'SKILL.md')));
  } finally {
    await new Promise(done => server.close(done));
    rmSync(scratch, { recursive: true, force: true });
  }
});
