import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { HELP_TOPICS } from '../dist/cli/main.js';
import { parseArgs } from '../dist/cli/args.js';
import { progressReporter, reportedFailure } from '../dist/cli/progress.js';
import { closestName } from '../dist/compiler/suggest.js';
import { executeNatlangCall } from '../dist/native/natlang-command.js';

const cli = new URL('../bin/natlang.mjs', import.meta.url).pathname;
function run(args, cwd, env = process.env) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [cli, ...args], { cwd, env, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '', err = '';
    child.stdout.on('data', chunk => { out += chunk; }); child.stderr.on('data', chunk => { err += chunk; });
    child.once('error', reject); child.once('close', code => resolve({ code, out, err }));
  });
}
const isolated = root => ({ ...process.env, NATLANG_CALL_STORE: 'off', NATLANG_HASH_CACHE: join(root, 'hashes.jsonl'), NATLANG_CONFIG_HOME: root,
  NATLANG_STATE_HOME: join(root, 'state'), NATLANG_HOME: join(root, 'packages') });

test('the version is the package version, and help covers every command it lists', async () => {
  const version = JSON.parse(readFileSync(new URL('../../npm-packages/node/package.json', import.meta.url), 'utf8')).version;
  const shown = await run(['--version'], tmpdir());
  assert.equal(shown.out, `${version}\n`);
  const help = await run(['help'], tmpdir()), advanced = await run(['help', 'advanced'], tmpdir());
  assert.match(help.out, /^Start:$/m); assert.match(help.out, /^Models:$/m); assert.match(help.out, /^Inspect:$/m);
  assert.match(help.out, /natlang help advanced/); assert.match(help.out, /Exit codes: 0 ok/);
  assert.doesNotMatch(help.out + advanced.out, /\bPi\b/, 'providers are named hosted model providers');
  const commands = [...`${help.out}\n${advanced.out}`.matchAll(/^ {2}natlang (\S+)/gm)].map(match => match[1]);
  // One description column per listing.
  for (const listing of [help.out, advanced.out])
    assert.equal(new Set([...listing.matchAll(/^ {2}natlang \S.*?\S {2,}(?=\S)/gm)].map(match => match[0].length)).size, 1);
  for (const command of new Set(commands)) {
    assert.ok(HELP_TOPICS[command], `natlang help ${command}`);
    const asked = await run([command, '--help'], tmpdir());
    assert.equal(asked.code, 0, `${command} --help: ${asked.err}`);
    assert.equal(asked.out, `${HELP_TOPICS[command]}\n`);
  }
});

test('an unknown option is named with the nearest one before it can take a value', async () => {
  assert.throws(() => parseArgs(['check', '--jsno']), /unknown option --jsno for natlang check \(did you mean --json\?\)/);
  assert.deepEqual(parseArgs(['skills', 'install', '--claude']).options.get('--claude'), true);
  const typo = await run(['check', '--jsno', '.'], tmpdir());
  assert.equal(typo.code, 2); assert.match(typo.err, /did you mean --json\?/);
  assert.equal((await run(['chek'], tmpdir())).code, 2);
  assert.equal(closestName('Strng', ['string', 'number']), 'string');
  assert.equal(closestName('zzz', ['string']), undefined);
});

test('usage errors exit 2, an unreachable model exits 3 with one line, and check --json is the contract shape', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-cli-ux-')), program = join(root, 'p');
  const env = { ...isolated(root), NATLANG_SERVER: 'http://127.0.0.1:9', NATLANG_MODEL: 'fixture' };
  try {
    assert.equal((await run(['init', program], root, env)).code, 0);
    const nested = await run(['init', 'inner'], program, env);
    assert.match(nested.err, /already a program/);
    for (const args of [['call', 'answer.nl'], ['call', 'nope.nl'], ['run', 'nope'], ['call', 'answer.nl', '--inputs', 'missing.json'], ['test', 'missing']]) {
      const result = await run(args, program, env);
      assert.equal(result.code, 2, `${args.join(' ')}: ${result.err}`);
      assert.equal(result.err.trim().split('\n').length, 1, result.err);
    }
    assert.match((await run(['call', 'nope.nl'], program, env)).err, /nope\.nl does not exist/);
    assert.match((await run(['run', 'nope'], program, env)).err, /neither a directory nor an installed application/);
    for (const args of [['call', 'answer.nl', 'hi'], ['run'], ['test']]) {
      const result = await run(args, program, env);
      assert.equal(result.code, 3, `${args.join(' ')}: ${result.err}`);
      assert.equal(result.err.split('\n').filter(line => /not reachable/.test(line)).length, 1, result.err);
    }
    const unavailable = await run(['test', '--json'], program, env);
    assert.equal(unavailable.code, 3);
    assert.equal(JSON.parse(unavailable.out).error.kind, 'model-unavailable');
    writeFileSync(join(program, 'bad.ts'), 'const x: number = "a";\nexport {};\n');
    const checked = await run(['check', '--json'], program, env);
    assert.equal(checked.code, 1);
    const report = JSON.parse(checked.out);
    assert.deepEqual(Object.keys(report).sort(), ['diagnostics', 'functions', 'ok']);
    assert.deepEqual(report.diagnostics, [{ file: 'bad.ts', line: 1, column: 7, code: 'TS2322', message: "Type 'string' is not assignable to type 'number'." }]);
    assert.equal(report.functions[0].source, 'answer.nl');
    assert.ok(JSON.parse((await run(['check', '--json', '--manifest'], program, env)).out).manifest);
    const failedTest = await run(['test', '--json'], program, env);
    assert.equal(JSON.parse(failedTest.out).ok, false); assert.ok(JSON.parse(failedTest.out).diagnostics.length);
    rmSync(join(program, 'bad.ts'));
    const source = readFileSync(join(program, 'answer.nl'), 'utf8');
    writeFileSync(join(program, 'answer.nl'), source.replace(/^returns: .*$/m, 'returns: Strng'));
    const unknownType = await run(['check'], program, env);
    assert.match(unknownType.out, /answer\.nl:5:1 .*unknown type name Strng\n {2}hint: did you mean string\?/);
    writeFileSync(join(program, 'answer.nl'), source.replace(/^returns: (.*)$/m, 'returns: $1\nreadout: sometimes'));
    assert.match((await run(['check'], program, env)).out, /answer\.nl:6:1 .*readout must be decision or template/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('call prints text results as text, reads piped text for a text parameter, and names its call on stderr', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-cli-call-')), program = join(root, 'p');
  const server = createServer((request, response) => {
    request.resume(); request.on('end', () => {
      response.setHeader('content-type', 'application/json');
      if (request.url.endsWith('/models')) return response.end(JSON.stringify({ data: [{ id: 'fixture' }] }));
      response.end(JSON.stringify({ choices: [{ message: { content: '"billing"', tool_calls: [] }, finish_reason: 'stop' }], usage: { completion_tokens: 1 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const env = { ...isolated(root), NATLANG_CALL_STORE: join(root, 'calls'), NATLANG_SERVER: `http://127.0.0.1:${server.address().port}`, NATLANG_MODEL: 'fixture' };
  try {
    assert.equal((await run(['init', program], root, env)).code, 0);
    const json = await run(['call', 'answer.nl', 'hi', '--json'], program, env);
    assert.equal(json.out, '"billing"\n');
    assert.match(json.err, /^natlang: call (\S+) \(natlang traces show \1\)$/m);
    const piped = await new Promise(resolve => {
      const child = spawn(process.execPath, [cli, 'call', 'answer.nl'], { cwd: program, env, stdio: ['pipe', 'pipe', 'pipe'] });
      let out = ''; child.stdout.on('data', chunk => { out += chunk; });
      child.stdin.end('I was charged twice\n');
      child.once('close', code => resolve({ code, out }));
    });
    assert.deepEqual(piped, { code: 0, out: 'billing\n' });
  } finally { await new Promise(resolve => server.close(resolve)); rmSync(root, { recursive: true, force: true }); }
  const seen = [];
  assert.equal(await executeNatlangCall('f', 'plain text\n', 'whole', async (_name, input) => { seen.push(input); return 'ok'; }, { text: true }), 'ok\n');
  assert.equal(await executeNatlangCall('f', '{"input":"x"}', 'whole', async (_name, input) => { seen.push(input); return 'ok'; }, { text: true, json: true }), '"ok"\n');
  assert.deepEqual(seen, ['plain text', { input: 'x' }]);
});

test('progress reports a failed call once, without the request error it failed on', () => {
  let written = '';
  const stream = { isTTY: false, write(chunk) { written += chunk; return true; } };
  const reporter = progressReporter(stream, 'line');
  const call = { taskId: 't', callId: 'c', parentCallId: null, definitionId: 'd', name: 'answer' };
  reporter.live({ type: 'call_start', ...call });
  reporter.live({ type: 'trace', ...call, event: { kind: 'model_request', phase: 'error', error: 'connection refused' } });
  reporter.live({ type: 'call_end', ...call, outcome: 'failed', detail: 'model request failed: connection refused' });
  reporter.close();
  assert.equal(written, 'natlang: answer failed: model request failed: connection refused\n');
  assert.ok(reportedFailure('answer: failed: model request failed: connection refused'));
  written = '';
  const silent = progressReporter(stream, 'line', { roots: false });
  silent.live({ type: 'call_start', ...call, callId: 'r' });
  silent.live({ type: 'call_end', ...call, callId: 'r', outcome: 'failed', detail: 'x' });
  silent.close();
  assert.equal(written, '', 'natlang test prints root outcomes itself');
});
