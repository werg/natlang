import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { evaluateCodeObjective, runPythonSolveBatch } from '../dist/skills/code-objective.js';
import { PYTHON_SANDBOX_IMAGE } from '../dist/skills/graded.js';

const task = overrides => ({
  schema: 'natlang.skill-code-objective/1', id: 'sum-offset', revision: 'fixture-v1',
  description: 'Define solve(x) to add the two numbers in the input object.', functionName: 'solve',
  cases: [
    { id: 'small', group: 'arithmetic/basic', input: { left: 2, right: 3 }, expected: 5 },
    { id: 'negative', group: 'arithmetic/sign', input: { left: -4, right: 9 }, expected: 5 },
  ],
  ...overrides,
});
const correct = 'def solve(x):\n    return x["left"] + x["right"]\n';
const resultsFor = (inputs, fn) => ({ kind: 'results', results: inputs.map(input => fn(input)) });

test('one sandbox batch receives candidate source and inputs only; expected values stay host-side', () => {
  const requests = [];
  const result = evaluateCodeObjective(correct, task(), {
    execute(request) {
      requests.push(request);
      return resultsFor(request.inputs, input => ({ kind: 'ok', value: input.left + input.right }));
    },
  });
  assert.equal(result.status, 'scored');
  assert.equal(result.quality, 1);
  assert.deepEqual(result.gates, { source_size: true, all_correct: true });
  assert.equal(requests.length, 1);
  assert.deepEqual(Object.keys(requests[0]).sort(), ['inputs', 'source']);
  assert.equal(requests[0].inputs.length, 2);
  assert.ok(!Object.hasOwn(requests[0], 'expected') && !Object.hasOwn(requests[0], 'task'));
});

test('any wrong output gets zero correctness-first quality and size cannot rescue it', () => {
  let calls = 0;
  const result = evaluateCodeObjective(correct, task({ sizeObjective: {
    referenceSource: 'def solve(x): return x["left"] + x["right"]', bestKnownLowerBoundBytes: 5,
  } }), { execute(request) { calls++; return resultsFor(request.inputs, input => ({ kind: 'ok', value: input.left })); } });
  assert.equal(calls, 1, 'all cases are evaluated in one sandbox launch');
  assert.equal(result.status, 'candidate-error');
  assert.equal(result.quality, 0);
  assert.equal(result.gates.all_correct, false);
  assert.deepEqual(result.detail.failed_case_ids, ['small', 'negative']);
});

test('candidate exceptions are scored as candidate failures, infrastructure remains unscored', () => {
  const failed = evaluateCodeObjective(correct, task(), { execute: ({ inputs }) => resultsFor(inputs, () => ({ kind: 'candidate-error', error: { category: 'ValueError', message: 'bad input' } })) });
  assert.equal(failed.status, 'candidate-error');
  assert.equal(failed.quality, 0);
  const infrastructure = evaluateCodeObjective(correct, task(), {
    execute: () => ({ kind: 'infrastructure-error', category: 'docker-unavailable', message: 'daemon unavailable' }),
  });
  assert.equal(infrastructure.status, 'infrastructure-error');
  assert.equal(Object.hasOwn(infrastructure, 'quality'), false);
});

test('optional size score validates a reference batch after candidate correctness', () => {
  const calls = [];
  const referenceSource = 'def solve(x): return x["left"] + x["right"]';
  const objective = task({ sizeObjective: { referenceSource, bestKnownLowerBoundBytes: 5 } });
  const candidate = 'def solve(x):return x["left"]+x["right"]';
  const result = evaluateCodeObjective(candidate, objective, {
    execute(request) {
      calls.push(request.source);
      return resultsFor(request.inputs, input => ({ kind: 'ok', value: input.left + input.right }));
    },
  });
  assert.deepEqual(calls, [candidate, referenceSource]);
  assert.equal(result.status, 'scored');
  assert.ok(result.quality > 0 && result.quality <= 1);
  assert.equal(result.detail.reference_bytes, Buffer.byteLength(referenceSource));
  assert.equal(result.detail.lower_bound_is_proven_minimum, false);
});

test('invalid references and oversized sources fail closed before execution', () => {
  let calls = 0;
  const badReference = evaluateCodeObjective(correct, task({ sizeObjective: {
    referenceSource: 'def solve(x): return 0', bestKnownLowerBoundBytes: 1,
  } }), { execute(request) {
    calls++;
    return resultsFor(request.inputs, input => ({ kind: 'ok', value: request.source === correct ? input.left + input.right : 0 }));
  } });
  assert.equal(badReference.status, 'invalid-task');
  assert.equal(calls, 2, 'candidate batch runs, then the reference batch is validated');
  const tooLarge = evaluateCodeObjective(correct, task(), { maxSourceBytes: 1, execute() { throw Error('must not execute'); } });
  assert.equal(tooLarge.status, 'candidate-error');
  assert.equal(tooLarge.gates.source_size, false);
  const invalidTask = evaluateCodeObjective(correct, task({ cases: [{ id: 'bad', group: 'g', input: 1, expected: undefined }] }), { execute() { calls++; throw Error('must not execute'); } });
  assert.equal(invalidTask.status, 'invalid-task');
  assert.equal(calls, 2);
});

test('local sandbox isolates candidate stdout and mounts no host-held expected values', {
  skip: spawnSync('docker', ['image', 'inspect', PYTHON_SANDBOX_IMAGE, '--format', '{{.Id}}'], { encoding: 'utf8', timeout: 5000 }).status === 0 ? false : 'pinned Python image is not already available locally',
}, () => {
  const result = runPythonSolveBatch({
    source: 'import os\nprint("{\\"kind\\":\\"results\\",\\"results\\":[]}")\ndef solve(x): return {"files": sorted(os.listdir("/work")), "input": x}',
    inputs: [{ safe: 7 }, { safe: 8 }],
  });
  assert.deepEqual(result, { kind: 'results', results: [
    { kind: 'ok', value: { files: ['case_runner.py', 'runner.py', 'solution.py'], input: { safe: 7 } } },
    { kind: 'ok', value: { files: ['case_runner.py', 'runner.py', 'solution.py'], input: { safe: 8 } } },
  ] });
  const frameProbe = runPythonSolveBatch({
    source: [
      'import inspect',
      'def solve(x):',
      '    frame = inspect.currentframe()',
      '    while frame:',
      '        value = frame.f_locals.get("request")',
      '        if isinstance(value, dict) and isinstance(value.get("inputs"), list): return value["inputs"]',
      '        frame = frame.f_back',
      '    return x',
    ].join('\n'),
    inputs: [{ marker: 'heldout-a' }, { marker: 'heldout-b' }],
  });
  assert.deepEqual(frameProbe, { kind: 'results', results: [
    { kind: 'ok', value: { marker: 'heldout-a' } },
    { kind: 'ok', value: { marker: 'heldout-b' } },
  ] }, 'each candidate invocation can see only its current input through Python frames');
  const parentProbe = runPythonSolveBatch({
    source: [
      'import os',
      'def solve(x):',
      '    try:',
      '        handle = open("/proc/" + str(os.getppid()) + "/mem", "rb")',
      '        handle.close()',
      '        return True',
      '    except PermissionError:',
      '        return False',
    ].join('\n'),
    inputs: [1],
  });
  assert.deepEqual(parentProbe, { kind: 'results', results: [{ kind: 'ok', value: false }] },
    'the sandbox denies this same-UID child access to the batch runner memory on the tested Linux image');
  const allocationFailure = evaluateCodeObjective('def solve(x): raise MemoryError("bounded allocation exhausted")', task());
  assert.equal(allocationFailure.status, 'infrastructure-error', 'known memory exhaustion is not recorded as an incorrect answer');
  assert.equal(Object.hasOwn(allocationFailure, 'quality'), false);
  const syntaxFailure = runPythonSolveBatch({ source: 'def solve(:\n pass', inputs: [1, 2] });
  assert.equal(syntaxFailure.kind, 'results');
  assert.deepEqual(syntaxFailure.results.map(item => item.kind), ['candidate-error', 'candidate-error']);
});

test('non-finite host references cannot silently become JSON null and earn correctness', () => {
  const bad=task({cases:[{id:'bad',group:'bad',input:1,expected:NaN}]});
  let calls=0;const result=evaluateCodeObjective(correct,bad,{execute:()=>{calls++;return {kind:'results',results:[{kind:'ok',value:null}]};}});
  assert.equal(result.status,'invalid-task');assert.equal(calls,0);
});
