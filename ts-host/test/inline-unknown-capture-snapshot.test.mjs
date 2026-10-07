import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const runtimeBase = process.env.NATLANG_RUNTIME_TEST_BUILD ?
  pathToFileURL(`${resolve(process.env.NATLANG_RUNTIME_TEST_BUILD)}/`) : new URL('../dist/', import.meta.url);
const [{ callableMeta }, { inline }, { NATLANG_COMPILE_VERSION }, { fingerprint }] = await Promise.all([
  import(new URL('runtime/callable.js', runtimeBase)), import(new URL('runtime/lowered.js', runtimeBase)),
  import(new URL('compiler/intrinsics.js', runtimeBase)), import(new URL('adaptation/identity.js', runtimeBase)),
]);

const sha = value => createHash('sha256').update(value).digest('hex');
const origin = { parentInvocationId: 'parent-invocation', toolCallId: 'eval-call-7', actionOrdinal: 3,
  writtenCodeSha256: sha('authored code'), checkedCodeSha256: sha('checked code') };

function makePlan(capture) {
  return { definitionId: 'unknown-capture-test', sourceSpan: { file: 'eval', start: 0, end: 32 },
    templateSpan: { start: 12, end: 32 }, strings: ['semantic instruction'], interpolations: [],
    instructions: 'semantic instruction', parameters: [], returns: { text: 'string', natlang: 'string' },
    captures: [capture], explicitCaptures: true };
}

function create(capture, getValue, mode = 'snapshot') {
  return inline(makePlan({ name: 'observed', type: capture.type, source: 'local', mutable: mode === 'live', mode }), [],
    { observed: [getValue] }, undefined, NATLANG_COMPILE_VERSION, undefined, origin);
}

test('unknown descriptor records a host-observed primitive type after one getter read', () => {
  let reads = 0;
  const fn = create({ type: { text: 'any', natlang: 'unknown' } }, () => { reads++; return 'actual host value'; });
  const site = callableMeta(fn).options.manifest.inline_instruction_site;
  assert.equal(reads, 1);
  assert.deepEqual(site.runtime_capture_snapshots.captures, [{ name: 'observed', type: 'string', declared_type: 'unknown',
    source: 'local', value: 'actual host value', mode: 'snapshot',
    value_canonical: JSON.stringify({ type: 'string', value: 'actual host value' }),
    value_sha256: fingerprint({ type: 'string', value: 'actual host value' }, 'natlang.inline-capture-snapshot/v1'),
    creation: { parentInvocationId: origin.parentInvocationId, toolCallId: origin.toolCallId,
      actionOrdinal: origin.actionOrdinal, writtenCodeSha256: origin.writtenCodeSha256,
      checkedCodeSha256: origin.checkedCodeSha256, definitionId: 'unknown-capture-test',
      sourceSpan: { file: 'eval', start: 0, end: 32 }, templateSpan: { start: 12, end: 32 },
      checkedTemplateSpan: { start: 12, end: 32 } } }]);
  assert.match(site.runtime_capture_snapshots.captures[0].value_sha256, /^[a-f0-9]{64}$/);
});

test('unknown descriptors use the actual boolean or finite number type and reject negative zero', () => {
  for (const [value, type] of [[true, 'boolean'], [12.5, 'number']]) {
    const fn = create({ type: { text: 'any', natlang: 'unknown' } }, () => value);
    const snapshot = callableMeta(fn).options.manifest.inline_instruction_site.runtime_capture_snapshots.captures[0];
    assert.equal(snapshot.type, type);
    assert.equal(snapshot.declared_type, 'unknown');
    assert.equal(snapshot.value, value);
  }
  const negativeZero = create({ type: { text: 'any', natlang: 'unknown' } }, () => -0);
  assert.equal(Object.hasOwn(callableMeta(negativeZero).options.manifest.inline_instruction_site, 'runtime_capture_snapshots'), false);
});

test('unknown descriptors never attest object, FileHandle, or live captures', () => {
  const cases = [
    [{ text: 'any', natlang: 'unknown' }, { note: 'opaque object' }, 'snapshot'],
    [{ text: 'FileHandle', natlang: 'unknown' }, { path: 'records/private.md' }, 'snapshot'],
    [{ text: 'any', natlang: 'unknown' }, 'changing value', 'live'],
  ];
  for (const [type, value, mode] of cases) {
    const fn = create({ type }, () => value, mode);
    assert.equal(Object.hasOwn(callableMeta(fn).options.manifest.inline_instruction_site, 'runtime_capture_snapshots'), false);
  }
});
