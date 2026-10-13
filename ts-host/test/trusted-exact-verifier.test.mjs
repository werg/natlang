import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { test } from 'node:test';
import { checkTrustedExactVerifier } from '../dist/teacher/trusted-exact-verifier.js';

async function fixtureCase(manifestPath) {
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const spec = { path: manifest.external_verifier_path,
    sha256: createHash('sha256').update(await readFile(manifest.external_verifier_path)).digest('hex'),
    oracle: JSON.parse(await readFile(manifest.oracle_path, 'utf8')) };
  const fixture = Object.fromEntries(await Promise.all(manifest.fixture_files.map(async path =>
    [path, await readFile(join(manifest.fixture_root, path), 'utf8')])));
  const inputs = { request: JSON.parse(await readFile(manifest.request_path, 'utf8')) };
  if (manifest.snapshot_path) inputs.snapshot = JSON.parse(await readFile(manifest.snapshot_path, 'utf8'));
  const value = JSON.parse(await readFile(manifest.reference_path, 'utf8'));
  return { spec, input: { value, oracle: spec.oracle, inputs, fixture } };
}

for (const [label, manifestPath, identityField] of [
  ['Saga', 'training/semantic_applications/astra/saga-forensics/project.json', 'caseId'],
  ['Witness', 'training/semantic_applications/sol/witness-rewrite-studio/project.json', 'requestId'],
]) test(`${label} exact verifier accepts canonical invariants, rejects mutation, and stays review pending`, async () => {
  const { spec, input } = await fixtureCase(manifestPath);
  const good = await checkTrustedExactVerifier(spec, input);
  assert.equal(good.accepted, true);
  assert.equal(good.needs_review, true);
  const bad = await checkTrustedExactVerifier(spec,
    { ...input, value: { ...input.value, [identityField]: 'wrong-identity' } });
  assert.equal(bad.accepted, false);
  await assert.rejects(checkTrustedExactVerifier({ ...spec, sha256: '0'.repeat(64) }, input),
    /source changed/);
});
