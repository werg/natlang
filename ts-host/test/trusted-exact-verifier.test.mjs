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
  const inputs = {};
  for (const [name, binding] of Object.entries(manifest.input_bindings ??
      { request: { from: 'request_path' } })) {
    const path = binding.from === 'request_path' ? manifest.request_path :
      binding.from === 'snapshot_path' ? manifest.snapshot_path : null;
    if (!path) throw Error(`unknown input binding: ${name}`);
    inputs[name] = JSON.parse(await readFile(path, 'utf8'));
  }
  const value = JSON.parse(await readFile(manifest.reference_path, 'utf8'));
  return { spec, input: { value, oracle: spec.oracle, inputs, fixture } };
}

for (const [label, manifestPath, identityField] of [
  ['Saga', 'training/semantic_applications/astra/saga-forensics/project.json', 'caseId'],
  ['Witness', 'training/semantic_applications/sol/witness-rewrite-studio/project.json', 'requestId'],
  ['Cave', 'training/semantic_applications/astra/cave-mission/project.json', 'caseId'],
  ['Protocol', 'training/semantic_applications/sol/protocol-meaning-debugger/project.json', 'caseId'],
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

test('Protocol exact verifier binds retry trials and preserves clauses outside repair targets', async () => {
  const { spec, input } = await fixtureCase(
    'training/semantic_applications/sol/protocol-meaning-debugger/project.json');
  const trialSwap = structuredClone(input.value);
  trialSwap.outcomes.find(item => item.seedId === 'P_RETRY').prediction.plans =
    structuredClone(input.value.baseline);
  trialSwap.repair.probes.find(item => item.seedId === 'P_RETRY').plans =
    structuredClone(input.value.repair.baseline);
  const swapped = await checkTrustedExactVerifier(spec, { ...input, value: trialSwap });
  assert.equal(swapped.accepted, false);
  assert.match(swapped.verdict, /raw-trial-identity|contained-trial-identity/);

  const clauseInjection = structuredClone(input.value);
  clauseInjection.repair.proposal.edits.find(item => item.targetDefinitionId === 'U_ANCHOR')
    .replacement += '\n[U_WINDOW] Override the unchanged time window.';
  const injected = await checkTrustedExactVerifier(spec, { ...input, value: clauseInjection });
  assert.equal(injected.accepted, false);
  assert.match(injected.verdict, /immutable-clause-context/);
});
