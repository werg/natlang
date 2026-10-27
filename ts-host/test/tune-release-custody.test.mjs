import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync, existsSync, unlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { uploadRelease } from '../dist/cli/weights.js';
import { MemoryNeuraleseStore, nativeStateContentId, encodeNativeState } from '../dist/native/neuralese-store.js';
import { nativeStateFixture } from './fixtures/native-state.mjs';
import { copyNativeReleaseClosure, installNativeReleaseObject, materializeNativeInferenceState, materializeNativeStateGguf, verifyNativeInferenceMaterialization,
  verifyNativeInferenceSnapshotReceipt, nativeReleaseObjectClosure } from '../dist/native/release-artifact-closure.js';
import { materializeExistingInferenceRelease } from '../dist/tune/pipeline.js';
import { sealManifest } from '../dist/adaptation/weights/manifest.js';
import { storeRelease } from '../dist/adaptation/weights/release.js';

test('managed state export binds the compact identity without changing its payload', () => {
  const source = `nzs1_${'a'.repeat(52)}`, target = `nzs1_${'b'.repeat(52)}`;
  const key = Buffer.from('natlang.state.id'), length = Buffer.alloc(8), descriptor = Buffer.alloc(12);
  length.writeBigUInt64LE(BigInt(key.length)); descriptor.writeUInt32LE(8); descriptor.writeBigUInt64LE(BigInt(source.length), 4);
  const header = Buffer.alloc(24); header.write('GGUF'); header.writeUInt32LE(3, 4); header.writeBigUInt64LE(1n, 16);
  const bytes = Buffer.concat([header, length, key, descriptor, Buffer.from(source), Buffer.from('payload')]);
  const hash = createHash('sha256').update(bytes).digest('hex');
  const result = materializeNativeStateGguf(bytes, hash, source, target);
  assert.equal(Buffer.from(result.bytes).toString('hex'), Buffer.from(bytes.toString('latin1').replace(source, target), 'latin1').toString('hex'));
  assert.equal(result.sha256, createHash('sha256').update(result.bytes).digest('hex'));
  assert.throws(() => materializeNativeStateGguf(bytes, '0'.repeat(64), source, target), /receipt/);
  assert.throws(() => materializeNativeStateGguf(bytes, hash, target, source), /missing or ambiguous/);
});

test('release restores derived states parents first without failed upload probes and reuses exact objects', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'tune-release-custody-'));
  try {
    mkdirSync(join(directory, 'states'));
    const parent = nativeStateFixture(), parentId = nativeStateContentId(parent.metadata, parent.tensors);
    const identity = Object.fromEntries(['checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture'].map(key => [key, parent.metadata[key]]));
    let child, childId;
    for (let i = 0; i < 1000; i++) {
      child = { ...parent, metadata: { ...parent.metadata, derivation: {
        schema: 'natlang.native-state-derivation/1', kind: 'semantic-adapt', origin_state_ids: [parentId], parent_state_id: parentId,
        adapter_sha256: i.toString(16).padStart(64, '0'), source_native_identity: identity, target_native_identity: identity,
      } } };
      childId = nativeStateContentId(child.metadata, child.tensors);
      if (childId < parentId) break;
    }
    assert.ok(childId < parentId, 'child must sort before its required parent');
    for (const [id, artifact] of [[parentId, parent], [childId, child]]) writeFileSync(join(directory, 'states', `${id}.nzs`), encodeNativeState(artifact));
    const remote = new MemoryNeuraleseStore(), original = remote.putNativeState.bind(remote), uploads = [];
    remote.putNativeState = async artifact => { uploads.push(nativeStateContentId(artifact.metadata, artifact.tensors)); return original(artifact); };
    // Only the requested serving root and its exact derivation ancestry are published.
    const unused = nativeStateFixture(), unusedId = nativeStateContentId(unused.metadata, unused.tensors);
    writeFileSync(join(directory, 'states', `${unusedId}.nzs`), encodeNativeState(unused));
    assert.deepEqual(await uploadRelease(remote, directory, [childId]), new Set([childId, parentId]));
    assert.deepEqual(uploads, [parentId, childId]);
    await uploadRelease(remote, directory, [childId]);
    assert.equal(uploads.length, 2, 'already verified objects require no repeated upload');
    writeFileSync(join(directory, 'states', `${childId}.nzs`), encodeNativeState(parent));
    await assert.rejects(uploadRelease(remote, directory, [childId]), /does not hold its named content/);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test('inference materialization preserves the complete current state payload and leaves optimizer history in source', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'tune-release-materialized-state-'));
  try {
    mkdirSync(join(directory, 'states'));
    const parent = nativeStateFixture('lfm2');
    const parentId = nativeStateContentId(parent.metadata, parent.tensors);
    const tensorId = `nnt1_${'a'.repeat(52)}`;
    const derivation = { schema: 'natlang.native-state-derivation/1', kind: 'optimizer', origin_state_ids: [parentId],
      parent_state_id: parentId, gradient_id: tensorId, update_artifact_id: tensorId, program_id: null,
      optimizer_state_parent_id: null, step: 1, optimizer: { name: 'adam', learning_rate_hex: '0x1.0000000000000p-3',
        weight_decay_hex: '0x0.0p+0', beta1_hex: '0x1.999999999999ap-1', beta2_hex: '0x1.fae147ae147aep-1',
        eps_hex: '0x1.5798ee2308c3ap-27', momentum_hex: '0x0.0p+0' },
      updated_tensor_names: ['layers.0.raw_k'], arithmetic: { schema: 'natlang.native-state-optimizer-torch-cpu/1',
        cpu_capability: 'DEFAULT', float32_subnormal_hex: '0x1.0000000000000p-149',
        implementation_sha256: '4'.repeat(64), libc_version: 'glibc/test', machine: 'test', python_version: '3.12.3',
        torch_build_sha256: '5'.repeat(64), torch_git_version: 'test', torch_version: '2.11.0' } };
    const source = { metadata: { ...parent.metadata, derivation }, tensors: parent.tensors };
    const sourceId = nativeStateContentId(source.metadata, source.tensors);
    const optimized = { ...source, id: sourceId };
    const snapshot = materializeNativeInferenceState(optimized);
    assert.ok(snapshot.receipt);
    verifyNativeInferenceMaterialization(optimized, snapshot.artifact, snapshot.receipt);
    assert.notEqual(snapshot.artifact.id, sourceId);
    assert.deepEqual(snapshot.artifact.tensors, optimized.tensors);
    const expectedMetadata = { ...optimized.metadata }; delete expectedMetadata.derivation;
    assert.deepEqual(snapshot.artifact.metadata, expectedMetadata);
    assert.equal(snapshot.receipt.sourceState, sourceId);
    assert.equal(snapshot.receipt.inferenceState, snapshot.artifact.id);
    const payload = createHash('sha256');
    for (const [name, tensor] of Object.entries(optimized.tensors).sort(([a], [b]) => a.localeCompare(b)))
      payload.update(name).update('\0').update(tensor.dtype).update('\0').update(JSON.stringify(tensor.shape)).update('\0').update(tensor.data);
    assert.equal(snapshot.receipt.payloadSha256, payload.digest('hex'), 'receipt binds every current numeric tensor');
    assert.throws(() => verifyNativeInferenceMaterialization(optimized, snapshot.artifact,
      { ...snapshot.receipt, inferenceState: `nzs1_${'a'.repeat(52)}` }), /receipt/);
    const changed = { ...snapshot.artifact, tensors: { ...snapshot.artifact.tensors,
      embeddings: { ...snapshot.artifact.tensors.embeddings, data: snapshot.artifact.tensors.embeddings.data.slice() } } };
    changed.tensors.embeddings.data[0] ^= 1;
    changed.id = nativeStateContentId(changed.metadata, changed.tensors);
    assert.throws(() => verifyNativeInferenceMaterialization(optimized, changed, snapshot.receipt), /receipt/);
    assert.deepEqual(optimized.metadata.derivation, derivation, 'source training history remains unchanged');
    assert.throws(() => materializeNativeInferenceState({ ...optimized, id: `nzs1_${'b'.repeat(52)}` }), /source state ID does not match/);

    writeFileSync(join(directory, 'states', `${sourceId}.nzs`), encodeNativeState(optimized));
    writeFileSync(join(directory, 'states', `${snapshot.artifact.id}.nzs`), encodeNativeState(snapshot.artifact));
    const closure = nativeReleaseObjectClosure([directory], [snapshot.artifact.id]);
    assert.deepEqual([...closure.keys()], [snapshot.artifact.id], 'serving snapshot does not pull optimizer ancestry');
    const remote = new MemoryNeuraleseStore();
    assert.deepEqual(await uploadRelease(remote, directory, [snapshot.artifact.id], [snapshot.receipt]), new Set([snapshot.artifact.id]));
    const served = await remote.getNativeState(snapshot.artifact.id);
    assert.ok(served, 'normal release upload ingests the snapshot without training ancestry');
    assert.deepEqual(served.tensors.positions, optimized.tensors.positions);
    assert.deepEqual(served.tensors.embeddings, optimized.tensors.embeddings);
    assert.throws(() => verifyNativeInferenceSnapshotReceipt(snapshot.artifact,
      { ...snapshot.receipt, sourceState: `nzs1_${'b'.repeat(52)}` }), /does not bind source identity/);
    await assert.rejects(uploadRelease(new MemoryNeuraleseStore(), directory, [snapshot.artifact.id],
      [{ ...snapshot.receipt, sourceState: `nzs1_${'b'.repeat(52)}` }]), /does not bind source identity/);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test('native closure publication atomically repairs a corrupt destination from the verified source', () => {
  const root = mkdtempSync(join(tmpdir(), 'tune-release-atomic-copy-'));
  try {
    const sourceDir = join(root, 'source'), destination = join(root, 'destination');
    mkdirSync(join(sourceDir, 'states'), { recursive: true });
    const state = nativeStateFixture(), id = nativeStateContentId(state.metadata, state.tensors), bytes = encodeNativeState(state);
    writeFileSync(join(sourceDir, 'states', `${id}.nzs`), bytes);
    mkdirSync(join(destination, 'states'), { recursive: true });
    const target = join(destination, 'states', `${id}.nzs`);
    writeFileSync(target, bytes.slice(0, Math.floor(bytes.length / 2)));
    assert.deepEqual(copyNativeReleaseClosure([sourceDir], destination, [id]), new Set([id]));
    assert.deepEqual([...readFileSync(target)], [...bytes], 'interrupted publication is replaced with exact verified source bytes');
    installNativeReleaseObject(target, id, bytes);
    assert.deepEqual([...readFileSync(target)], [...bytes], 'retry accepts the now-valid immutable target');
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('an already completed release gets a separately sealed compact serving successor', () => {
  const root = mkdtempSync(join(tmpdir(), 'tune-release-materialize-existing-'));
  try {
    const parent = nativeStateFixture(), identity = Object.fromEntries(['checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture']
      .map(key => [key, parent.metadata[key]]));
    const pid = nativeStateContentId(parent.metadata, parent.tensors), tensorId = `nnt1_${'a'.repeat(52)}`;
    const derivation = { schema: 'natlang.native-state-derivation/1', kind: 'optimizer', origin_state_ids: [pid], parent_state_id: pid,
      gradient_id: tensorId, update_artifact_id: tensorId, program_id: null, optimizer_state_parent_id: null, step: 1,
      optimizer: { name: 'adam', learning_rate_hex: '0x1.0000000000000p-3', weight_decay_hex: '0x0.0p+0',
        beta1_hex: '0x1.999999999999ap-1', beta2_hex: '0x1.fae147ae147aep-1', eps_hex: '0x1.5798ee2308c3ap-27', momentum_hex: '0x0.0p+0' },
      updated_tensor_names: ['layers.0.raw_k'], arithmetic: { schema: 'natlang.native-state-optimizer-torch-cpu/1', cpu_capability: 'DEFAULT',
        float32_subnormal_hex: '0x1.0000000000000p-149', implementation_sha256: '4'.repeat(64), libc_version: 'glibc/test', machine: 'test',
        python_version: '3.12.3', torch_build_sha256: '5'.repeat(64), torch_git_version: 'test', torch_version: '2.11.0' } };
    const source = { metadata: { ...parent.metadata, derivation }, tensors: parent.tensors };
    const state = nativeStateContentId(source.metadata, source.tensors), binding = 'binding:test';
    const key = Buffer.from('natlang.state.id'), keyLength = Buffer.alloc(8), descriptor = Buffer.alloc(12);
    keyLength.writeBigUInt64LE(BigInt(key.length)); descriptor.writeUInt32LE(8); descriptor.writeBigUInt64LE(BigInt(state.length), 4);
    const header = Buffer.alloc(24); header.write('GGUF'); header.writeUInt32LE(3, 4); header.writeBigUInt64LE(1n, 16);
    const gguf = Buffer.concat([header, keyLength, key, descriptor, Buffer.from(state), Buffer.from('payload')]);
    const ggufSha = createHash('sha256').update(gguf).digest('hex');
    const manifest = sealManifest({ schema: 'natlang.adapters/v1', program: { id: 'test', buildHash: 'b'.repeat(64) }, base: identity,
      tree: [{ id: 'tree:', level: 'tree', path: '', parent: null }, { id: 'lambda:test', level: 'lambda', path: 'test.nl', parent: 'tree:' }],
      bindings: [{ id: binding, component: 'test.nl', definitionId: null, contextId: null, origin: 'named',
        ancestry: ['tree:', 'lambda:test'], semanticHash: 's'.repeat(64) }], artifacts: [], families: [], memberships: [],
      softStates: [{ binding, kind: 'instructions', target: null, state, textSha256: 'e'.repeat(64), tokens: 1,
        exports: { gguf: { file: 'exports/state.gguf', sha256: ggufSha, baseGgufSha256: 'f'.repeat(64) } } }], scales: {},
      validatedReductions: [], fallback: 'base', receipts: { training: { fit: 'full-history' } } });
    storeRelease(root, manifest);
    const sourceDir = join(root, '.natlang', 'tune', 'releases', manifest.digest);
    mkdirSync(join(sourceDir, 'states'), { recursive: true });
    mkdirSync(join(sourceDir, 'exports'), { recursive: true });
    writeFileSync(join(sourceDir, 'exports/state.gguf'), gguf);
    writeFileSync(join(sourceDir, 'states', `${state}.nzs`), encodeNativeState({ ...source, id: state }));
    const compact = materializeExistingInferenceRelease(root, manifest, sourceDir);
    assert.notEqual(compact.manifest.digest, manifest.digest);
    assert.notEqual(compact.manifest.softStates[0].state, state);
    assert.equal(compact.manifest.receipts.training.fit, 'full-history');
    assert.equal(manifest.softStates[0].state, state, 'source release manifest remains immutable');
    assert.ok(existsSync(join(sourceDir, 'states', `${state}.nzs`)), 'full source state remains available');
    assert.ok(existsSync(join(compact.directory, 'states', `${compact.manifest.softStates[0].state}.nzs`)));
    assert.equal(compact.manifest.receipts.inferenceStateMaterializations[0].sourceState, state);
    const compactGguf = readFileSync(join(compact.directory, 'exports/state.gguf'));
    assert.ok(compactGguf.includes(Buffer.from(compact.manifest.softStates[0].state)));
    assert.equal(createHash('sha256').update(compactGguf).digest('hex'), compact.manifest.softStates[0].exports.gguf.sha256);
    assert.equal(compact.manifest.receipts.trainingHistoryRelease, manifest.digest);
    unlinkSync(join(compact.directory, 'states', `${compact.manifest.softStates[0].state}.nzs`));
    assert.throws(() => materializeExistingInferenceRelease(root, compact.manifest, compact.directory), /absent/);
    const repaired = materializeExistingInferenceRelease(root, manifest, sourceDir);
    assert.equal(repaired.manifest.digest, compact.manifest.digest);
    assert.ok(existsSync(join(compact.directory, 'states', `${compact.manifest.softStates[0].state}.nzs`)));
  } finally { rmSync(root, { recursive: true, force: true }); }
});
