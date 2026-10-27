import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { uploadRelease } from '../dist/cli/weights.js';
import { MemoryNeuraleseStore, nativeStateContentId, encodeNativeState } from '../dist/native/neuralese-store.js';
import { nativeStateFixture } from './fixtures/native-state.mjs';

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
    assert.deepEqual(await uploadRelease(remote, directory), new Set([childId, parentId]));
    assert.deepEqual(uploads, [parentId, childId]);
    await uploadRelease(remote, directory);
    assert.equal(uploads.length, 2, 'already verified objects require no repeated upload');
    writeFileSync(join(directory, 'states', `${childId}.nzs`), encodeNativeState(parent));
    await assert.rejects(uploadRelease(remote, directory), /does not hold its named content/);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});
