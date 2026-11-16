import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { encodeNativeTensorArtifact, nativeTensorArtifactContentId, encodeNativeState, nativeStateContentId }
  from '../../dist/native/neuralese-store.js';
import { nativeStateFixture } from './native-state.mjs';

/** Small real objects for orchestration tests whose numerical trainer is a stand-in. */
export function writeNativeTuneTensor(directory, base, tag) {
  const metadata = { schema: 'natlang.neuralese-native-tensor-artifact/1', kind: 'parameter-set',
    native_identity: base, source_artifact_id: null, provenance: { fixture: true, tag },
    tensors: [{ name: 'weight', dtype: 'F32', shape: [1] }] };
  const tensors = { weight: { dtype: 'F32', shape: [1], data: Uint8Array.from([0, 0, 128, 63]) } };
  const id = nativeTensorArtifactContentId(metadata, tensors);
  mkdirSync(join(directory, 'artifacts'), { recursive: true });
  writeFileSync(join(directory, 'artifacts', `${id}.nnt`), encodeNativeTensorArtifact({ metadata, tensors }));
  return id;
}

export function nativeTuneState(base) {
  const source = nativeStateFixture(base.architecture);
  const artifact = { ...source, metadata: { ...source.metadata, ...base } };
  const id = nativeStateContentId(artifact.metadata, artifact.tensors);
  return { id, artifact, wire: encodeNativeState(artifact) };
}

/** A soft state as tune/build.py writes it: its store file, served as it is by every engine. */
export function writeNativeTuneState(directory, base, binding) {
  const { id, artifact, wire } = nativeTuneState(base);
  mkdirSync(join(directory, 'states'), { recursive: true });
  writeFileSync(join(directory, 'states', `${id}.nzs`), wire);
  return { binding, kind: 'instructions', target: null, state: id,
    textSha256: createHash('sha256').update(artifact.metadata.text).digest('hex'), tokens: 2 };
}
