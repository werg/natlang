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

export function nativeTuneStateExport(base) {
  const source = nativeStateFixture(base.architecture);
  const artifact = { ...source, metadata: { ...source.metadata, ...base } };
  const id = nativeStateContentId(artifact.metadata, artifact.tensors);
  const key = Buffer.from('natlang.state.id'), length = Buffer.alloc(8), descriptor = Buffer.alloc(12);
  length.writeBigUInt64LE(BigInt(key.length)); descriptor.writeUInt32LE(8); descriptor.writeBigUInt64LE(BigInt(id.length), 4);
  const header = Buffer.alloc(24); header.write('GGUF'); header.writeUInt32LE(3, 4); header.writeBigUInt64LE(1n, 16);
  const gguf = Buffer.concat([header, length, key, descriptor, Buffer.from(id), Buffer.from('span')]);
  return { id, artifact, gguf, sha256: createHash('sha256').update(gguf).digest('hex') };
}

export function writeNativeTuneState(directory, base, binding) {
  const { id, artifact, gguf, sha256 } = nativeTuneStateExport(base);
  mkdirSync(join(directory, 'states'), { recursive: true });
  mkdirSync(join(directory, 'exports'), { recursive: true });
  writeFileSync(join(directory, 'states', `${id}.nzs`), encodeNativeState(artifact));
  writeFileSync(join(directory, 'exports', `${id}.state.gguf`), gguf);
  return { binding, kind: 'instructions', target: null, state: id,
    textSha256: createHash('sha256').update(artifact.metadata.text).digest('hex'), tokens: 2,
    exports: { gguf: { file: `exports/${id}.state.gguf`, sha256, baseGgufSha256: 'f'.repeat(64) } } };
}
