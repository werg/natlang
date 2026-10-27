/**
 * Compact serving releases (tune/pipeline.ts writes them at activation): a completed release's soft states materialized
 * as inference states, sealed as their own release with the training history's digest. Shared by the tuner and every
 * runtime that serves an active release, so it sits below both.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { sealManifest, type SoftState, type WeightManifest } from './manifest.js';
import { releaseFile, storeRelease } from './release.js';
import { copyNativeReleaseClosure, installNativeReleaseObject, installReleaseExport, materializeNativeInferenceState, materializeNativeStateGguf,
  nativeReleaseObjectClosure, verifyNativeInferenceMaterialization, verifyNativeInferenceSnapshotReceipt,
  type NativeInferenceMaterializationReceipt } from '../../native/release-artifact-closure.js';
import { decodeNativeState, encodeNativeState } from '../../native/neuralese-store.js';

export function remapSoftStateExport(state: SoftState, targetId: string, sourceDirectory: string,
    outputs: { file: string; bytes: Uint8Array; sha256: string }[]): SoftState {
  if (targetId === state.state || !state.exports?.gguf) return { ...state, state: targetId };
  const exportInfo = state.exports.gguf;
  const result = materializeNativeStateGguf(readFileSync(join(sourceDirectory, exportInfo.file)), exportInfo.sha256, state.state, targetId);
  outputs.push({ file: exportInfo.file, ...result });
  return { ...state, state: targetId, exports: { ...state.exports, gguf: { ...exportInfo, sha256: result.sha256 } } };
}

/** Derive a compact, separately sealed serving release from an already completed release. */
export function materializeExistingInferenceRelease(root: string, sourceManifest: WeightManifest, sourceDirectory: string):
  { readonly manifest: WeightManifest; readonly directory: string } {
  const compacted = sourceManifest.softStates.map(state => {
    const path = join(sourceDirectory, 'states', `${state.state}.nzs`);
    if (!existsSync(path)) throw new Error(`native state ${state.state} is absent from source release ${sourceManifest.digest}`);
    const source = decodeNativeState(new Uint8Array(readFileSync(path)));
    if (source.id !== state.state) throw new Error(`native state ${path} does not hold its named content (${source.id})`);
    const result = materializeNativeInferenceState(source);
    if (result.receipt) verifyNativeInferenceMaterialization(source, result.artifact, result.receipt);
    return { state, source, ...result };
  });
  const replacements = new Map(compacted.map(item => [item.state.state, item.artifact.id]));
  const prior = sourceManifest.receipts.inferenceStateMaterializations;
  if (prior !== undefined) {
    if (!Array.isArray(prior)) throw new Error('release inference-state materialization receipts must be an array');
    for (const value of prior) {
      if (!value || typeof value !== 'object' || typeof (value as NativeInferenceMaterializationReceipt).inferenceState !== 'string')
        throw new Error('release inference-state materialization receipt is malformed');
      const receipt = value as NativeInferenceMaterializationReceipt;
      const source = compacted.find(item => item.state.state === receipt.inferenceState)?.source;
      if (!source) throw new Error(`release materialization snapshot ${receipt.inferenceState} is absent`);
      verifyNativeInferenceSnapshotReceipt(source, receipt);
    }
  }
  const receipts = (Array.isArray(prior) ? prior : []).filter((item): item is Record<string, unknown> => !!item && typeof item === 'object' &&
    typeof (item as Record<string, unknown>).inferenceState === 'string' && replacements.has((item as Record<string, unknown>).inferenceState as string));
  for (const item of compacted) if (item.receipt) receipts.push({ ...item.receipt, sourceRelease: sourceManifest.digest });
  const changed = compacted.some(item => item.state.state !== item.artifact.id);
  if (!changed) {
    nativeReleaseObjectClosure([sourceDirectory], [...sourceManifest.artifacts.map(item => item.artifact),
      ...sourceManifest.families.map(item => item.artifact), ...sourceManifest.softStates.map(item => item.state)]);
    for (const item of [...sourceManifest.artifacts, ...sourceManifest.families, ...sourceManifest.softStates]) if (item.exports?.gguf) {
      const bytes = readFileSync(join(sourceDirectory, item.exports.gguf.file));
      if (createHash('sha256').update(bytes).digest('hex') !== item.exports.gguf.sha256)
        throw new Error(`release export ${item.exports.gguf.file} does not match its receipt`);
    }
    return { manifest: sourceManifest, directory: sourceDirectory };
  }
  const exports: { file: string; bytes: Uint8Array; sha256: string }[] = [];
  const { digest: _digest, ...unsigned } = sourceManifest;
  const manifest = sealManifest({ ...unsigned, softStates: sourceManifest.softStates.map(item =>
    remapSoftStateExport(item, replacements.get(item.state)!, sourceDirectory, exports)), receipts: { ...sourceManifest.receipts, inferenceStateMaterializations: receipts,
      trainingHistoryRelease: sourceManifest.digest } });
  const directory = dirname(releaseFile(root, manifest.digest));
  for (const item of exports) installReleaseExport(join(directory, item.file), item.bytes, item.sha256);
  for (const item of compacted) if (item.receipt) {
    const target = join(directory, 'states', `${item.artifact.id}.nzs`);
    mkdirSync(dirname(target), { recursive: true });
    installNativeReleaseObject(target, item.artifact.id, encodeNativeState(item.artifact));
  }
  const roots = [...manifest.artifacts.map(item => item.artifact), ...manifest.families.map(item => item.artifact),
    ...manifest.softStates.map(item => item.state)];
  copyNativeReleaseClosure([sourceDirectory, directory], directory, roots);
  for (const item of [...manifest.artifacts, ...manifest.families, ...manifest.softStates]) if (item.exports?.gguf) {
    const target = join(directory, item.exports.gguf.file), source = join(sourceDirectory, item.exports.gguf.file);
    if (!exports.some(value => value.file === item.exports!.gguf!.file))
      installReleaseExport(target, readFileSync(source), item.exports.gguf.sha256);
  }
  storeRelease(root, manifest);
  return { manifest, directory };
}

