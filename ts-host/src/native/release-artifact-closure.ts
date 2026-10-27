/** Exact executable native-object closure for an immutable release. Training receipts stay in the fit output. */
import { createHash, randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync, renameSync, unlinkSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { decodeNativeState, decodeNativeTensorArtifact, nativeStateArtifactDependencies, nativeTensorArtifactDependencies,
  nativeStateContentId, nativeCanonicalJson, type NativeStateArtifact } from './neuralese-store.js';

export type NativeInferenceMaterializationReceipt = { readonly sourceState: string; readonly inferenceState: string;
  readonly payloadSha256: string; readonly derivation: unknown };

/** Rewrite only the fixed-size GGUF state identity, after checking the original export's receipt. */
export function materializeNativeStateGguf(bytes: Uint8Array, expectedSha256: string, sourceId: string, targetId: string):
    { bytes: Uint8Array; sha256: string } {
  const hash = (value: Uint8Array) => createHash('sha256').update(value).digest('hex');
  if (hash(bytes) !== expectedSha256) throw new Error('native state GGUF differs from its export receipt');
  const source = Buffer.from(sourceId), target = Buffer.from(targetId);
  if (source.length !== target.length || !/^nzs1_[a-z2-7]{52}$/.test(sourceId) || !/^nzs1_[a-z2-7]{52}$/.test(targetId))
    throw new Error('invalid native state GGUF identity');
  const data = Buffer.from(bytes);
  if (data.length < 24 || data.subarray(0, 4).toString('ascii') !== 'GGUF' || data.readUInt32LE(4) !== 3)
    throw new Error('invalid native state GGUF header');
  let offset = 24, found = -1;
  const take = (count: number) => { if (!Number.isSafeInteger(count) || count < 0 || offset + count > data.length)
    throw new Error('invalid native state GGUF metadata'); const at = offset; offset += count; return at; };
  const u64 = () => { const value = data.readBigUInt64LE(take(8));
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('invalid native state GGUF length'); return Number(value); };
  const string = () => { const size = u64(), at = take(size); return { at, size, value: data.subarray(at, at + size).toString('utf8') }; };
  const skip = (type: number): void => {
    const widths: Record<number, number> = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
    if (type === 8) { string(); return; }
    if (type === 9) { const element = data.readUInt32LE(take(4)), count = u64();
      if (element === 9 || count > data.length) throw new Error('invalid native state GGUF array');
      for (let i = 0; i < count; i++) skip(element); return; }
    if (!(type in widths)) throw new Error('invalid native state GGUF value type');
    take(widths[type]!);
  };
  const countBig = data.readBigUInt64LE(16);
  if (countBig > BigInt(data.length)) throw new Error('invalid native state GGUF metadata count');
  for (let i = 0; i < Number(countBig); i++) {
    const key = string().value, type = data.readUInt32LE(take(4));
    if (key === 'natlang.state.id') {
      if (type !== 8 || found !== -1) throw new Error('native state GGUF identity is missing or ambiguous');
      const value = string();
      if (value.value !== sourceId || value.size !== source.length) throw new Error('native state GGUF identity is missing or ambiguous');
      found = value.at;
    } else skip(type);
  }
  if (found === -1) throw new Error('native state GGUF identity is missing or ambiguous');
  target.copy(data, found);
  return { bytes: data, sha256: hash(data) };
}

/** Install an export atomically and repair a corrupt existing copy. */
export function installReleaseExport(target: string, bytes: Uint8Array, expectedSha256: string): void {
  const hash = (value: Uint8Array) => createHash('sha256').update(value).digest('hex');
  if (hash(bytes) !== expectedSha256) throw new Error('release export bytes do not match their receipt');
  if (existsSync(target) && hash(readFileSync(target)) === expectedSha256) return;
  mkdirSync(dirname(target), { recursive: true });
  const temporary = `${target}.${process.pid}.${randomBytes(8).toString('hex')}.tmp`;
  try {
    writeFileSync(temporary, bytes, { flag: 'wx' });
    if (hash(readFileSync(temporary)) !== expectedSha256) throw new Error('release export temporary file differs from receipt');
    renameSync(temporary, target);
  } catch (error) { try { unlinkSync(temporary); } catch { /* already renamed */ } throw error; }
}

function nativeStatePayloadSha256(state: NativeStateArtifact): string {
  const payload = createHash('sha256');
  for (const [name, tensor] of Object.entries(state.tensors).sort(([a], [b]) => a.localeCompare(b)))
    payload.update(name).update('\0').update(tensor.dtype).update('\0').update(JSON.stringify(tensor.shape)).update('\0').update(tensor.data);
  return payload.digest('hex');
}

/** Make a serving snapshot of an optimized state. Its current tensors are complete; update ancestry stays with training. */
export function materializeNativeInferenceState(source: NativeStateArtifact): {
  readonly artifact: NativeStateArtifact;
  readonly receipt: NativeInferenceMaterializationReceipt | null;
} {
  if (nativeStateContentId(source.metadata, source.tensors) !== source.id)
    throw new Error(`native inference source state ID does not match its content (${source.id})`);
  const derivation = source.metadata.derivation;
  if (!derivation || derivation.kind !== 'optimizer') return { artifact: source, receipt: null };
  const { derivation: _history, ...metadata } = source.metadata;
  const artifact = { metadata, tensors: source.tensors, id: nativeStateContentId(metadata, source.tensors) };
  return { artifact, receipt: { sourceState: source.id, inferenceState: artifact.id, payloadSha256: nativeStatePayloadSha256(source), derivation } };
}

/** Recheck that a release snapshot changes only optimizer-history metadata and its content ID. */
export function verifyNativeInferenceMaterialization(source: NativeStateArtifact, snapshot: NativeStateArtifact,
    receipt: NativeInferenceMaterializationReceipt): void {
  const expected = materializeNativeInferenceState(source);
  if (!expected.receipt || receipt.sourceState !== source.id || receipt.inferenceState !== snapshot.id ||
      receipt.inferenceState !== expected.artifact.id || nativeCanonicalJson(receipt.derivation) !== nativeCanonicalJson(source.metadata.derivation) ||
      nativeCanonicalJson(snapshot.metadata) !== nativeCanonicalJson(expected.artifact.metadata) ||
      nativeStateContentId(snapshot.metadata, snapshot.tensors) !== snapshot.id ||
      nativeCanonicalJson(Object.fromEntries(Object.entries(snapshot.tensors).map(([name, tensor]) =>
        [name, { dtype: tensor.dtype, shape: tensor.shape }]))) !== nativeCanonicalJson(Object.fromEntries(Object.entries(expected.artifact.tensors).map(([name, tensor]) =>
        [name, { dtype: tensor.dtype, shape: tensor.shape }]))) ||
      Object.entries(snapshot.tensors).some(([name, tensor]) => !expected.artifact.tensors[name] ||
        !tensor.data.every((byte, index) => byte === expected.artifact.tensors[name]!.data[index])) ||
      receipt.payloadSha256 !== nativeStatePayloadSha256(snapshot))
    throw new Error(`native inference materialization receipt does not match exact source state ${source.id}`);
}

/** Verify a transported snapshot from its retained derivation receipt, without shipping ancestor tensors. */
export function verifyNativeInferenceSnapshotReceipt(snapshot: NativeStateArtifact,
    receipt: NativeInferenceMaterializationReceipt): void {
  const derivation = receipt.derivation;
  if (!derivation || typeof derivation !== 'object' || (derivation as { kind?: unknown }).kind !== 'optimizer' ||
      receipt.inferenceState !== snapshot.id || nativeStateContentId(snapshot.metadata, snapshot.tensors) !== snapshot.id ||
      'derivation' in snapshot.metadata ||
      nativeStateContentId({ ...snapshot.metadata, derivation } as NativeStateArtifact['metadata'], snapshot.tensors) !== receipt.sourceState ||
      receipt.payloadSha256 !== nativeStatePayloadSha256(snapshot))
    throw new Error(`native inference snapshot receipt does not bind source identity and payload for ${snapshot.id}`);
}

export function nativeReleaseObjectClosure(directories: readonly string[], roots: Iterable<string>): Map<string, string> {
  const files = new Map<string, string>();
  for (const directory of directories) {
    for (const [folder, extension] of [['artifacts', '.nnt'], ['states', '.nzs']] as const) {
      const path = join(directory, folder);
      if (!existsSync(path)) continue;
      // The caller provides exact roots. Index names only; decode bytes only when reached.
      for (const file of readdirSync(path)) {
        // An interrupted atomic install may leave its private temporary file.
        // It is never a release object and must not prevent a retry.
        if (/^(?:nzs1_|nnt1_)[a-z2-7]{52}\.(?:nzs|nnt)\.\d+\.[a-f0-9]{16}\.tmp$/.test(file)) continue;
        if (!file.endsWith(extension)) throw new Error(`unexpected native release file ${folder}/${file}`);
        const id = file.slice(0, -extension.length), existing = files.get(id);
        if (!existing) files.set(id, join(path, file));
      }
    }
  }
  const closure = new Map<string, string>(), visiting = new Set<string>();
  // An iterative post-order walk avoids retaining every decoded object on the JS stack for long histories.
  const pending = [...new Set(roots)].reverse().map(id => ({ id, exit: false }));
  while (pending.length) {
    const item = pending.pop()!;
    if (closure.has(item.id)) continue;
    if (item.exit) {
      visiting.delete(item.id);
      closure.set(item.id, files.get(item.id)!);
      continue;
    }
    if (visiting.has(item.id)) throw new Error(`native release dependency cycle at ${item.id}`);
    const path = files.get(item.id);
    if (!path) throw new Error(`native release is missing dependency ${item.id}`);
    visiting.add(item.id);
    const bytes = new Uint8Array(readFileSync(path));
    const artifact = item.id.startsWith('nzs1_') ? decodeNativeState(bytes)
      : item.id.startsWith('nnt1_') ? decodeNativeTensorArtifact(bytes)
      : (() => { throw new Error(`invalid native release object ID ${item.id}`); })();
    if (artifact.id !== item.id) throw new Error(`release file ${path} does not hold its named content (${artifact.id})`);
    const dependencies = item.id.startsWith('nzs1_') ? nativeStateArtifactDependencies(artifact as ReturnType<typeof decodeNativeState>)
      : nativeTensorArtifactDependencies(artifact as ReturnType<typeof decodeNativeTensorArtifact>);
    pending.push({ id: item.id, exit: true });
    for (const dependency of dependencies) pending.push({ id: dependency, exit: false });
  }
  return closure;
}

/** Install verified native bytes atomically; replace a corrupt/interrupted target from the verified caller source. */
export function installNativeReleaseObject(target: string, id: string, bytes: Uint8Array): void {
  const decode = (value: Uint8Array) => id.startsWith('nzs1_') ? decodeNativeState(value)
    : id.startsWith('nnt1_') ? decodeNativeTensorArtifact(value) : undefined;
  const verify = (value: Uint8Array, path: string) => {
    const artifact = decode(value);
    if (!artifact || artifact.id !== id) throw new Error(`release object ${path} does not hold its named content (${artifact?.id ?? 'invalid bytes'})`);
  };
  verify(bytes, 'source');
  if (existsSync(target)) {
    try { verify(new Uint8Array(readFileSync(target)), target); return; } catch { /* overwrite an interrupted or corrupt destination */ }
  }
  mkdirSync(dirname(target), { recursive: true });
  const temporary = `${target}.${process.pid}.${randomBytes(8).toString('hex')}.tmp`;
  try {
    writeFileSync(temporary, bytes, { flag: 'wx' });
    verify(new Uint8Array(readFileSync(temporary)), temporary);
    renameSync(temporary, target);
    verify(new Uint8Array(readFileSync(target)), target);
  } catch (error) {
    try { unlinkSync(temporary); } catch { /* absent after a successful rename */ }
    throw error;
  }
}

export function copyNativeReleaseClosure(directories: readonly string[], destination: string, roots: Iterable<string>): Set<string> {
  const closure = nativeReleaseObjectClosure(directories, roots);
  for (const [id, source] of closure) {
    const folder = id.startsWith('nzs1_') ? 'states' : 'artifacts';
    const target = join(destination, folder, `${id}.${folder === 'states' ? 'nzs' : 'nnt'}`);
    installNativeReleaseObject(target, id, new Uint8Array(readFileSync(source)));
  }
  return new Set(closure.keys());
}
