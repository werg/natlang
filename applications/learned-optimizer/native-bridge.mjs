// Copy exact native evidence and its dependency closure from the executing task into the optimizer store.
import { pathToFileURL } from 'node:url';
import { join } from 'node:path';

export function nativeBridge(repository, python, plan, out) {
  let modules;
  return async digests => {
    if (!digests?.length) return;
    modules ??= Promise.all([import(pathToFileURL(join(repository, 'ts-host/dist/runtime/context.js')).href),
      import(pathToFileURL(join(repository, 'ts-host/dist/native/neuralese-store.js')).href)]);
    const [{ currentFrame }, native] = await modules;
    const store = currentFrame()?.task.runtime.options.neuralese?.store;
    if (!store) throw new Error('controller evidence needs the executing native state store');
    const done = new Set(), active = new Set(), artifacts = [];
    const visit = async id => {
      if (done.has(id)) return;
      if (active.has(id)) throw new Error('native evidence dependency cycle');
      active.add(id);
      const state = id.startsWith('nzs1_');
      const item = state ? await store.getNativeState(id) : await store.getNativeTensorArtifact(id);
      if (!item || item.id !== id) throw new Error(`native evidence artifact ${id} is unavailable`);
      for (const dependency of state ? native.nativeStateArtifactDependencies(item) : native.nativeTensorArtifactDependencies(item))
        await visit(dependency);
      artifacts.push({ id, bytes: Buffer.from(state ? native.encodeNativeState(item) : native.encodeNativeTensorArtifact(item)).toString('base64') });
      active.delete(id); done.add(id);
    };
    for (const item of digests) await visit(item.digest.$neuralese.id);
    await python.request({ plan, out, op: 'import-artifacts', artifacts });
  };
}
