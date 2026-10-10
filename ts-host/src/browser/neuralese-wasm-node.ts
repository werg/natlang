/** Node: the WebAssembly Neuralese service in process, with the model files' directories mounted (NODEFS). */
import { basename, dirname } from 'node:path';
import { MemoryNeuraleseStore, type NeuraleseStore } from '../native/neuralese-store.js';
import { NeuraleseWasmService, serveLocally, type NeuraleseWasmFactory, type NeuraleseWasmOptions, type StartedNeuralese } from './neuralese-wasm.js';

/**
 * Node: load the module, mount the files' directories and serve `endpoint` (default `http://neuralese.local`). `store`
 * is the block archive restore uploads from (default: in memory).
 */
export async function startNodeNeuralese(options: NeuraleseWasmOptions & { factory: NeuraleseWasmFactory; model: string; heads: string;
  endpoint?: string; store?: NeuraleseStore }): Promise<StartedNeuralese & { service: NeuraleseWasmService }> {
  // The threaded build (neuralese-wasm-mt) needs its worker pool sized up front.
  const module = await options.factory({ pthreadPoolSize: options.threads ?? 1 });
  const mounted = new Map<string, string>();
  const mount = (file: string) => {
    const dir = dirname(file);
    if (!mounted.has(dir)) {
      const target = `/mnt${mounted.size}`;
      module.FS.mkdir(target);
      module.FS.mount(module.NODEFS, { root: dir }, target);
      mounted.set(dir, target);
    }
    return `${mounted.get(dir)}/${basename(file)}`;
  };
  const service = new NeuraleseWasmService(module);
  const hello = await service.load(mount(options.model), mount(options.heads), options);
  const endpoint = options.endpoint ?? 'http://neuralese.local';
  let chain = Promise.resolve();  // one request at a time
  const stop = serveLocally(endpoint, (method, path, body, onEvent) => {
    const run = chain.then(() => service.handle(method, path, body, onEvent));
    chain = run.then(() => undefined, () => undefined);
    return run;
  });
  return { endpoint, ...hello, service, store: options.store ?? new MemoryNeuraleseStore(), async close() { stop(); await service.unload(); } };
}
