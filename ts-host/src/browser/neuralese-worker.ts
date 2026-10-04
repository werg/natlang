/**
 * The Web Worker side of `startBrowserNeuralese` (neuralese-wasm.ts): loads the WebAssembly service, mounts the model
 * and heads Blobs with WORKERFS (read in place, not copied), and answers requests one at a time.
 */
import { NeuraleseWasmService, type NeuraleseWasmFactory } from './neuralese-wasm.js';

let service: NeuraleseWasmService | undefined;
const scope = self as unknown as { onmessage: ((event: MessageEvent) => void) | null; postMessage(message: unknown, transfer?: Transferable[]): void };

scope.onmessage = async (event: MessageEvent) => {
  const { id, kind, ...data } = event.data as { id: number; kind: string } & Record<string, any>;
  try {
    if (kind === 'load') {
      const factory = (await import(/* @vite-ignore */ data.moduleUrl)).default as NeuraleseWasmFactory;
      const module = await factory();
      module.FS.mkdir('/models');
      module.FS.mount(module.WORKERFS, { blobs: [{ name: 'model.gguf', data: data.model }, { name: 'heads.gguf', data: data.heads }] }, '/models');
      service = new NeuraleseWasmService(module);
      scope.postMessage({ id, ok: true, value: service.load('/models/model.gguf', '/models/heads.gguf', data.options) });
    } else if (kind === 'request') {
      if (!service) throw new Error('neuralese worker: load first');
      const result = service.handle(data.method, data.path, data.body);
      scope.postMessage({ id, ok: true, value: result }, [result.body.buffer as ArrayBuffer]);
    } else if (kind === 'unload') {
      service?.unload();
      service = undefined;
      scope.postMessage({ id, ok: true });
    }
  } catch (error) {
    scope.postMessage({ id, ok: false, error: error instanceof Error ? error.message : String(error) });
  }
};
