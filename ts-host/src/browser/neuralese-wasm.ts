/**
 * The Neuralese service as WebAssembly: the llama.cpp fork's server engine (tools/neuralese/neuralese-service.cpp,
 * built by tools/neuralese/wasm/build.sh) running in the browser or in Node, answering the same protocol as the
 * reference and fork servers. The runtime reaches it through an in-process endpoint, so `neuraleseServerModelTurn`,
 * `HttpNeuraleseStore`, decision readout, encode, write and digest work unchanged.
 *
 * In a browser the module runs in a Web Worker (computation is synchronous; model files are mounted from Blobs with
 * WORKERFS, without copying them into memory twice): `startBrowserNeuralese`. In Node it runs in process with the
 * files' directories mounted (NODEFS): `startNodeNeuralese`. CPU (SIMD). The threaded build
 * (`neuralese-wasm-mt.mjs`, `threads` > 1) needs cross-origin isolation in browsers (COOP/COEP headers). The WebGPU
 * build (`neuralese-wasm-gpu.mjs`, `gpuLayers` > 0; Chromium, JSPI) returns Promises from load, handle and unload,
 * which is why those are awaited here for every build.
 */
import { registerLocalEndpoint } from '../model/chat-completion.js';

/** The Emscripten module's surface the service uses. */
export type NeuraleseWasmModule = {
  FS: { mkdir(path: string): void; mount(type: unknown, options: unknown, path: string): void; writeFile(path: string, data: Uint8Array): void };
  NODEFS?: unknown; WORKERFS?: unknown; HEAPU8: Uint8Array;
  _nzw_load(options: number): number | Promise<number>; _nzw_error(): number; _nzw_hello(): number;
  _nzw_handle(method: number, path: number, body: number, length: number): number | Promise<number>;
  _nzw_body(): number; _nzw_body_length(): number; _nzw_content_type(): number; _nzw_unload(): void | Promise<void>;
  _malloc(size: number): number; _free(pointer: number): void;
  UTF8ToString(pointer: number): string; stringToUTF8(text: string, pointer: number, max: number): void; lengthBytesUTF8(text: string): number;
};
export type NeuraleseWasmFactory = (options?: Record<string, unknown>) => Promise<NeuraleseWasmModule>;
export type NeuraleseWasmOptions = { nCtx?: number; maxBlock?: number; threads?: number; gpuLayers?: number; dialect?: string; alias?: string };
export type NeuraleseWasmResponse = { status: number; body: Uint8Array; contentType: string };

/** A loaded service: one request at a time. */
export class NeuraleseWasmService {
  constructor(private readonly module: NeuraleseWasmModule) {}

  // wasm32 pointers above 2 GB arrive as negative numbers: read them unsigned before indexing the heap.
  private string(text: string): number {
    const size = this.module.lengthBytesUTF8(text) + 1;
    const pointer = this.module._malloc(size) >>> 0;
    this.module.stringToUTF8(text, pointer, size);
    return pointer;
  }

  /** Load model and heads GGUF files from paths in the module's file system. */
  async load(model: string, heads: string, options: NeuraleseWasmOptions = {}):
    Promise<{ dialect: string; cutoff: number; devices?: NeuraleseDevice[]; gpu_layers?: number }> {
    const config = this.string(JSON.stringify({ model, heads, n_ctx: options.nCtx ?? 8192, max_block: options.maxBlock ?? 64,
      threads: options.threads ?? 1, gpu_layers: options.gpuLayers ?? 0, ...(options.dialect ? { dialect: options.dialect } : {}), ...(options.alias ? { alias: options.alias } : {}) }));
    try {
      if (await this.module._nzw_load(config) !== 0) throw new Error(`neuralese wasm: ${this.module.UTF8ToString(this.module._nzw_error() >>> 0)}`);
    } finally { this.module._free(config); }
    return JSON.parse(this.module.UTF8ToString(this.module._nzw_hello() >>> 0));
  }

  async handle(method: string, path: string, body: Uint8Array = new Uint8Array()): Promise<NeuraleseWasmResponse> {
    const m = this.string(method), p = this.string(path);
    const b = this.module._malloc(Math.max(1, body.length)) >>> 0;
    this.module.HEAPU8.set(body, b);
    try {
      const status = await this.module._nzw_handle(m, p, b, body.length);
      const start = this.module._nzw_body() >>> 0, length = this.module._nzw_body_length();
      return { status, body: this.module.HEAPU8.slice(start, start + length), contentType: this.module.UTF8ToString(this.module._nzw_content_type() >>> 0) };
    } finally { this.module._free(m); this.module._free(p); this.module._free(b); }
  }

  async unload(): Promise<void> { await this.module._nzw_unload(); }
}

async function requestBody(init: RequestInit): Promise<Uint8Array> {
  if (init.body === undefined || init.body === null) return new Uint8Array();
  if (typeof init.body === 'string') return new TextEncoder().encode(init.body);
  if (init.body instanceof Uint8Array) return init.body;
  if (init.body instanceof ArrayBuffer) return new Uint8Array(init.body);
  return new Uint8Array(await new Response(init.body as BodyInit).arrayBuffer());
}

/** Serve `endpoint` in process with `answer` (a service, or a worker bridge). */
function serveLocally(endpoint: string, answer: (method: string, path: string, body: Uint8Array) => Promise<NeuraleseWasmResponse>): () => void {
  const base = endpoint.replace(/\/$/, '');
  return registerLocalEndpoint(base, async (url, init) => {
    const path = new URL(url).pathname;
    const result = await answer((init.method ?? 'GET').toUpperCase(), path, await requestBody(init));
    return new Response(result.status === 204 ? null : result.body as BodyInit,
      { status: result.status, headers: { 'content-type': result.contentType } });
  });
}

/**
 * Which build to run here: the WebGPU build when a GPU adapter offers `shader-f16` (ggml's WebGPU backend needs it;
 * Chromium on Linux often does not offer it, and the build would then run on one CPU thread), else the threaded build
 * when the page is cross-origin isolated, else the single-threaded build.
 */
export async function chooseNeuraleseBuild(scope: { navigator?: any; crossOriginIsolated?: boolean } = globalThis as any,
  options: { gpu?: boolean; maxThreads?: number } = {}): Promise<{ build: 'neuralese-wasm-gpu' | 'neuralese-wasm-mt' | 'neuralese-wasm';
  threads: number; gpuLayers: number; reason: string }> {
  if (options.gpu !== false && scope.navigator?.gpu) {
    try {
      const adapter = await scope.navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter?.features?.has('shader-f16')) return { build: 'neuralese-wasm-gpu', threads: 1, gpuLayers: 999, reason: 'WebGPU adapter with shader-f16' };
    } catch { /* no adapter: CPU */ }
  }
  const threads = scope.crossOriginIsolated ? Math.max(1, Math.min(options.maxThreads ?? 8, scope.navigator?.hardwareConcurrency || 4)) : 1;
  return threads > 1 ? { build: 'neuralese-wasm-mt', threads, gpuLayers: 0, reason: 'cross-origin isolated: threads' }
    : { build: 'neuralese-wasm', threads: 1, gpuLayers: 0, reason: 'not cross-origin isolated: one thread' };
}

export type NeuraleseDevice = { name: string; description: string; gpu: boolean };
export type StartedNeuralese = { endpoint: string; dialect: string; cutoff: number; devices?: NeuraleseDevice[]; gpu_layers?: number;
  close(): Promise<void> };

/** Node: load the module, mount the files' directories and serve `endpoint` (default `http://neuralese.local`). */
export async function startNodeNeuralese(options: NeuraleseWasmOptions & { factory: NeuraleseWasmFactory; model: string; heads: string;
  endpoint?: string }): Promise<StartedNeuralese & { service: NeuraleseWasmService }> {
  const { dirname, basename } = await import('node:path');
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
  const stop = serveLocally(endpoint, (method, path, body) => {
    const run = chain.then(() => service.handle(method, path, body));
    chain = run.then(() => undefined, () => undefined);
    return run;
  });
  return { endpoint, ...hello, service, async close() { stop(); await service.unload(); } };
}

/** Browser: start the service in a Web Worker (`worker`, running `neuralese-worker`) and serve `endpoint`. */
export async function startBrowserNeuralese(options: NeuraleseWasmOptions & { worker: Worker; moduleUrl: string; model: Blob; heads: Blob;
  endpoint?: string }): Promise<StartedNeuralese> {
  const { worker } = options;
  let next = 0;
  const pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
  worker.onmessage = (event: MessageEvent) => {
    const { id, ok, value, error } = event.data as { id: number; ok: boolean; value?: unknown; error?: string };
    const entry = pending.get(id);
    pending.delete(id);
    if (ok) entry?.resolve(value); else entry?.reject(new Error(error));
  };
  const call = <T>(message: Record<string, unknown>, transfer: Transferable[] = []) => new Promise<T>((resolve, reject) => {
    const id = next++;
    pending.set(id, { resolve: resolve as (value: unknown) => void, reject });
    worker.postMessage({ id, ...message }, transfer);
  });
  const { model, heads, worker: _, endpoint: __, moduleUrl, ...rest } = options;
  const hello = await call<{ dialect: string; cutoff: number; devices?: NeuraleseDevice[]; gpu_layers?: number }>({ kind: 'load', moduleUrl, model, heads, options: rest });
  const endpoint = options.endpoint ?? 'http://neuralese.local';
  const stop = serveLocally(endpoint, (method, path, body) =>
    call<NeuraleseWasmResponse>({ kind: 'request', method, path, body }, [body.buffer as ArrayBuffer]));
  return { endpoint, ...hello, async close() { stop(); await call({ kind: 'unload' }); worker.terminate(); } };
}
