import { BrowserNatlangHost, type BrowserRunRequest } from './host.js';
import { BrowserLocalModel, type BrowserModelLoadOptions, type BrowserModelDiagnostics } from './local-model.js';
import { TypeScriptEnvironment } from './environment.js';
import { loadBrowserModelCatalog } from './models.js';

export type BrowserModelSource =
  | { kind: 'url'; url: string; id?: string; templateUrl?: string; chatTemplate?: string }
  | { kind: 'files'; files: Blob[]; id?: string; templateUrl?: string; chatTemplate?: string }
  | { kind: 'huggingface'; repo: string; file?: string; quant?: string; id?: string;
    templateUrl?: string; chatTemplate?: string };

export type BrowserClientLoadOptions = BrowserModelLoadOptions & {
  /** Try CPU only when automatic full-GPU loading fails. */
  cpuFallback?: boolean;
};

export type BrowserClientModelStatus = { id: string; diagnostics: BrowserModelDiagnostics;
  loadMs: number | null; gpuFallbackReason: string | null };

export type BrowserClientOptions = { host?: object; mode?: 'fresh' | 'retained';
  environment?: TypeScriptEnvironment;
  /** Override asset URLs when a bundler does not keep WASM beside the browser entrypoint. */
  wasmUrl?: string; compatWasmUrl?: string; compatWorkerUrl?: string;
  firefoxGpuCompatibility?: boolean; allowOffline?: boolean;
  modelFactory?: () => BrowserLocalModel };

export type BrowserClientRun = Awaited<ReturnType<BrowserNatlangHost['run']>> & {
  model: (BrowserClientModelStatus & { turns: BrowserLocalModel['turnHistory'] }) | null;
};

/** Reusable browser lifecycle for GGUF loading and native natlang execution. */
export class BrowserNatlangClient {
  private current: BrowserLocalModel | null = null;
  private statusValue: BrowserClientModelStatus | null = null;
  private loading = false;
  private running = false;
  private closed = false;
  private readonly factory: () => BrowserLocalModel;
  private readonly hostObject?: object;
  private readonly mode?: 'fresh' | 'retained';
  private readonly environment?: TypeScriptEnvironment;
  private readonly ownsEnvironment: boolean;

  constructor(options: BrowserClientOptions = {}) {
    this.hostObject = options.host;
    this.mode = options.mode;
    this.ownsEnvironment = !options.environment && options.mode === 'retained';
    this.environment = options.environment ?? (this.ownsEnvironment ?
      new TypeScriptEnvironment({ host: options.host, mode: 'retained' }) : undefined);
    this.factory = options.modelFactory ?? (() => new BrowserLocalModel({
      wasmUrl: options.wasmUrl, compatWasmUrl: options.compatWasmUrl,
      compatWorkerUrl: options.compatWorkerUrl,
      firefoxGpuCompatibility: options.firefoxGpuCompatibility,
      allowOffline: options.allowOffline }));
  }

  get model(): BrowserLocalModel | null { return this.current; }
  get modelStatus(): BrowserClientModelStatus | null { return this.statusValue; }
  get busy(): boolean { return this.loading || this.running; }

  private ensureIdle(): void {
    if (this.closed) throw new Error('browser natlang client is closed');
    if (this.busy) throw new Error('browser natlang client is busy');
  }

  async loadModel(source: BrowserModelSource,
    options: BrowserClientLoadOptions = {}): Promise<BrowserClientModelStatus> {
    this.ensureIdle();
    this.loading = true;
    try {
      if (source.kind === 'files' && !source.files.length) throw new Error('select at least one GGUF file');
      const template = source.chatTemplate ?? (source.templateUrl ? await fetch(source.templateUrl)
        .then(response => {
          if (!response.ok) throw new Error(`model template unavailable: ${source.templateUrl}`);
          return response.text();
        }) : undefined);
      const { cpuFallback = true, ...given } = options;
      const loadOptions: BrowserModelLoadOptions = {
        ...given, ...(template ? { chatTemplate: template } : {}),
        ...(globalThis.crossOriginIsolated ? {} : { threads: given.threads ?? 1 }),
      };
      if (!Number.isInteger(loadOptions.contextTokens ?? 4096) || (loadOptions.contextTokens ?? 4096) < 512)
        throw new RangeError('contextTokens must be an integer of at least 512');
      await this.current?.close();
      this.current = null; this.statusValue = null;
      const load = async (override: Partial<BrowserModelLoadOptions> = {}) => {
        const candidate = this.factory();
        try {
          const params = { ...loadOptions, ...override };
          if (source.kind === 'url') await candidate.loadFromUrl(source.url, params);
          else if (source.kind === 'files') await candidate.loadFiles(source.files, params);
          else await candidate.loadFromHuggingFace({ repo: source.repo, file: source.file,
            quant: source.quant }, params);
          return { model: candidate, fallbackReason: null };
        } catch (error) {
          const gpuAttempted = candidate.diagnostics.requestedGpuLayers === 99999;
          try { await candidate.close(); } catch { /* preserve the load error */ }
          if (cpuFallback && loadOptions.gpuLayers === undefined && gpuAttempted &&
              override.gpuLayers === undefined) return { model: null, fallbackReason: String(error) };
          throw error;
        }
      };
      let fallbackReason: string | null = null;
      const first = await load();
      if (first.model) this.current = first.model;
      else { fallbackReason = first.fallbackReason;
        const second = await load({ gpuLayers: 0 });
        if (!second.model) throw new Error('CPU model load failed');
        this.current = second.model;
      }
      const id = source.id ?? (source.kind === 'url' ? source.url :
        source.kind === 'files' ? source.files.map(file => (file as File).name ?? 'GGUF').join(',') :
          `${source.repo}/${source.file ?? source.quant ?? ''}`);
      this.statusValue = { id, diagnostics: this.current.diagnostics,
        loadMs: this.current.lastLoadMs, gpuFallbackReason: fallbackReason };
      return this.statusValue;
    } finally { this.loading = false; }
  }

  /** Resolve the currently published default instead of baking a checkpoint into an app. */
  async loadDefaultModel(options: BrowserClientLoadOptions = {},
    catalogUrl?: string): Promise<BrowserClientModelStatus> {
    const catalog = await loadBrowserModelCatalog(catalogUrl);
    const selected = catalog.models.find(model => model.id === catalog.defaultId)!;
    return this.loadModel({ kind: 'url', id: selected.id, url: selected.url,
      templateUrl: selected.templateUrl },
    { contextTokens: selected.contextTokens, ...options });
  }

  async unloadModel(): Promise<void> {
    this.ensureIdle();
    await this.current?.close(); this.current = null; this.statusValue = null;
  }

  async run(request: BrowserRunRequest): Promise<BrowserClientRun> {
    this.ensureIdle();
    this.running = true;
    const model = this.current, firstTurn = model?.turnHistory.length ?? 0;
    let host: BrowserNatlangHost | null = null;
    try {
      host = new BrowserNatlangHost({ model: model ?? undefined, host: this.hostObject,
        mode: this.mode, environment: this.environment });
      const result = await host.run(request);
      return { ...result, model: model && this.statusValue ? {
        ...this.statusValue, turns: structuredClone(model.turnHistory.slice(firstTurn)),
      } : null };
    } finally { host?.close(); this.running = false; }
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.ensureIdle();
    await this.current?.close(); this.current = null; this.statusValue = null;
    if (this.ownsEnvironment) this.environment?.close();
    this.closed = true;
  }
}
