/** Browser GPU capability relevant to Wllama's ggml WebGPU backend. */
export type BrowserGpuCapability = {
  apiAvailable: boolean;
  adapterAvailable: boolean;
  shaderF16: boolean;
  browser: 'chromium' | 'firefox' | 'safari' | 'other';
  usable: boolean;
  reason: string;
};

type GpuApi = { requestAdapter(options?: { powerPreference?: 'high-performance' }): Promise<{
  features: { has(name: string): boolean };
} | null> };

function browserName(userAgent: string): BrowserGpuCapability['browser'] {
  if (/Firefox\//.test(userAgent)) return 'firefox';
  if (/Chrome\/|Chromium\/|Edg\//.test(userAgent)) return 'chromium';
  if (/Safari\//.test(userAgent)) return 'safari';
  return 'other';
}

/** Probe the high-performance adapter; this does not prove that Wllama offloaded a layer. */
export async function probeBrowserGpu(options: {
  gpu?: GpuApi | null; userAgent?: string; firefoxCompatibility?: boolean;
} = {}): Promise<BrowserGpuCapability> {
  const browser = browserName(options.userAgent ?? globalThis.navigator?.userAgent ?? '');
  const gpu = options.gpu === undefined ?
    (globalThis.navigator as Navigator & { gpu?: GpuApi } | undefined)?.gpu : options.gpu;
  if (!gpu) return { apiAvailable: false, adapterAvailable: false, shaderF16: false,
    browser, usable: false, reason: 'WebGPU API unavailable' };
  let adapter: Awaited<ReturnType<GpuApi['requestAdapter']>>;
  try { adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' }); }
  catch { adapter = null; }
  if (!adapter) return { apiAvailable: true, adapterAvailable: false, shaderF16: false,
    browser, usable: false, reason: 'No WebGPU adapter available' };
  const shaderF16 = adapter.features.has('shader-f16');
  if (!shaderF16) return { apiAvailable: true, adapterAvailable: true, shaderF16: false,
    browser, usable: false, reason: 'WebGPU adapter lacks shader-f16' };
  if (browser === 'firefox' && !options.firefoxCompatibility)
    return { apiAvailable: true, adapterAvailable: true, shaderF16: true,
      browser, usable: false, reason: 'Firefox WebGPU requires slow Wllama compatibility mode' };
  return { apiAvailable: true, adapterAvailable: true, shaderF16: true,
    browser, usable: true, reason: 'WebGPU adapter supports shader-f16' };
}
