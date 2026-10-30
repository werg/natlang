/**
 * Stock GGUF models the managed llama.cpp backend downloads, verifies and serves without any setup beyond
 * `natlang setup`. While our own LFM3 checkpoints train, the default is the stock instruction model
 * LiquidAI/LFM3-1.7B (13/16 on the held-v3 cadence subset as a base model,
 * plans/neuralese/LFM3_1P7B_BASE_EVAL_20261027.md). A profile selects another quantization with `local.stock`.
 * Files are pinned to one repository revision and checked by size and SHA-256.
 */
export type LocalModelRelease = {
  readonly id: string; readonly label: string; readonly family: string; readonly file: string; readonly quant: string;
  readonly bytes: number; readonly sha256: string;
  /** Default context per server slot. */
  readonly contextTokens: number;
  /** Measured context memory per token (KV, sliding-window and recurrent state), which sizes the server's slots. */
  readonly kvBytesPerToken: number;
  /**
   * The model has sliding-window attention layers, so the server runs with `--swa-full`. Without it llama.cpp
   * cannot resume from a cached prompt prefix on those layers (a restored warm opening, or any earlier prompt): it logs
   * "forcing full prompt re-processing" and evaluates the whole prompt again. `kvBytesPerToken` is the cache size with it.
   */
  readonly fullSlidingWindowCache?: boolean;
  readonly downloadUrl: string;
  /** Chat template under models/templates/ (packaged as model-assets/); the GGUF's own template, extracted verbatim. */
  readonly template: string;
  readonly source: { readonly repository: string; readonly revision: string };
  /** Request fields the model is served with; a profile's request overrides them. */
  readonly request?: Readonly<Record<string, unknown>>;
};

const LFM3_REPOSITORY = 'LiquidAI/LFM3-1.7B-GGUF', LFM3_REVISION = 'b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf';
const lfm3 = (quant: string, bytes: number, sha256: string): LocalModelRelease => Object.freeze({
  id: `lfm3-1.7b-${quant.toLowerCase()}`, label: `LFM3 1.7B · ${quant}`, family: 'lfm3', file: `LFM3-1.7B-${quant}.gguf`, quant, bytes, sha256,
  // Most layers are short convolutions or 256-token windows. Measured on llama.cpp 0.8 at 163,840 tokens of Q8_0: 3,809 MiB
  // resident (12,800 bytes per token over the weights) with the default window cache, 5,380 MiB (about 23,000) with --swa-full,
  // which the server needs to reuse a cached prompt prefix at all.
  contextTokens: 32768, kvBytesPerToken: 23000, fullSlidingWindowCache: true, template: 'LFM3-1.7B.jinja',
  downloadUrl: `https://huggingface.co/${LFM3_REPOSITORY}/resolve/${LFM3_REVISION}/LFM3-1.7B-${quant}.gguf`,
  source: Object.freeze({ repository: LFM3_REPOSITORY, revision: LFM3_REVISION }),
  // Greedy, with reasoning kept in the history: the settings of the base evaluation.
  request: Object.freeze({ temperature: 0, chat_template_kwargs: Object.freeze({ preserve_thinking: true }) }),
});

export const STOCK_MODELS: readonly LocalModelRelease[] = Object.freeze([
  lfm3('Q4_K_M', 1099627808, '3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4'),
  lfm3('Q5_K_M', 1258663200, '41e51fb0be5f472d323055f2734d986d5a2429570973508fa6322f3f07f6d7f3'),
  lfm3('Q6_K', 1427638304, '54e1bd947c77167823d45bf495f1c3e2d226ca7fb242f83cfa141ee3a6ab55c6'),
  lfm3('Q8_0', 1846238496, '8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6'),
  lfm3('BF16', 3466626336, '39158a79f56edb85e0c4252ce6447b8d7883ef82ffedd02f0cda7c84da5e9f68'),
]);

/** The managed local default: near-lossless 8-bit LFM3 1.7B (1.8 GB). */
export const DEFAULT_LOCAL_MODEL: LocalModelRelease = STOCK_MODELS.find(model => model.quant === 'Q8_0')!;

export function stockModel(id: string): LocalModelRelease {
  const found = STOCK_MODELS.find(model => model.id === id);
  if (!found) throw new Error(`unknown stock model ${id}; choose ${STOCK_MODELS.map(model => model.id).join(', ')}`);
  return found;
}
