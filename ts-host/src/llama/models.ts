/**
 * The model a managed llama.cpp server serves: where its GGUF and chat template come from, the stock models'
 * download with size and SHA-256 verification, the content hash tuned adapters bind to, and the server's identity
 * (`runtimeLabel`, `warmTag`). Files live under the host's directories (llama/host.ts).
 */
import { createReadStream, createWriteStream, existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { basename, dirname, join, resolve } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { DEFAULT_LOCAL_MODEL, type LocalModelRelease } from '../model-catalog.js';
import { discoverLlamaRuntime } from './runtime.js';
import { planServerSlots, type LocalServerSettings, type SlotPlan } from './slots.js';
import type { LlamaHost } from './host.js';

/** Where a stock model is downloaded to (the checkout's models/ directory is used first when it holds the file). */
export function stockModelPath(host: LlamaHost, release: LocalModelRelease): string {
  return join(host.directories.models, `${release.sha256.slice(0, 12)}-${release.file}`);
}

function checkoutModelPath(host: LlamaHost, release: LocalModelRelease): string | null {
  return host.directories.checkoutModels ? resolve(host.directories.checkoutModels, release.file) : null;
}

export function localModelPrerequisites(host: LlamaHost, release: LocalModelRelease = DEFAULT_LOCAL_MODEL) {
  const { environment } = host;
  const runtime = discoverLlamaRuntime(host), found = runtime.selected?.path ?? null;
  const explicitModel = environment.NATLANG_MODEL_PATH ? resolve(environment.NATLANG_MODEL_PATH) : null;
  const checkoutModel = checkoutModelPath(host, release);
  const modelPath = (explicitModel ? [explicitModel] : [checkoutModel, stockModelPath(host, release)])
    .find((value): value is string => Boolean(value && existsSync(value))) ?? null;
  const template = templateCandidates(host, release).find(existsSync) ?? null;
  const modelAvailable = Boolean(modelPath || (!explicitModel && release.downloadUrl));
  // A configured GGUF without a template uses the one it carries.
  const templateAvailable = Boolean(template || (explicitModel && !environment.NATLANG_TEMPLATE));
  return { executable: found, available: Boolean(found && modelAvailable && templateAvailable),
    command: environment.NATLANG_LLAMA_SERVER ?? 'llama-server', runtime,
    model: explicitModel ? basename(explicitModel) : release.id, modelPath, template,
    downloadable: !explicitModel && Boolean(release.downloadUrl), downloadBytes: explicitModel ? null : release.bytes };
}

/** The slot count a managed local server would start with, and why (`natlang doctor`). */
export function localSlotPlan(host: LlamaHost, local: LocalServerSettings | undefined,
  modelPath: string | null = localModelPrerequisites(host).modelPath, release: LocalModelRelease = DEFAULT_LOCAL_MODEL): SlotPlan {
  const modelBytes = modelPath && existsSync(modelPath) ? statSync(modelPath).size : release.bytes;
  // A configured GGUF is another model: only the stock file's measured cache size applies to it.
  const stock = !host.environment.NATLANG_MODEL_PATH;
  return planServerSlots({ local, modelBytes, defaultContextTokens: release.contextTokens, environment: host.environment,
    ...(stock ? { defaultKvBytesPerToken: release.kvBytesPerToken } : {}) });
}

async function fileHash(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

/**
 * The sha256 of the base GGUF the managed server runs: the pinned default's declared hash, or the configured file's
 * content hash (cached beside it by size and mtime). Tuned GGUF adapters must bind exactly this file.
 */
export async function managedModelSha256(host: LlamaHost, release: LocalModelRelease = DEFAULT_LOCAL_MODEL): Promise<string> {
  const { environment } = host;
  if (!environment.NATLANG_MODEL_PATH) return release.sha256;
  const path = resolve(environment.NATLANG_MODEL_PATH), stat = statSync(path);
  const cache = join(host.directories.modelHashes, createHash('sha256').update(path).digest('hex').slice(0, 24));
  const stamp = `${stat.size}:${stat.mtimeMs}`;
  try { const [seen, hash] = readFileSync(cache, 'utf8').split('\n'); if (seen === stamp && hash && /^[0-9a-f]{64}$/.test(hash)) return hash; } catch { /* not cached */ }
  const hash = await fileHash(path);
  mkdirSync(dirname(cache), { recursive: true }); writeFileSync(cache, `${stamp}\n${hash}`);
  return hash;
}

/** A verified file is remembered by size and mtime beside it, so a 2 GB model is hashed once, not at every start. */
async function verifiedRelease(path: string, release: LocalModelRelease): Promise<boolean> {
  if (!existsSync(path)) return false;
  const stat = statSync(path), stamp = `${path}.verified`, expected = `${stat.size}:${stat.mtimeMs}:${release.sha256}`;
  if (stat.size !== release.bytes) return false;
  try { if (readFileSync(stamp, 'utf8') === expected) return true; } catch { /* not verified yet */ }
  if (await fileHash(path) !== release.sha256) return false;
  try { writeFileSync(stamp, expected); } catch { /* a read-only location is verified again next time */ }
  return true;
}

/** The download of `release` another live process is writing now (its `.partial-<pid>` file), if any. */
export function stockModelDownload(host: LlamaHost, release: LocalModelRelease):
  { received: number; total: number; pid: number } | null {
  const destination = stockModelPath(host, release), prefix = `${basename(destination)}.partial-`;
  let names: string[];
  try { names = readdirSync(dirname(destination)); } catch { return null; }
  for (const name of names) {
    const pid = name.startsWith(prefix) ? Number(name.slice(prefix.length)) : NaN;
    if (!Number.isInteger(pid) || pid <= 0 || pid === process.pid) continue;
    try { process.kill(pid, 0); } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'EPERM') continue; }
    try { return { received: statSync(join(dirname(destination), name)).size, total: release.bytes, pid }; } catch { continue; }
  }
  return null;
}

/** The stock model's verified local file, downloaded first when absent (with progress on the host's log). A download
 * another process is already making is awaited instead of repeated. */
export async function ensureStockModel(host: LlamaHost, release: LocalModelRelease,
  onProgress?: (received: number, total: number) => void): Promise<string> {
  const { log } = host;
  const checkout = checkoutModelPath(host, release);
  if (checkout && await verifiedRelease(checkout, release)) return checkout;
  const destination = stockModelPath(host, release);
  if (await verifiedRelease(destination, release)) return destination;
  for (let other = stockModelDownload(host, release), announced = false; other; other = stockModelDownload(host, release)) {
    if (!announced) log.write(`natlang: waiting for the ${release.label} download already running in process ${other.pid}\n`);
    announced = true; onProgress?.(other.received, other.total);
    await new Promise(wake => setTimeout(wake, 1000));
  }
  if (await verifiedRelease(destination, release)) return destination;
  mkdirSync(dirname(destination), { recursive: true });
  const temporary = `${destination}.partial-${process.pid}`;
  log.write(`natlang: downloading ${release.label} from ${release.source.repository} (${Math.ceil(release.bytes / 1_000_000)} MB, once)\n`);
  try {
    const response = await host.fetch(release.downloadUrl, { redirect: 'follow' });
    if (!response.ok || !response.body) throw new Error(`model download failed: HTTP ${response.status}`);
    let received = 0, reported = 0;
    const counted = async function* (body: AsyncIterable<Uint8Array>) {
      for await (const chunk of body) {
        received += chunk.length; onProgress?.(received, release.bytes);
        // A terminal gets one updating line; logs (the dashboard's jobs, pipes) a line every 5%.
        const terminal = log.isTTY === true;
        if (!onProgress && received - reported > (terminal ? 16_000_000 : release.bytes / 20)) {
          reported = received;
          const line = `natlang: downloaded ${Math.round(received / 1_000_000)} of ${Math.ceil(release.bytes / 1_000_000)} MB (${Math.floor(100 * received / release.bytes)}%)`;
          log.write(terminal ? `\r\x1b[2K${line}` : `${line}\n`);
        }
        yield chunk;
      }
    };
    await pipeline(Readable.from(counted(response.body as AsyncIterable<Uint8Array>)), createWriteStream(temporary, { flags: 'wx' }));
    if (reported && log.isTTY === true) log.write('\r\x1b[2K');
    if (!await verifiedRelease(temporary, release)) throw new Error(`downloaded ${release.file} failed its size or SHA-256 check`);
    try { renameSync(temporary, destination); renameSync(`${temporary}.verified`, `${destination}.verified`); }
    catch (failure) { if (!await verifiedRelease(destination, release)) throw failure; }
    return destination;
  } finally { rmSync(temporary, { force: true }); rmSync(`${temporary}.verified`, { force: true }); }
}

/** The GGUF to serve: the configured file (`NATLANG_MODEL_PATH`), or the stock model, downloaded first when absent. */
export async function ensureModel(host: LlamaHost, release: LocalModelRelease): Promise<string> {
  const { environment } = host;
  if (environment.NATLANG_MODEL_PATH) {
    const path = resolve(environment.NATLANG_MODEL_PATH);
    if (!existsSync(path)) throw new Error(`NATLANG_MODEL_PATH does not exist: ${path}`);
    return path;
  }
  return ensureStockModel(host, release);
}

function templateCandidates(host: LlamaHost, release: LocalModelRelease): string[] {
  const { environment } = host;
  if (environment.NATLANG_TEMPLATE) return [resolve(environment.NATLANG_TEMPLATE)];
  if (environment.NATLANG_MODEL_PATH) return [];
  return host.directories.templates.map(directory => resolve(directory, release.template));
}

/** The chat template file to pass, or null for a configured GGUF without one (it is served with its own). */
export function templatePath(host: LlamaHost, release: LocalModelRelease): string | null {
  const candidates = templateCandidates(host, release);
  if (!candidates.length) return null;
  const selected = candidates.find(existsSync);
  if (!selected) throw new Error(`missing the chat template ${candidates[0]} for ${host.environment.NATLANG_TEMPLATE ? 'NATLANG_TEMPLATE' : release.id}`);
  return selected;
}

/** A llama.cpp binary's identity as `b<build>g<commit>`: the build number alone does not tell fork commits apart. */
export const runtimeLabel = (runtime: { build: number | null; commit: string | null } | null | undefined) =>
  runtime ? `b${runtime.build ?? 'x'}${runtime.commit ? `g${runtime.commit.slice(0, 7)}` : ''}` : 'bx';
/**
 * The key part of a warm state for this model file, llama.cpp build and slot layout: a saved slot state restores only
 * into a server with the same number of slots and context per slot.
 */
export const warmTag = (modelSha256: string, runtime: string, plan: Pick<SlotPlan, 'slots' | 'contextPerSlot'>) =>
  `${modelSha256.slice(0, 12)}-${runtime}-p${plan.slots}x${plan.contextPerSlot}`;
