#!/usr/bin/env node
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, open, readFile, rename, stat } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { SOURCES, huggingFaceRequestHeaders } from './sources.mjs';
import { normalizeCodeSearchNet, normalizeMagicoder, normalizeMcEvalInstruct, normalizeTinyCodes, normalizeXlam, normalizeCase2Code } from './datasets.mjs';

const PAGE_SIZE = 100;
const normalizers = { codesearchnet: normalizeCodeSearchNet, magicoder: normalizeMagicoder, mceval: normalizeMcEvalInstruct, 'tiny-codes': normalizeTinyCodes, xlam: normalizeXlam, case2code: normalizeCase2Code };
const sha = value => createHash('sha256').update(value).digest('hex');
const jsonl = rows => rows.map(row => JSON.stringify(row)).join('\n') + (rows.length ? '\n' : '');
const atomicWrite = async (path, value) => {
  const temp = `${path}.${randomUUID()}.tmp`;
  const h = await open(temp, 'wx', 0o600);
  try { await h.writeFile(value); await h.sync(); } finally { await h.close(); }
  await rename(temp, path);
};
const safeError = e => String(e?.message ?? e).replace(/Bearer\s+\S+/gi, 'Bearer [redacted]');
const TRANSIENT_STATUSES = new Set([408, 425, 429, 500, 502, 503, 504]);

class AcquisitionStopped extends Error {}

function retryAfterMs(response) {
  const value = response.headers?.get?.('retry-after');
  if (!value) return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1000;
  const date = Date.parse(value);
  return Number.isFinite(date) ? Math.max(0, date - Date.now()) : null;
}

function makeJsonRequester({ fetcher, token, shouldStop, minIntervalMs, maxRetries, sleep }) {
  let lastRequestAt = 0;
  const pause = async milliseconds => {
    let remaining = milliseconds;
    while (remaining > 0) {
      if (shouldStop()) throw new AcquisitionStopped('Acquisition stopped while waiting to retry');
      const interval = Math.min(remaining, 100);
      await sleep(interval);
      remaining -= interval;
    }
    if (shouldStop()) throw new AcquisitionStopped('Acquisition stopped while waiting to retry');
  };
  return async url => {
    let lastError;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      if (shouldStop()) throw new AcquisitionStopped('Acquisition stopped before request');
      const pacingWait = Math.max(0, minIntervalMs - (Date.now() - lastRequestAt));
      if (pacingWait) await pause(pacingWait);
      if (shouldStop()) throw new AcquisitionStopped('Acquisition stopped before request');
      let response;
      try {
        response = await fetcher(url, { signal: AbortSignal.timeout(30000),
          headers: huggingFaceRequestHeaders(url, token ? { HF_TOKEN: token } : {}) });
        lastRequestAt = Date.now();
        if (!response.ok) {
          const error = new Error(`Hugging Face request failed (${response.status})`);
          if (!TRANSIENT_STATUSES.has(response.status)) throw error;
          lastError = error;
        } else {
          return await response.json();
        }
      } catch (error) {
        if (error instanceof AcquisitionStopped) throw error;
        lastError = error;
        // HTTP errors thrown above are retried only when explicitly transient.
        if (response && !response.ok && !TRANSIENT_STATUSES.has(response.status)) throw error;
      }
      if (attempt === maxRetries) break;
      const backoff = Math.min(500 * (2 ** attempt), 30000);
      const wait = Math.max(backoff, response && !response.ok ? retryAfterMs(response) ?? 0 : 0);
      await pause(wait);
    }
    throw new Error(`Hugging Face request failed after ${maxRetries + 1} attempts: ${safeError(lastError)}`);
  };
}

export async function readTokenFile(path) {
  const file = resolve(path), info = await stat(file);
  if (!info.isFile() || (info.mode & 0o077) !== 0 || (typeof process.getuid === 'function' && info.uid !== process.getuid()))
    throw new Error('--token-file must be a regular file owned by this user with permissions no broader than 0600');
  const token = (await readFile(file, 'utf8')).trim();
  if (!token || /\s/.test(token)) throw new Error('--token-file must contain one nonempty token');
  return token;
}

async function datasetRevision(source, revision, requestJson) {
  const url = new URL(`https://huggingface.co/api/datasets/${source.dataset}/revision/${encodeURIComponent(revision)}`);
  const info = await requestJson(url);
  if (typeof info.sha !== 'string' || !info.sha) throw new Error('Hugging Face revision response has no commit SHA');
  return info.sha;
}

async function pageRequest(source, { config, split, offset, length, revision }, requestJson) {
  const url = new URL(`https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(source.dataset)}&config=${encodeURIComponent(config)}&split=${encodeURIComponent(split)}&offset=${offset}&length=${length}&revision=${encodeURIComponent(revision)}`);
  const data = await requestJson(url);
  if (!Array.isArray(data.rows)) throw new Error('Unexpected datasets-server rows response');
  const total = data.num_rows_total ?? data.total_rows;
  if (!Number.isSafeInteger(total) || total < 0) throw new Error('Rows response has no valid total row count');
  if (data.rows.length > length || offset > total || data.rows.length > total - offset) throw new Error('Rows response exceeds requested page bounds or total row count');
  return { rows: data.rows, total };
}

function taskFor(source, row, meta) {
  if (row.truncated_cells?.length) return {
    version:'natlang.code_task/1', id:`${source.id}:truncated:${sha(JSON.stringify(row)).slice(0,32)}`,
    group_id:`${source.id}:truncated:${row.row_idx ?? sha(JSON.stringify(row))}`, kind:'instruction',
    language:'unknown', instruction:'', source:{name:source.id, ...meta}, function:null, cases:[],
    verification:{status:'rejected', reasons:['truncated_preview_cells']}, raw:{source_row:row},
  };
  row = row.row ?? row;
  const normalize = normalizers[source.id];
  if (normalize) return normalize(row, meta);
  return { version: 'natlang.code_task/1', id: `${source.id}:${sha(JSON.stringify(row)).slice(0, 32)}`, group_id: `${source.id}:${sha(JSON.stringify(row)).slice(0, 32)}`, kind: 'instruction', language: 'unknown', instruction: '', source: { name: source.id, revision: meta.revision, license: source.license, upstream_id: row.id ?? null }, function: null, cases: [], verification: { status: 'rejected', reasons: ['source_adapter_deferred'] }, raw: { source_row: row } };
}

export async function acquireShards({ out, ids, config = 'default', split = 'train', revision = 'main', limit = Infinity, pageSize = PAGE_SIZE, fetcher = globalThis.fetch, token = null, shouldStop = () => false, minIntervalMs = 250, maxRetries = 5, sleep = ms => new Promise(resolve => setTimeout(resolve, ms)) } = {}) {
  if (!out || !Array.isArray(ids) || !ids.length) throw new Error('out and at least one source id are required');
  if (!(limit === Infinity || Number.isSafeInteger(limit) && limit > 0)) throw new Error('limit must be a positive integer');
  if (!Number.isSafeInteger(pageSize) || pageSize < 1 || pageSize > PAGE_SIZE) throw new Error(`pageSize must be 1..${PAGE_SIZE}`);
  if (!Number.isFinite(minIntervalMs) || minIntervalMs < 0 || !Number.isInteger(maxRetries) || maxRetries < 0) throw new Error('invalid request pacing/retry settings');
  const requestJson = makeJsonRequester({ fetcher, token, shouldStop, minIntervalMs, maxRetries, sleep });
  const root = resolve(out); await mkdir(root, { recursive: true });
  const manifestPath = join(root, 'manifest.json');
  const immutableConfig = { ids, config, split, revision, pageSize };
  let manifest;
  try { manifest = JSON.parse(await readFile(manifestPath, 'utf8')); }
  catch (e) { if (e.code !== 'ENOENT') throw new Error(`Invalid manifest: ${e.message}`); }
  if (manifest) {
    if (JSON.stringify(manifest.config) !== JSON.stringify(immutableConfig)) throw new Error('Acquisition config differs from manifest; use the same sources/config/split/revision/pageSize');
    if (limit < (manifest.limit === null ? Infinity : manifest.limit)) throw new Error('limit cannot decrease on resume');
    for (const [id, state] of Object.entries(manifest.sources)) for (const shard of state.shards ?? []) {
      for (const [name, expected] of [[shard.raw, shard.rawSha256], [shard.tasks, shard.tasksSha256]]) {
        let content;
        try { content = await readFile(join(root, name), 'utf8'); } catch { throw new Error(`Committed shard missing: ${name}`); }
        if (sha(content) !== expected) throw new Error(`Committed shard hash mismatch: ${name}`);
      }
    }
  } else manifest = { version: 1, config: immutableConfig, limit, sources: Object.fromEntries(ids.map(id => [id, { status: 'pending', nextOffset: 0, shards: [] }])) };
  manifest.limit = limit;
  let dirty = false;
  const commitManifest = async () => { await atomicWrite(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`); dirty = false; };
  await commitManifest();
  for (const id of ids) {
    const source = SOURCES.find(s => s.id === id && s.type === 'huggingface');
    if (!source) { manifest.sources[id] = { status: 'error', error: 'unknown Hugging Face source id', nextOffset: 0, shards: [] }; dirty = true; await commitManifest(); continue; }
    const state = manifest.sources[id];
    if (state.status === 'complete' && (state.totalRows !== undefined && state.nextOffset >= state.totalRows || limit <= state.limitAtCompletion)) continue;
    try {
      const currentRevision = await datasetRevision(source, revision, requestJson);
      if (state.resolvedRevision && state.resolvedRevision !== currentRevision) throw new Error('Dataset revision changed during resumable acquisition');
      state.resolvedRevision = currentRevision;
      state.provenance = 'datasets-server preview: raw shard hashes identify exact observations; repository revision is metadata, not a guaranteed immutable rows endpoint';
      state.status = 'running'; delete state.error; dirty = true;
      while (!shouldStop()) {
        const remaining = limit === Infinity ? pageSize : Math.min(pageSize, limit - state.nextOffset);
        if (remaining <= 0 || state.totalRows !== undefined && state.nextOffset >= state.totalRows) break;
        const { rows, total } = await pageRequest(source, { config: source.defaultConfig === 'default' ? config : config === 'default' ? source.defaultConfig : config, split, offset: state.nextOffset, length: remaining, revision: state.resolvedRevision }, requestJson);
        if (state.totalRows !== undefined && state.totalRows !== total) throw new Error('Dataset total row count changed during resumable acquisition');
        state.totalRows = total;
        if (!rows.length) {
          if (state.nextOffset < total) throw new Error('Unexpected empty page before total row count');
          break;
        }
        if (!shouldStop() && await datasetRevision(source, revision, requestJson) !== state.resolvedRevision) throw new Error('Dataset revision changed during page acquisition');
        const index = state.shards.length;
        const rawText = jsonl(rows), rawName = `${id}-${String(index).padStart(6, '0')}.raw.jsonl`, taskName = `${id}-${String(index).padStart(6, '0')}.tasks.jsonl`;
        const meta = { revision: state.resolvedRevision, license: source.license, split };
        const tasks = rows.map(row => taskFor(source, row, meta));
        await atomicWrite(join(root, rawName), rawText);
        await atomicWrite(join(root, taskName), jsonl(tasks));
        state.shards.push({ offset: state.nextOffset, rows: rows.length, raw: rawName, rawSha256: sha(rawText), tasks: taskName, tasksSha256: sha(jsonl(tasks)) });
        state.nextOffset += rows.length; dirty = true;
        await commitManifest();
        if (state.nextOffset >= total) break;
      }
      state.status = state.totalRows !== undefined && state.nextOffset >= state.totalRows || limit !== Infinity && state.nextOffset >= limit ? 'complete' : shouldStop() ? 'paused' : 'complete';
      state.limitAtCompletion = limit; dirty = true; await commitManifest();
    } catch (error) {
      if (error instanceof AcquisitionStopped) {
        state.status = 'paused'; delete state.error; dirty = true; await commitManifest();
        break;
      }
      state.status = 'error'; state.error = safeError(error); dirty = true; await commitManifest();
    }
  }
  if (dirty) await commitManifest();
  return manifest;
}

function parseArgs(argv) {
  const options = {};
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i]; if (!key.startsWith('--')) throw new Error(`Unexpected argument: ${key}`);
    const value = argv[++i]; if (!value || value.startsWith('--')) throw new Error(`${key} requires a value`);
    options[key.slice(2)] = value;
  }
  if (!options.out || !options.sources) throw new Error('Usage: acquire-shards.mjs --out DIR --sources id,id [--config NAME] [--split NAME] [--revision REF] [--limit N] [--token-file PATH]');
  return options;
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv), token = args['token-file'] ? await readTokenFile(args['token-file']) : process.env.HF_TOKEN || process.env.HUGGINGFACE_HUB_TOKEN || null;
  const ids = args.sources.split(',').filter(Boolean);
  let stopping = false;
  const stop = () => { stopping = true; };
  process.on('SIGINT', stop); process.on('SIGTERM', stop);
  try {
    const manifest = await acquireShards({ out: args.out, ids, config: args.config ?? 'default', split: args.split ?? 'train', revision: args.revision ?? 'main', limit: args.limit ? Number(args.limit) : Infinity, pageSize: args['page-size'] ? Number(args['page-size']) : PAGE_SIZE, token, shouldStop: () => stopping });
    console.log(JSON.stringify(Object.fromEntries(Object.entries(manifest.sources).map(([id, s]) => [id, { status: s.status, nextOffset: s.nextOffset, totalRows: s.totalRows }]))));
    if (Object.values(manifest.sources).some(s => s.status === 'error')) process.exitCode = 1;
    else if (stopping || Object.values(manifest.sources).some(s => s.status !== 'complete')) process.exitCode = 75;
  } finally { process.removeListener('SIGINT', stop); process.removeListener('SIGTERM', stop); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch(error => { console.error(safeError(error)); process.exitCode = 1; });
