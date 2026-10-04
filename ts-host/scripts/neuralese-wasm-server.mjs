#!/usr/bin/env node
/**
 * The WebAssembly Neuralese service behind a local HTTP port (for conformance tests against the reference and fork
 * servers): prints {"listening", "dialect", "cutoff"} like them.
 *
 *   node scripts/neuralese-wasm-server.mjs -m model.gguf --nz neuralese.gguf [--port 0] [--max-block N] [-c ctx]
 */
import { createServer } from 'node:http';
import { parseArgs } from 'node:util';
import { startNodeNeuralese } from '../dist/browser/neuralese-wasm.js';

const { values } = parseArgs({ options: { m: { type: 'string' }, nz: { type: 'string' }, port: { type: 'string', default: '0' },
  'max-block': { type: 'string' }, c: { type: 'string' }, module: { type: 'string' } } });
const factory = (await import(values.module ?? new URL('../vendor/neuralese-wasm/neuralese-wasm.mjs', import.meta.url).href)).default;
const started = await startNodeNeuralese({ factory, model: values.m, heads: values.nz,
  ...(values['max-block'] ? { maxBlock: Number(values['max-block']) } : {}), ...(values.c ? { nCtx: Number(values.c) } : {}) });
let chain = Promise.resolve();
const server = createServer((req, res) => {
  const chunks = [];
  req.on('data', chunk => chunks.push(chunk));
  req.on('end', () => {
    chain = chain.then(() => {
      const result = started.service.handle(req.method, new URL(req.url, 'http://x').pathname, new Uint8Array(Buffer.concat(chunks)));
      res.writeHead(result.status, { 'content-type': result.contentType });
      res.end(Buffer.from(result.body));
    }).catch(error => { res.writeHead(500); res.end(String(error)); });
  });
});
server.listen(Number(values.port), '127.0.0.1', () => {
  console.log(JSON.stringify({ listening: `http://127.0.0.1:${server.address().port}`, dialect: started.dialect, cutoff: started.cutoff }));
});
