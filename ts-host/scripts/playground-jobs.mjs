/** Localhost-only, allowlisted training and evaluation jobs for the browser workbench. */
import { createHash, randomBytes, randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createReadStream } from 'node:fs';
import { appendFile, mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { basename, join, resolve, sep } from 'node:path';

const json = (response, status, value) => {
  const body = JSON.stringify(value);
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store', 'Content-Length': Buffer.byteLength(body) });
  response.end(body);
};
const slug = value => typeof value === 'string' && /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(value);
const fail = message => { throw new Error(message); };

export function createPlaygroundJobs(root) {
  const token = randomBytes(24).toString('hex');
  const active = new Map();
  const records = new Map();
  const jobRoot = join(root, 'runs/playground-jobs');
  const inside = (path, prefixes) => {
    if (typeof path !== 'string' || path.includes('\0')) fail('invalid path');
    const full = resolve(root, path);
    if (!full.startsWith(root + sep) || !prefixes.some(prefix => full.startsWith(join(root, prefix) + sep)))
      fail('path must remain in an allowed repository directory');
    return full;
  };
  async function catalog() {
    async function files(folder, suffix) {
      const directory = join(root, folder);
      const entries = await readdir(directory, { withFileTypes: true }).catch(() => []);
      const found = [];
      for (const entry of entries) if (entry.isFile() && entry.name.endsWith(suffix)) {
        const path = `${folder}/${entry.name}`, info = await stat(join(directory, entry.name));
        found.push({ path, bytes: info.size, updatedAt: info.mtime.toISOString() });
      }
      return found.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
    }
    const runs = await readdir(join(root, 'runs'), { withFileTypes: true }).catch(() => []);
    return { datasets: await files('data', '.jsonl'), models: await files('models', '.gguf'),
      checkpoints: runs.filter(entry => entry.isDirectory() && entry.name !== 'playground-jobs')
        .map(entry => ({ name: entry.name, merged: `runs/${entry.name}/merged` })) };
  }
  function command(config) {
    const { kind, dataset, name } = config;
    if (!slug(name)) fail('job name must use letters, digits, hyphen, or underscore');
    const data = kind === 'gguf' ? null : inside(dataset, ['data']);
    if (data && !data.endsWith('.jsonl')) fail('dataset must be a JSONL file');
    const output = `runs/playground-${name}`;
    switch (kind) {
      case 'materialize':
        return { cmd: 'python', args: ['scripts/materialize_ir.py', data,
          inside(`data/${name}-materialized.jsonl`, ['data'])] };
      case 'teacher': {
        const modelId = String(config.modelId ?? '').trim();
        if (!modelId || modelId.length > 128) fail('teacher model ID is required');
        const seed = Number(config.rootSeed ?? 0);
        if (!Number.isSafeInteger(seed) || seed < 0) fail('root seed must be a nonnegative integer');
        return { cmd: 'python', args: ['scripts/collect_scenario_teacher.py', data,
          inside(`data/${name}-teacher.jsonl`, ['data']), '--model-id', modelId,
          '--root-seed', String(seed)] };
      }
      case 'export':
        return { cmd: 'python', args: ['scripts/export_sft.py', data,
          inside(`data/${name}-sft.jsonl`, ['data']), '--resume'] };
      case 'train': {
        const steps = Number(config.steps ?? 300);
        if (!Number.isInteger(steps) || steps < 1 || steps > 100000) fail('steps must be 1–100000');
        const model = config.model ? inside(config.model, ['runs']) : null;
        return { cmd: 'python', args: ['scripts/train_lora.py', data, inside(output, ['runs']),
          '--steps', String(steps), ...(model ? ['--model', model] : [])] };
      }
      case 'evaluate':
        return { cmd: 'python', args: ['scripts/eval_turns.py', data,
          '--out', inside(`${output}/eval.json`, ['runs']), '--model-label', name] };
      case 'gguf': {
        const source = inside(config.checkpoint, ['runs']);
        if (!source.endsWith('/merged')) fail('checkpoint must be a merged model directory');
        const quant = config.quant ?? 'Q4_K_M';
        if (!['Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0'].includes(quant)) fail('unsupported quantization');
        return { cmd: 'bash', args: ['scripts/to_gguf.sh',
          source.slice(root.length + 1), `models/${name}-${quant}.gguf`, quant] };
      }
      default: fail('unknown job kind');
    }
  }
  async function readBody(request) {
    let body = '';
    for await (const chunk of request) {
      body += chunk;
      if (body.length > 10000) fail('request is too large');
    }
    return JSON.parse(body);
  }
  async function start(config) {
    const { cmd, args } = command(config);
    if (active.size >= 2) fail('two workbench jobs are already running');
    const id = randomUUID(), directory = join(jobRoot, id);
    await mkdir(directory, { recursive: true });
    const record = { id, kind: config.kind, name: config.name, state: 'running',
      startedAt: new Date().toISOString(), endedAt: null, exitCode: null,
      command: [cmd, ...args.map(arg => arg.startsWith(root + sep) ? arg.slice(root.length + 1) : arg)],
      configHash: createHash('sha256').update(JSON.stringify(config)).digest('hex') };
    await writeFile(join(directory, 'job.json'), JSON.stringify(record, null, 2) + '\n');
    records.set(id, record);
    if (config.kind === 'evaluate' || config.kind === 'train')
      await mkdir(join(root, `runs/playground-${config.name}`), { recursive: true });
    const child = spawn(cmd, args, { cwd: root, env: process.env, stdio: ['ignore', 'pipe', 'pipe'],
      detached: process.platform !== 'win32' });
    active.set(id, child);
    const log = join(directory, 'output.log');
    let pending = Promise.resolve();
    for (const stream of [child.stdout, child.stderr]) stream.on('data', chunk => {
      pending = pending.then(() => appendFile(log, chunk)).catch(() => {});
    });
    child.on('error', error => { pending = pending.then(() => appendFile(log, `\n${error.message}\n`)); });
    child.on('close', async (code, signal) => {
      await pending; active.delete(id);
      record.state = signal ? 'stopped' : code === 0 ? 'completed' : 'failed';
      record.exitCode = code; record.endedAt = new Date().toISOString();
      await writeFile(join(directory, 'job.json'), JSON.stringify(record, null, 2) + '\n');
    });
    return record;
  }
  async function list() {
    const directories = await readdir(jobRoot, { withFileTypes: true }).catch(() => []);
    const found = [];
    for (const entry of directories) if (entry.isDirectory()) {
      try {
        const record = records.get(entry.name) ?? JSON.parse(await readFile(join(jobRoot, entry.name, 'job.json'), 'utf8'));
        if (record.state === 'running' && !active.has(record.id)) record.state = 'interrupted';
        found.push(record);
      } catch { /* incomplete record */ }
    }
    return found.sort((a, b) => b.startedAt.localeCompare(a.startedAt));
  }
  return async function handle(request, response) {
    const url = new URL(request.url, 'http://127.0.0.1');
    if (!url.pathname.startsWith('/api/playground/')) return false;
    try {
      if (!/^(?:127\.0\.0\.1|localhost):\d+$/.test(request.headers.host ?? ''))
        return json(response, 403, { error: 'localhost is required' }), true;
      if (request.method !== 'GET') {
        if (request.headers.origin !== `http://${request.headers.host}` ||
            request.headers['x-natlang-token'] !== token) return json(response, 403, { error: 'invalid local session' }), true;
      }
      if (url.pathname === '/api/playground/session' && request.method === 'GET')
        return json(response, 200, { token }), true;
      if (url.pathname === '/api/playground/catalog' && request.method === 'GET')
        return json(response, 200, await catalog()), true;
      if (url.pathname === '/api/playground/jobs' && request.method === 'GET')
        return json(response, 200, await list()), true;
      if (url.pathname === '/api/playground/jobs' && request.method === 'POST')
        return json(response, 201, await start(await readBody(request))), true;
      const match = /^\/api\/playground\/jobs\/([0-9a-f-]+)(?:\/(log|stop))?$/.exec(url.pathname);
      if (match && match[2] === 'log' && request.method === 'GET') {
        const file = join(jobRoot, match[1], 'output.log');
        const size = (await stat(file).catch(() => ({ size: 0 }))).size;
        const start = Math.max(0, size - 65536);
        response.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' });
        if (!size) response.end(); else createReadStream(file, { start }).pipe(response);
        return true;
      }
      if (match && match[2] === 'stop' && request.method === 'POST') {
        const child = active.get(match[1]); if (!child) fail('job is not running');
        if (process.platform === 'win32') child.kill('SIGTERM');
        else process.kill(-child.pid, 'SIGTERM');
        const force = setTimeout(() => {
          if (active.has(match[1])) {
            try { if (process.platform === 'win32') child.kill('SIGKILL');
              else process.kill(-child.pid, 'SIGKILL'); } catch { /* already exited */ }
          }
        }, 10000);
        force.unref();
        return json(response, 200, { stopping: true }), true;
      }
      return json(response, 404, { error: 'unknown workbench endpoint' }), true;
    } catch (error) { return json(response, 400, { error: error.message }), true; }
  };
}
