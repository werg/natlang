// Optional trusted desktop host. This is not a sandbox.
const readline = require('node:readline');
const fs = require('node:fs');
const cp = require('node:child_process');
const vm = require('node:vm');

const jobs = new Map();
const buffers = new Map();
let nextId = 1;
let events = [];
const id = (prefix) => `${prefix}-${nextId++}`;
const bounded = (value) => String(value ?? '').slice(0, 1024 * 1024);
const record = (operation, detail) => events.push({ operation, ...detail });

const host = Object.freeze({
  readText(path) {
    const value = fs.readFileSync(path, 'utf8');
    record('file.readText', { path, bytes: Buffer.byteLength(value) });
    return value;
  },
  readBytes(path) {
    const key = id('buffer');
    const value = fs.readFileSync(path);
    buffers.set(key, value);
    record('file.readBytes', { path, id: key, bytes: value.length });
    return key;
  },
  buffer(id) {
    if (!buffers.has(id)) throw new Error(`unknown or disposed buffer ${id}`);
    return buffers.get(id);
  },
  run(argv, options = {}) {
    if (!Array.isArray(argv) || !argv.length || !argv.every(x => typeof x === 'string'))
      throw new Error('run requires a nonempty string argv');
    const result = cp.spawnSync(argv[0], argv.slice(1), {
      cwd: options.cwd, input: options.input, encoding: 'utf8',
      timeout: options.timeoutMs ?? 5000, maxBuffer: 1024 * 1024,
    });
    const out = { status: result.status, signal: result.signal,
                  stdout: bounded(result.stdout), stderr: bounded(result.stderr),
                  error: result.error?.message ?? null };
    record('process.run', { argv, status: out.status, signal: out.signal });
    return out;
  },
  start(argv, options = {}) {
    if (!Array.isArray(argv) || !argv.length || !argv.every(x => typeof x === 'string'))
      throw new Error('start requires a nonempty string argv');
    const key = id('job');
    const child = cp.spawn(argv[0], argv.slice(1), { cwd: options.cwd, stdio: ['ignore', 'pipe', 'pipe'] });
    const job = { child, status: 'running', exitCode: null, signal: null, stdout: '', stderr: '' };
    jobs.set(key, job);
    child.stdout.on('data', chunk => job.stdout = bounded(job.stdout + chunk));
    child.stderr.on('data', chunk => job.stderr = bounded(job.stderr + chunk));
    child.on('error', error => { job.status = 'failed'; job.error = error.message; });
    child.on('exit', (code, signal) => { job.status = 'finished'; job.exitCode = code; job.signal = signal; });
    record('process.start', { argv, id: key });
    return key;
  },
  poll(key) {
    const job = jobs.get(key);
    if (!job) throw new Error(`unknown or disposed job ${key}`);
    return { id: key, status: job.status, exitCode: job.exitCode, signal: job.signal,
             stdout: job.stdout, stderr: job.stderr, error: job.error ?? null };
  },
  cancel(key) {
    const job = jobs.get(key);
    if (!job) throw new Error(`unknown or disposed job ${key}`);
    const requested = job.status === 'running' && job.child.kill();
    record('process.cancel', { id: key, requested });
    return { id: key, requested, status: job.status };
  },
});

const context = vm.createContext({ host, console: undefined });
function portableResult(value, seen = new Set()) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value) || (Number.isInteger(value) && !Number.isSafeInteger(value)))
      throw new Error('result is not an exact portable number');
    return value;
  }
  if (typeof value !== 'object' || Buffer.isBuffer(value) || ArrayBuffer.isView(value))
    throw new Error('result contains a native or unsupported value');
  if (seen.has(value)) throw new Error('result contains a cycle');
  seen.add(value);
  if (Array.isArray(value)) {
    const result = value.map(v => portableResult(v, seen));
    seen.delete(value);
    return result;
  }
  if (Object.prototype.toString.call(value) !== '[object Object]')
    throw new Error('result contains a native object');
  const result = {};
  for (const [key, item] of Object.entries(value)) result[key] = portableResult(item, seen);
  seen.delete(value);
  return result;
}
function execute(request) {
  events = [];
  const scope = structuredClone(request.scope);
  context.self = scope;
  context.args = scope.args;
  context.locals = scope.let || {};
  const code = request.body
    ? `(function(self,args,host){ 'use strict'; ${request.code}\n})(self,args,host)`
    : request.code;
  const result = vm.runInContext(code, context, { timeout: request.timeoutMs ?? 2000 });
  return { kind: 'result', value: portableResult(result === undefined ? null : result), events };
}

const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
lines.on('line', line => {
  try {
    const request = JSON.parse(line);
    if (request.kind === 'dispose') {
      for (const job of jobs.values()) if (job.status === 'running') job.child.kill();
      jobs.clear(); buffers.clear();
      process.stdout.write(JSON.stringify({ kind: 'disposed' }) + '\n');
      process.exit(0);
    }
    process.stdout.write(JSON.stringify(execute(request)) + '\n');
  } catch (error) {
    process.stdout.write(JSON.stringify({ kind: 'error', message: error.message, events }) + '\n');
  }
});
