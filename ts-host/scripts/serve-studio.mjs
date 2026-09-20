#!/usr/bin/env node
/** Trusted localhost studio, with authenticated, cancellable, individually journaled jobs. */
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { readFile, writeFile, mkdir, stat, rename, appendFile } from 'node:fs/promises';
import { resolve, join, extname, sep } from 'node:path';
import { randomBytes, randomUUID } from 'node:crypto';
import { fork } from 'node:child_process';
import { pathToFileURL } from 'node:url';
const root = resolve(import.meta.dirname, '../..');
const MIME = { '.html': 'text/html; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.wasm': 'application/wasm', '.mp4': 'video/mp4' };
const slug = value => typeof value === 'string' && /^[a-zA-Z0-9_-]{1,80}$/.test(value);
const respond = (res, status, value) => { res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }); res.end(JSON.stringify(value)); };
export async function startStudio({ port = 8766, dataRoot = join(root, 'runs/studio') } = {}) {
    await mkdir(dataRoot, { recursive: true });
    const token = randomBytes(32).toString('hex'), jobs = new Map(), sessions = new Set(), journalTails = new Map();
    async function journal(job) { const { child, ...record } = job, body = JSON.stringify(record); const prior = journalTails.get(job.id) ?? Promise.resolve(); const next = prior.then(async () => { const temporary = join(dataRoot, `${job.id}.${randomUUID()}.tmp`); await writeFile(temporary, body); await rename(temporary, join(dataRoot, `${job.id}.json`)); }); const settled=next.catch(()=>{});journalTails.set(job.id,settled);try{await next;}finally{if(journalTails.get(job.id)===settled)journalTails.delete(job.id);} }
    async function body(req) { const chunks = []; let size = 0; for await (const chunk of req) {
        size += chunk.length;
        if (size > 32 * 1024 * 1024)
            throw new Error('Request exceeds 32 MiB transport allowance');
        chunks.push(chunk);
    } return Buffer.concat(chunks); }
    const server = createServer(async (req, res) => {
        try {
            const origin = `http://127.0.0.1:${server.address().port}`;
            if (req.headers.host !== new URL(origin).host)
                return respond(res, 403, { error: 'Unrecognized host' });
            if (req.headers.origin && req.headers.origin !== origin)
                return respond(res, 403, { error: 'Cross-origin requests are not accepted' });
            const url = new URL(req.url, origin);
            if (url.pathname === '/favicon.ico') {
                res.writeHead(204);
                return res.end();
            }
            if (url.pathname === '/api/studio/session' && req.method === 'GET') {
                let session = url.searchParams.get('session');
                if (!slug(session))
                    session = randomUUID();
                sessions.add(session);
                return respond(res, 200, { token, session, authority: 'trusted-local-host' });
            }
            if (url.pathname.startsWith('/api/studio/')) {
                if (req.headers['x-studio-token'] !== token)
                    return respond(res, 403, { error: 'Studio token required' });
                const session = req.headers['x-studio-session'];
                if (!sessions.has(session))
                    return respond(res, 403, { error: 'Unknown studio session' });
                const directory = join(dataRoot, session);
                await mkdir(directory, { recursive: true });
                if (url.pathname === '/api/studio/upload' && req.method === 'POST') {
                    const asset = `asset-${randomUUID()}.mp4`;
                    await mkdir(join(directory, 'assets'), { recursive: true });
                    await writeFile(join(directory, 'assets', asset), await body(req));
                    return respond(res, 200, { asset });
                }
                if (url.pathname === '/api/studio/jobs' && req.method === 'POST') {
                    const config = JSON.parse((await body(req)).toString());
                    if (typeof config.operation !== 'string' || !config.payload)
                        throw new Error('Operation and payload required');
                    // A browser event/operation key is an idempotency key, including after server restart.
                    if (!slug(config.id))
                        throw new Error('Operation ID required');
                    let prior = jobs.get(config.id);
                    if (!prior) {
                        try {
                            prior = JSON.parse(await readFile(join(dataRoot, `${config.id}.json`), 'utf8'));
                            if (prior.status === 'running')
                                prior.status = 'unknown';
                        }
                        catch (error) {
                            if (error.code !== 'ENOENT')
                                throw error;
                        }
                    }
                    if (prior) {
                        if (prior.session !== session || JSON.stringify(prior.request) !== JSON.stringify(config))
                            throw new Error('Operation ID collision');
                        return respond(res, 200, { id: prior.id, status: prior.status });
                    }
                    if ([...jobs.values()].some(job => job.session === session && job.status === 'running'))
                        return respond(res, 409, { error: 'This session already owns a running job' });
                    const job = { id: config.id, session, status: 'running', request: config, output: '', started: new Date().toISOString() };
                    jobs.set(job.id, job);
                    await journal(job);
                    const env = { ...process.env };
                    delete env.NODE_TEST_CONTEXT;
                    const child = fork(join(root, 'ts-host/scripts/studio-operations.mjs'), ['--worker'], { env, detached: process.platform !== 'win32', stdio: ['ignore', 'ignore', 'pipe', 'ipc'] });
                    job.child = child;
                    let chain = Promise.resolve();
                    const persist = () => { chain = chain.then(() => journal(job)); chain.catch(error => console.error('Job journal:', error)); return chain; };
                    const log = chunk => { job.output = (job.output + chunk).slice(-65536); chain = chain.then(() => appendFile(join(dataRoot, `${job.id}.log`), chunk)); chain.catch(error => console.error('Job log:', error)); return chain; };
                    child.stderr.on('data', chunk => log(chunk.toString()));
                    child.on('message', message => { if (message.progress)
                        log(message.progress).then(()=>{if(child.connected)child.send({progressAck:message.progressId},()=>{});}).catch(()=>{if(child.connected)child.send({progressAck:message.progressId},()=>{});}); if ('result' in message) {
                        job.result = message.result;
                        job.status = 'complete';
                        persist();
                    } if (message.error) {
                        job.error = message.error;
                        job.status = 'failed';
                        persist();
                    } });
                    child.once('error', error => { job.status = 'failed'; job.error = String(error); persist(); });
                    child.once('exit', () => { if (job.status === 'running') {
                        job.status = 'unknown';
                        job.error = 'Worker exited without a result; inspect effects before retrying.';
                    } delete job.child; persist().then(()=>jobs.delete(job.id),()=>{}); });
                    child.send({ operation: config.operation, payload: config.payload, root: directory });
                    return respond(res, 202, { id: job.id, status: job.status });
                }
                const match = /^\/api\/studio\/jobs\/([\w-]+)(\/cancel)?$/.exec(url.pathname);
                if (match) {
                    let job = jobs.get(match[1]);
                    if (!job) {
                        try {
                            job = JSON.parse(await readFile(join(dataRoot, `${match[1]}.json`), 'utf8'));
                            if (job.status === 'running')
                                job.status = 'unknown';
                        }
                        catch { }
                    }
                    if (!job || job.session !== session)
                        return respond(res, 404, { error: 'Unknown job' });
                    if (match[2] && req.method === 'POST' && job.child) {
                        job.status = 'unknown';
                        job.error = 'Cancelled: effects may have occurred.';
                        try {
                            if (process.platform === 'win32')
                                job.child.kill('SIGTERM');
                            else
                                process.kill(-job.child.pid, 'SIGTERM');
                        }
                        catch (error) {
                            if (error.code !== 'ESRCH')
                                throw error;
                        }
                        await journal(job);
                    }
                    const { child, ...record } = job;
                    return respond(res, 200, record);
                }
                return respond(res, 404, { error: 'Unknown endpoint' });
            }
            let file;
            const asset = /^\/studio-assets\/([\w-]+)\/(asset-[\w-]+\.mp4)$/.exec(url.pathname);
            if (asset) {
                if (!sessions.has(asset[1]))
                    return respond(res, 404, { error: 'Unknown session' });
                file = join(dataRoot, asset[1], 'assets', asset[2]);
            }
            else {
                const pathname = url.pathname === '/' ? '/ts-host/studio/' : decodeURIComponent(url.pathname);
                if (!pathname.startsWith('/ts-host/') && !pathname.startsWith('/models/'))
                    throw new Error('Not a studio asset');
                file = resolve(root, '.' + pathname, pathname.endsWith('/') ? 'index.html' : '');
                if (!['ts-host', 'models'].some(folder => file.startsWith(join(root, folder) + sep)))
                    throw new Error('Invalid asset path');
            }
            const info = await stat(file);
            if (!info.isFile())
                throw new Error('Not a file');
            const range = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range ?? '');
            const start = range ? Number(range[1]) : 0, end = range && range[2] ? Math.min(Number(range[2]), info.size - 1) : info.size - 1;
            if (start > end && info.size > 0) {
                res.writeHead(416);
                return res.end();
            }
            res.writeHead(range ? 206 : 200, { 'Content-Type': MIME[extname(file)] ?? 'text/plain; charset=utf-8', 'Content-Length': Math.max(0, end - start + 1), 'Accept-Ranges': 'bytes', 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp', ...(range ? { 'Content-Range': `bytes ${start}-${end}/${info.size}` } : {}) });
            if (req.method === 'HEAD' || info.size === 0)
                res.end();
            else
                createReadStream(file, { start, end }).on('error',error=>res.destroy(error)).pipe(res);
        }
        catch (error) {
            respond(res, error.code === 'ENOENT' ? 404 : 400, { error: String(error) });
        }
    });
    await new Promise(resolve => server.listen(port, '127.0.0.1', resolve));
    return { server, url: `http://127.0.0.1:${server.address().port}/ts-host/studio/`, async close() { for (const job of jobs.values())
            if (job.child) {
                try {
                    process.platform === 'win32' ? job.child.kill() : process.kill(-job.child.pid, 'SIGTERM');
                }
                catch { }
            } await new Promise(resolve => server.close(resolve)); } };
}
if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
    const port = Number(process.argv.find(a => a.startsWith('--port='))?.slice(7) ?? 8766);
    const app = await startStudio({ port });
    console.log(`Natlang Studio: ${app.url}`);
    for (const signal of ['SIGINT', 'SIGTERM'])
        process.once(signal, async () => { await app.close(); process.exit(0); });
}
