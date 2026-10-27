#!/usr/bin/env node
/**
 * Local server for the website with in-place copy editing.
 *
 *   node scripts/website-dev/server.mjs [--root website] [--page index.html] [--port 4180] [--host 127.0.0.1]
 *                                       [--open] [--read-only]
 *
 * It serves the site, reloads the browser when a file in it changes (stylesheets are swapped without a reload), and,
 * unless --read-only, adds an editing toolbar to the page. Edited text is written back into the page's HTML file on
 * disk, so the result is an ordinary diff to commit. Only text elements are editable: headings, paragraphs, list
 * items, captions and labels (see html-targets.mjs). Saving is refused when the file changed on disk since the page
 * was loaded.
 */
import { createServer } from 'node:http';
import { createHash } from 'node:crypto';
import { readFile, writeFile, rename, stat } from 'node:fs/promises';
import { watch } from 'node:fs';
import { spawn } from 'node:child_process';
import { dirname, extname, join, normalize, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { annotate, applyEdits, findEditableTargets } from './html-targets.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..', '..');

const HELP = `Serve the website locally and edit its text in place.

Usage: node scripts/website-dev/server.mjs [options]

Options:
  --root <dir>     Directory to serve (default: website)
  --page <file>    Page whose text is editable, relative to the root (default: index.html)
  --port <n>       Port to listen on (default: 4180; 0 picks a free port)
  --host <addr>    Address to bind (default: 127.0.0.1)
  --open           Open the page in the default browser
  --read-only      Serve with live reload but without the editing toolbar
  --help           Show this help
`;

export function parseArgs(argv) {
  const opts = { root: join(REPO, 'website'), page: 'index.html', port: 4180, host: '127.0.0.1', open: false, readOnly: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const value = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      return argv[++i];
    };
    if (a === '--root') opts.root = resolve(value());
    else if (a === '--page') opts.page = value();
    else if (a === '--port') opts.port = Number(value());
    else if (a === '--host') opts.host = value();
    else if (a === '--open') opts.open = true;
    else if (a === '--read-only') opts.readOnly = true;
    else if (a === '--help' || a === '-h') opts.help = true;
    else throw new Error(`unknown option ${a}; see --help`);
  }
  if (!Number.isInteger(opts.port) || opts.port < 0 || opts.port > 65535) throw new Error('--port must be an integer from 0 to 65535');
  return opts;
}

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.webp': 'image/webp',
  '.ico': 'image/x-icon', '.woff2': 'font/woff2', '.woff': 'font/woff', '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
};

const sha = text => createHash('sha256').update(text).digest('hex').slice(0, 16);

/** Start the server; resolves to { server, url, close }. */
export async function startServer(options) {
  const opts = { ...options };
  const pagePath = resolve(opts.root, opts.page);
  if (!pagePath.startsWith(resolve(opts.root) + sep)) throw new Error('--page must be inside --root');
  await stat(pagePath);

  const clients = new Set();
  const broadcast = (event, data) => {
    const msg = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
    for (const res of clients) res.write(msg);
  };

  // Reload on change. The site is a flat folder; a non-recursive watch works on every platform.
  const pending = new Map();
  const watcher = watch(opts.root, (_type, name) => {
    if (!name || name.startsWith('.') || name.endsWith('~') || name.includes('.tmp-')) return;
    clearTimeout(pending.get(name));
    pending.set(name, setTimeout(async () => {
      pending.delete(name);
      let version = null;
      if (resolve(opts.root, name) === pagePath) {
        try { version = sha(await readFile(pagePath, 'utf8')); } catch { return; }
      }
      broadcast('change', { file: name, kind: extname(name) === '.css' ? 'css' : 'reload', version });
    }, 120));
  });
  const heartbeat = setInterval(() => { for (const res of clients) res.write(': keep-alive\n\n'); }, 25000);

  const sendJSON = (res, status, body) => {
    res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
    res.end(JSON.stringify(body));
  };

  async function servePage(res) {
    const source = await readFile(pagePath, 'utf8');
    let html = source;
    const inject = [`<script>window.__NL_DEV__ = ${JSON.stringify({ version: sha(source), page: opts.page, editable: !opts.readOnly })};</script>`];
    if (!opts.readOnly) {
      html = annotate(source, findEditableTargets(source));
      inject.push('<link rel="stylesheet" href="/__dev/editor.css">');
    }
    inject.push('<script src="/__dev/editor.js" defer></script>');
    const at = html.lastIndexOf('</body>');
    html = at < 0 ? html + inject.join('\n') : html.slice(0, at) + inject.join('\n') + '\n' + html.slice(at);
    res.writeHead(200, { 'content-type': TYPES['.html'], 'cache-control': 'no-store' });
    res.end(html);
  }

  async function save(req, res) {
    if (opts.readOnly) return sendJSON(res, 403, { error: 'The server was started with --read-only.' });
    // Only the page served by this server may save: browsers send Origin on POST, and other sites cannot set the header.
    const origin = req.headers.origin;
    if (req.headers['x-natlang-dev'] !== '1' || (origin && origin !== `http://${req.headers.host}`)) {
      return sendJSON(res, 403, { error: 'Saving is only allowed from the page this server serves.' });
    }
    const chunks = [];
    let bytes = 0;
    for await (const chunk of req) {
      bytes += chunk.length;
      if (bytes > 2_000_000) return sendJSON(res, 413, { error: 'The request is too large.' });
      chunks.push(chunk);
    }
    let parsed;
    try { parsed = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks))); } catch { return sendJSON(res, 400, { error: 'The request is not valid JSON.' }); }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return sendJSON(res, 400, { error: 'The request must be an object.' });
    const { version, edits } = parsed;
    if (!Array.isArray(edits) || !edits.length) return sendJSON(res, 400, { error: 'There are no edits to save.' });
    const source = await readFile(pagePath, 'utf8');
    if (sha(source) !== version) {
      return sendJSON(res, 409, { error: `${opts.page} changed on disk after this page was loaded. Reload, then make your edits again.`, version: sha(source) });
    }
    let next;
    try { next = applyEdits(source, edits); } catch (e) { return sendJSON(res, 422, { error: e.message }); }
    const tmp = `${pagePath}.tmp-${process.pid}`;
    await writeFile(tmp, next, 'utf8');
    await rename(tmp, pagePath);
    const saved = sha(next);
    console.log(`saved ${edits.length} edit${edits.length === 1 ? '' : 's'} to ${relative(process.cwd(), pagePath)}`);
    sendJSON(res, 200, { version: saved, saved: edits.length });
  }

  let pendingSave = Promise.resolve();
  const server = createServer(async (req, res) => {
    try {
      // Check authority against the bound interface, before serving pages or accepting edits.
      // An attacker's DNS name can resolve to loopback without becoming a trusted Host.
      const hosts = new Set([opts.host, req.socket.localAddress]);
      if (['127.0.0.1', '::1', '0.0.0.0', '::'].includes(opts.host)) hosts.add('localhost');
      let trusted = false;
      try {
        const authority = new URL('http://' + req.headers.host);
        trusted = !authority.username && !authority.password && hosts.has(authority.hostname.replace(/^\[|\]$/g, ''))
          && Number(authority.port || 80) === server.address().port;
      } catch { /* invalid authority */ }
      if (!trusted) return sendJSON(res, 403, { error: 'Untrusted Host header.' });
      const url = new URL(req.url, 'http://localhost');
      if (url.pathname === '/__dev/events') {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-store', connection: 'keep-alive' });
        res.write(': connected\n\n');
        clients.add(res);
        req.on('close', () => clients.delete(res));
        return;
      }
      if (url.pathname === '/__dev/save' && req.method === 'POST') {
        // Serialize version checks and atomic replacement across tabs.
        const saving = pendingSave.then(() => save(req, res));
        pendingSave = saving.catch(() => {});
        return await saving;
      }
      if (url.pathname === '/__dev/editor.js' || url.pathname === '/__dev/editor.css') {
        const file = join(HERE, url.pathname.slice('/__dev/'.length));
        res.writeHead(200, { 'content-type': TYPES[extname(file)], 'cache-control': 'no-store' });
        return res.end(await readFile(file));
      }
      if (req.method !== 'GET' && req.method !== 'HEAD') return sendJSON(res, 405, { error: 'Method not allowed.' });

      let rel = decodeURIComponent(url.pathname);
      if (rel.endsWith('/')) rel += 'index.html';
      const file = normalize(join(opts.root, rel));
      if (!file.startsWith(resolve(opts.root) + sep) || rel.split('/').some(part => part.startsWith('.'))) {
        return sendJSON(res, 404, { error: 'Not found.' });
      }
      if (file === pagePath) return await servePage(res);
      const data = await readFile(file).catch(() => null);
      if (!data) return sendJSON(res, 404, { error: 'Not found.' });
      res.writeHead(200, { 'content-type': TYPES[extname(file)] || 'application/octet-stream', 'cache-control': 'no-store' });
      res.end(req.method === 'HEAD' ? undefined : data);
    } catch (e) {
      console.error(e);
      if (!res.headersSent) sendJSON(res, 500, { error: String(e.message || e) });
      else res.end();
    }
  });

  await new Promise((ok, fail) => { server.once('error', fail); server.listen(opts.port, opts.host, ok); });
  const { port } = server.address();
  const url = `http://${opts.host.includes(':') ? `[${opts.host}]` : opts.host}:${port}/${opts.page === 'index.html' ? '' : opts.page}`;
  const close = () => new Promise(ok => {
    watcher.close();
    clearInterval(heartbeat);
    for (const res of clients) res.end();
    server.close(() => ok());
  });
  return { server, url, close };
}

function openBrowser(url) {
  const cmd = process.platform === 'darwin' ? 'open' : process.platform === 'win32' ? 'cmd' : 'xdg-open';
  const args = process.platform === 'win32' ? ['/c', 'start', '', url] : [url];
  try { spawn(cmd, args, { stdio: 'ignore', detached: true }).on('error', () => {}).unref(); } catch { /* no browser available */ }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  let opts;
  try { opts = parseArgs(process.argv.slice(2)); } catch (e) { console.error(e.message); process.exit(2); }
  if (opts.help) { process.stdout.write(HELP); process.exit(0); }
  const { url, close } = await startServer(opts);
  console.log(`Serving ${relative(process.cwd(), opts.root) || '.'} at ${url}`);
  console.log(opts.readOnly ? 'Read-only: live reload is on, editing is off.' : `Editing is on: saved text is written to ${relative(process.cwd(), resolve(opts.root, opts.page))}.`);
  if (opts.open) openBrowser(url);
  const stop = () => close().then(() => process.exit(0));
  process.on('SIGINT', stop);
  process.on('SIGTERM', stop);
}
