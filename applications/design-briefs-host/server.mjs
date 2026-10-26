/** Authenticated transport of the declared per-program APIs. No domain routing decisions live here. */
import { createServer } from 'node:http';
import { createHash, timingSafeEqual } from 'node:crypto';
import { ApplicationError } from './runtime.mjs';

export function authenticateBearer(header, credentials) {
  if (typeof header !== 'string' || !header.startsWith('Bearer ') || header.length > 4096) throw new ApplicationError('unauthorized');
  const tokenHash = createHash('sha256').update(header.slice(7)).digest();
  const found = credentials.find(item => /^[a-f0-9]{64}$/.test(item.tokenHash) && timingSafeEqual(tokenHash, Buffer.from(item.tokenHash, 'hex')));
  if (!found) throw new ApplicationError('unauthorized');
  return structuredClone(found.principal);
}
async function readBody(request, maxBytes) {
  if ((request.headers['content-type'] ?? '').split(';')[0].trim().toLowerCase() !== 'application/json')
    throw new ApplicationError('json_content_type_required');
  let size = 0; const chunks = [];
  for await (const chunk of request) {
    size += chunk.length; if (size > maxBytes) throw new ApplicationError('payload_too_large'); chunks.push(chunk);
  }
  try {
    const value = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error();
    return value;
  } catch { throw new ApplicationError('invalid_json'); }
}
const statusFor = code => code === 'unauthorized' ? 401 : code === 'forbidden' ? 403 :
  /not_found$/.test(code) ? 404 : code === 'payload_too_large' ? 413 :
  /stale|conflict|lifecycle|binding|separation|already_exists|override/.test(code) ? 409 :
  /^storage|model_configuration/.test(code) ? 503 : 400;

export async function serveApplications({ service, credentials, port = 0, host = '127.0.0.1', maxBodyBytes = 2_000_000 }) {
  if (!Array.isArray(credentials) || !credentials.length) throw new ApplicationError('credentials_required');
  if (!Number.isSafeInteger(maxBodyBytes) || maxBodyBytes < 1) throw new ApplicationError('invalid_configuration');
  const controllers = new Set();
  const server = createServer(async (request, response) => {
    const controller = new AbortController(); controllers.add(controller);
    response.on('close', () => { if (!response.writableEnded) controller.abort(); });
    const reply = (status, value) => {
      response.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store',
        'x-content-type-options': 'nosniff', 'content-security-policy': "default-src 'none'; frame-ancestors 'none'" });
      response.end(JSON.stringify(value));
    };
    try {
      const url = new URL(request.url ?? '/', 'http://native-applications');
      if (url.pathname === '/health' && request.method === 'GET') return reply(200, { ready: true });
      const principal = authenticateBearer(request.headers.authorization, credentials);
      const parts = url.pathname.split('/').filter(Boolean).map(decodeURIComponent);
      const body = async () => readBody(request, maxBodyBytes);
      if (parts[0] === 'applications') {
        if (parts.length === 1 && request.method === 'GET') return reply(200, service.list(principal));
        const appId = parts[1];
        if (parts.length === 2 && request.method === 'GET') return reply(200, service.describe(principal, appId));
        if (parts[2] === 'cases') {
          const caseId = parts[3];
          if (parts.length === 3 && request.method === 'GET') return reply(200, service.cases(principal, appId, {
            after: url.searchParams.get('after') ?? '', limit: Number(url.searchParams.get('limit') ?? 50) }));
          if (parts.length === 4) {
            if (request.method === 'GET') { const record = service.read(principal, appId, caseId); return reply(record ? 200 : 404, record ?? { error: 'not_found' }); }
            if (request.method === 'POST') return reply(201, service.create(principal, appId, caseId, await body()));
            if (request.method === 'PUT') return reply(200, service.replace(principal, appId, caseId, await body()));
          }
          if (parts.length === 5 && parts[4] === 'invoke' && request.method === 'POST')
            return reply(200, await service.invoke(principal, appId, caseId, await body(), { signal: controller.signal }));
          if (parts.length === 5 && parts[4] === 'events' && request.method === 'GET')
            return reply(200, service.events(principal, appId, caseId, { after: Number(url.searchParams.get('after') ?? 0), limit: Number(url.searchParams.get('limit') ?? 50) }));
          if (parts[4] === 'proposals' && parts[5]) {
            if (parts.length === 6 && request.method === 'GET') return reply(200, service.proposal(principal, appId, caseId, parts[5]));
            if (parts.length === 7 && ['approve','commit','reject'].includes(parts[6]) && request.method === 'POST')
              return reply(200, service[parts[6]](principal, appId, caseId, parts[5], await body()));
          }
        }
      }
      if (parts[0] === 'outbox' && parts.length === 2 && request.method === 'GET') return reply(200, service.outbox(principal, Number(parts[1])));
      if (parts[0] === 'outbox' && parts.length === 3 && parts[2] === 'reconcile' && request.method === 'POST')
        return reply(200, service.reconcile(principal, Number(parts[1]), await body()));
      reply(404, { error: 'not_found' });
    } catch (e) {
      const code = typeof e.code === 'string' && /^[a-z_]+$/.test(e.code) ? e.code : 'invocation_failed';
      if (!response.headersSent) reply(statusFor(code), { error: code }); else response.destroy();
    } finally { controllers.delete(controller); }
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(port, host, resolve); });
  const address = server.address();
  return { server, url: `http://${host}:${address.port}`, async close() {
    for (const controller of controllers) controller.abort();
    await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  } };
}
