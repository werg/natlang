#!/usr/bin/env node
/** Minimal stdio MCP server for recording one declared Natlang action from OpenCode. */
import { appendFile } from 'node:fs/promises';

const toolName = 'submit_action';
const supportedProtocolVersions = ['2024-11-05'];
const logPath = process.env.NATLANG_OPENCODE_ACTION_LOG;
if (!logPath || !logPath.startsWith('/')) throw new Error('NATLANG_OPENCODE_ACTION_LOG must be an absolute path');
let buffer = '';

function reply(id, result) {
  if (id === undefined || id === null) return;
  process.stdout.write(`${JSON.stringify({ jsonrpc: '2.0', id, result })}\n`);
}
function error(id, code, message) {
  if (id === undefined || id === null) return;
  process.stdout.write(`${JSON.stringify({ jsonrpc: '2.0', id, error: { code, message } })}\n`);
}
async function handle(message) {
  const { id, method, params = {} } = message ?? {};
  if (method === 'initialize') {
    if (!supportedProtocolVersions.includes(params.protocolVersion)) {
      error(id, -32602, `unsupported MCP protocol version; supported: ${supportedProtocolVersions.join(', ')}`);
      return;
    }
    reply(id, { protocolVersion: params.protocolVersion, capabilities: { tools: {} },
      serverInfo: { name: 'natlang-action-bridge', version: '1.0.0' } });
    return;
  }
  if (method === 'notifications/initialized' || method === 'notifications/cancelled') return;
  if (method === 'ping') { reply(id, {}); return; }
  if (method === 'tools/list') {
    reply(id, { tools: [{ name: toolName,
      description: 'Record one declared Natlang action for the host; does not execute it.',
      inputSchema: { type: 'object', additionalProperties: false,
        properties: { name: { type: 'string', minLength: 1 }, arguments: { type: 'object' } },
        required: ['name', 'arguments'] } }] });
    return;
  }
  if (method === 'tools/call') {
    if (params.name !== toolName) { error(id, -32602, 'unknown MCP tool'); return; }
    const args = params.arguments;
    if (!args || typeof args !== 'object' || Array.isArray(args) ||
        typeof args.name !== 'string' || !args.name || !args.arguments || typeof args.arguments !== 'object' || Array.isArray(args.arguments)) {
      error(id, -32602, 'submit_action requires a nonempty action name and object arguments'); return;
    }
    const row = { at: new Date().toISOString(), name: args.name, arguments: args.arguments };
    const encoded = `${JSON.stringify(row)}\n`;
    if (Buffer.byteLength(encoded) > 64 * 1024) { error(id, -32602, 'action exceeds size limit'); return; }
    await appendFile(logPath, encoded, { flag: 'a', mode: 0o600 });
    reply(id, { content: [{ type: 'text', text: 'ACTION_RECORDED' }], isError: false });
    return;
  }
  error(id, -32601, 'method not found');
}
process.stdin.setEncoding('utf8');
for await (const chunk of process.stdin) {
  buffer += chunk;
  if (buffer.length > 1024 * 1024) { process.stderr.write('MCP input buffer limit exceeded\n'); process.exitCode = 1; break; }
  let newline;
  while ((newline = buffer.indexOf('\n')) >= 0) {
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (!line) continue;
    try { await handle(JSON.parse(line)); }
    catch { process.stderr.write('MCP request failed\n'); }
  }
}
