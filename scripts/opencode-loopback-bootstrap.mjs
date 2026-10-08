#!/usr/bin/env node
/** Start the official OpenCode SDK server and the bounded Natlang loopback adapter. */
import { access, mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { constants as fsConstants } from 'node:fs';
import { dirname, resolve, basename, delimiter } from 'node:path';
import { pathToFileURL } from 'node:url';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { createHash } from 'node:crypto';
import { createOpenCodeLoopbackChatAdapter } from './opencode-loopback-chat-adapter.mjs';

function parseArgs(argv) {
  const values = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith('--') || i + 1 >= argv.length) throw new Error(`invalid argument near ${token}`);
    const key = token.slice(2);
    if (!['sdk-module', 'client-bin', 'out', 'model', 'max-request-ms', 'max-concurrency'].includes(key))
      throw new Error(`unsupported option --${key}`);
    if (Object.hasOwn(values, key)) throw new Error(`duplicate option --${key}`);
    values[key] = argv[++i];
  }
  for (const key of ['sdk-module', 'client-bin', 'out'])
    if (!values[key]) throw new Error(`--${key} is required`);
  values.model ??= 'exo-free';
  values['max-request-ms'] = Number(values['max-request-ms'] ?? 180_000);
  if (!Number.isSafeInteger(values['max-request-ms']) || values['max-request-ms'] < 1)
    throw new Error('--max-request-ms must be a positive integer');
  values['max-concurrency'] = Number(values['max-concurrency'] ?? 1);
  if (!Number.isSafeInteger(values['max-concurrency']) || values['max-concurrency'] < 1 || values['max-concurrency'] > 8)
    throw new Error('--max-concurrency must be an integer from 1 to 8');
  if (!/^[a-z0-9][a-z0-9._-]*-free$/i.test(values.model))
    throw new Error('--model must name a free OpenCode model, such as exo-free');
  return values;
}

async function requireFile(path, label) {
  const absolute = resolve(path);
  try {
    await access(absolute, fsConstants.R_OK);
    if (!(await stat(absolute)).isFile()) throw new Error('not a file');
  }
  catch { throw new Error(`${label} is not a readable file`); }
  return absolute;
}

async function sha256File(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

function raceAbort(promise, signal, label) {
  if (signal.aborted) {
    Promise.resolve(promise).catch(() => {});
    return Promise.reject(signal.reason ?? new Error(`${label} aborted`));
  }
  let onAbort;
  const aborted = new Promise((_, reject) => {
    onAbort = () => reject(signal.reason ?? new Error(`${label} aborted`));
    signal.addEventListener('abort', onAbort, { once: true });
  });
  return Promise.race([Promise.resolve(promise), aborted]).finally(() => signal.removeEventListener('abort', onAbort));
}

function safeDiagnosticText(value, maxLength = 240) {
  if (typeof value !== 'string') return '';
  return value.replace(/authorization\s*[:=]\s*(?:bearer|basic)\s+[^\s,;]+/gi, 'Authorization=[redacted]')
    .replace(/\b(?:sk|pk|rk|sess|tok)[-_][A-Za-z0-9_-]{12,}\b/g, '[redacted]')
    .replace(/(authorization|api[-_]?key|token|secret)\s*[:=]\s*[^\s,;]+/gi, '$1=[redacted]')
    .replace(/\s+/g, ' ').slice(0, maxLength);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const sdkModule = await requireFile(args['sdk-module'], 'SDK module');
  const clientBin = await requireFile(args['client-bin'], 'OpenCode client binary');
  if (basename(clientBin) !== 'opencode') throw new Error('--client-bin must point to the official opencode executable');
  await access(clientBin, fsConstants.X_OK);
  if (!process.env.OPENCODE_API_KEY) throw new Error('OPENCODE_API_KEY must be present in the environment');

  const output = resolve(args.out);
  try { await mkdir(output, { recursive: false }); }
  catch (error) {
    if (error?.code !== 'EEXIST' || !(await stat(output).catch(() => null))?.isDirectory() ||
        (await readdir(output)).length !== 0)
      throw new Error('--out must be a new directory or an existing empty directory');
  }
  const scratch = resolve(output, 'scratch');
  await mkdir(scratch, { recursive: false });
  const isolatedHome = resolve(output, 'opencode-home');
  const isolatedConfig = resolve(isolatedHome, 'config');
  const isolatedData = resolve(isolatedHome, 'data');
  const isolatedCache = resolve(isolatedHome, 'cache');
  const isolatedState = resolve(isolatedHome, 'state');
  for (const path of [isolatedConfig, isolatedData, isolatedCache, isolatedState])
    await mkdir(path, { recursive: true, mode: 0o700 });
  const actionLog = resolve(output, 'action-mcp-calls.jsonl');
  const handshakeLog = resolve(output, 'mcp-handshake.jsonl');
  const actionServer = resolve(dirname(fileURLToPath(import.meta.url)), 'opencode-natlang-action-mcp-server.mjs');
  process.env.PATH = `${dirname(clientBin)}${delimiter}${process.env.PATH ?? ''}`;

  let official;
  let adapter;
  let stopping;
  const startupController = new AbortController();
  let signalStopResolve;
  const signalStopped = new Promise(resolveStopped => { signalStopResolve = resolveStopped; });
  const configReceipt = {
    schema: 'natlang.opencode-loopback-bootstrap/1',
    bootstrap_id: randomUUID(),
    started_at: new Date().toISOString(),
    official_sdk_module: sdkModule,
    official_sdk_module_sha256: await sha256File(sdkModule),
    official_client_binary: clientBin,
    official_client_binary_sha256: await sha256File(clientBin),
    provider_id: 'opencode',
    model_id: args.model,
    model_alias: `opencode/${args.model}`,
    credential_source: 'OPENCODE_API_KEY environment variable; secret excluded',
    server_bind: { hostname: '127.0.0.1', port: 0 },
    adapter_bind: { host: '127.0.0.1', port: 'ephemeral' },
    scratch_directory: scratch,
    isolated_opencode_home: isolatedHome,
    user_config_and_plugins: 'isolated XDG and OPENCODE_CONFIG_DIR paths; project config disabled',
    output_directory: output,
    max_concurrency: args['max-concurrency'],
    max_request_ms: args['max-request-ms'],
    response_mode: 'buffered JSON, including when stream=true',
    native_provider_tool_calls: false,
    native_opencode_tool_policy: 'official SDK default inventory retained; session permission wildcard ask; Natlang action MCP submit_action allowed; observed other permission requests rejected and session history audited after the turn; native execution prevention is not established',
    natlang_action_mcp: { server_id: 'natlang_action_bridge', tool_id: 'submit_action', action_log: actionLog,
      handshake_log: handshakeLog, server_script: actionServer,
      execution: 'records action only; no Natlang tool execution or host I/O' },
    incremental_token_streaming: false,
    provider_availability: 'not-probed',
    training_admission: false
  };

  const writeLifecycle = async status => {
    await writeFile(resolve(output, 'lifecycle.json'), JSON.stringify({
      bootstrap_id: configReceipt.bootstrap_id, status, observed_at: new Date().toISOString(),
      provider_availability: 'not-probed', training_admission: false
    }, null, 2) + '\n');
  };
  const stop = () => stopping ??= (async () => {
    try { await writeLifecycle('stopping'); } catch { /* close processes even if receipt storage fails */ }
    try { await adapter?.close(); } catch { /* still close the official server */ }
    try { official?.server.close(); } catch { /* shutdown is best effort */ }
    try { await writeLifecycle('stopped'); } catch { /* shutdown is complete */ }
  })();
  const onSignal = () => {
    if (!startupController.signal.aborted)
      startupController.abort(new Error('bootstrap received a shutdown signal'));
    signalStopResolve();
    void stop();
  };
  process.on('SIGINT', onSignal);
  process.on('SIGTERM', onSignal);

  try {
    process.chdir(scratch);
    // OpenCode derives global config/data paths from the process environment
    // during module initialization. Point all of them at this run and disable
    // upward project config discovery so user plugins and MCP servers cannot
    // silently enter the SDK tool registry.
    process.env.HOME = isolatedHome;
    process.env.OPENCODE_TEST_HOME = isolatedHome;
    process.env.XDG_CONFIG_HOME = isolatedConfig;
    process.env.XDG_DATA_HOME = isolatedData;
    process.env.XDG_CACHE_HOME = isolatedCache;
    process.env.XDG_STATE_HOME = isolatedState;
    process.env.OPENCODE_CONFIG_DIR = resolve(isolatedConfig, 'opencode');
    process.env.OPENCODE_DISABLE_PROJECT_CONFIG = '1';
    startupController.signal.throwIfAborted();
    const { createOpencode } = await import(pathToFileURL(sdkModule).href);
    if (typeof createOpencode !== 'function') throw new Error('SDK module does not export createOpencode');
    const startup = createOpencode({ hostname: '127.0.0.1', port: 0, timeout: 20_000,
      signal: startupController.signal,
      config: { provider: { opencode: { options: { apiKey: '{env:OPENCODE_API_KEY}' } } },
        mcp: { natlang_action_bridge: { type: 'local', command: [process.execPath, actionServer],
          cwd: scratch, environment: { NATLANG_OPENCODE_ACTION_LOG: actionLog,
            NATLANG_OPENCODE_MCP_HANDSHAKE_LOG: handshakeLog }, enabled: true } },
        share: 'disabled', autoupdate: false } });
    try {
      official = await raceAbort(startup, startupController.signal, 'OpenCode SDK startup');
    } catch (error) {
      void Promise.resolve(startup).then(lateServer => lateServer?.server?.close(), () => {});
      throw error;
    }
    startupController.signal.throwIfAborted();
    if (!official?.client || !official?.server?.close) throw new Error('SDK did not return a client and server handle');
    let mcpReadiness = { status: 'unverified' };
    try {
      const statusResult = await official.client.mcp.status({ directory: scratch });
      if (statusResult?.error) throw new Error('official SDK MCP status request failed');
      let statuses = statusResult?.data;
      if (!statuses || typeof statuses !== 'object') throw new Error('official SDK MCP status returned no map');
      if (statuses.natlang_action_bridge?.status !== 'connected') {
        const connectResult = await official.client.mcp.connect({ name: 'natlang_action_bridge', directory: scratch });
        if (connectResult?.error) throw new Error('official SDK MCP connect request failed');
        const afterConnect = await official.client.mcp.status({ directory: scratch });
        if (afterConnect?.error) throw new Error('official SDK MCP status retry failed');
        statuses = afterConnect?.data;
      }
      const handshakeRows = (await readFile(handshakeLog, 'utf8').catch(() => '')).split('\n').filter(Boolean)
        .map(line => { try { return JSON.parse(line); } catch { return null; } }).filter(Boolean);
      const toolsList = handshakeRows.filter(row => row.method === 'tools/list').flatMap(row =>
        Array.isArray(row.tool_ids) ? row.tool_ids : []);
      const connected = statuses?.natlang_action_bridge?.status === 'connected';
      mcpReadiness = { server: statuses?.natlang_action_bridge?.status ?? 'missing',
        server_tool_names: toolsList,
        effective_session_tool_id: 'natlang_action_bridge_submit_action',
        tool_id_registered: connected && toolsList.includes('submit_action'),
        inventory_source: 'official SDK MCP status and stdio tools/list handshake; pinned OpenCode MCP catalog naming' };
      if (statuses?.natlang_action_bridge?.status === 'failed')
        mcpReadiness.error = safeDiagnosticText(statuses.natlang_action_bridge.error);
    } catch (error) {
      mcpReadiness = { status: 'error', error: safeDiagnosticText(error?.message) || 'SDK MCP readiness check failed' };
    }
    if (mcpReadiness.tool_id_registered !== true)
      throw new Error('configured Natlang action MCP is not connected with the expected tool list');
    adapter = await createOpenCodeLoopbackChatAdapter({ client: official.client, providerID: 'opencode',
      modelID: args.model, directory: scratch, maxConcurrency: args['max-concurrency'], maxRequestMs: args['max-request-ms'] });
    startupController.signal.throwIfAborted();
    const config = Object.freeze({ ...configReceipt, adapter_bind: { host: adapter.config.host, port: adapter.config.port },
      provider_availability: 'not-probed', mcp_readiness: mcpReadiness });
    await writeFile(resolve(output, 'bootstrap-config.json'), JSON.stringify(config, null, 2) + '\n', { flag: 'wx' });
    await writeLifecycle('listening-provider-not-probed');
    process.stdout.write(JSON.stringify({ endpoint: adapter.url, model: config.model_alias,
      provider_availability: 'not-probed', config_receipt: resolve(output, 'bootstrap-config.json') }) + '\n');
    await signalStopped;
    await stopping;
  } catch (error) {
    await stop();
    // A shutdown can race with the adapter's asynchronous listen operation.
    // Close any handle that finished initializing after the first stop pass.
    try { await adapter?.close(); } catch { /* shutdown is best effort */ }
    try { official?.server.close(); } catch { /* shutdown is best effort */ }
    const detail = safeDiagnosticText(error?.message);
    process.stderr.write(`OpenCode loopback bootstrap failed (${error?.name ?? 'Error'}${detail ? `: ${detail}` : ''}).\n`);
    process.exitCode = 1;
  } finally {
    process.removeListener('SIGINT', onSignal);
    process.removeListener('SIGTERM', onSignal);
  }
}

main().catch(error => {
  const detail = safeDiagnosticText(error?.message);
  process.stderr.write(`OpenCode loopback bootstrap failed (${error?.name ?? 'Error'}${detail ? `: ${detail}` : ''}).\n`);
  process.exitCode = 1;
});
