#!/usr/bin/env node
/** Start isolated official OpenCode CLI server plus the Natlang CLI-backed bridge. */
import { access, appendFile, mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { resolve, dirname } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { randomUUID, createHash } from 'node:crypto';
import { createOpenCodeCliChatAdapter } from './opencode-cli-chat-adapter.mjs';

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    if (!key.startsWith('--') || i + 1 >= argv.length || Object.hasOwn(out, key)) throw new Error(`invalid argument ${key}`);
    out[key] = argv[++i];
  }
  for (const key of ['--sdk-module', '--client-bin', '--out']) if (!out[key]) throw new Error(`${key} is required`);
  out['--model'] ??= 'ling-3.1-flash-free';
  out['--max-request-ms'] = Number(out['--max-request-ms'] ?? 180_000);
  if (!Number.isSafeInteger(out['--max-request-ms']) || out['--max-request-ms'] < 1) throw new Error('--max-request-ms must be positive');
  return out;
}

const shaFile = async path => createHash('sha256').update(await readFile(path)).digest('hex');
const safe = value => String(value ?? '').replace(/Bearer\s+[^\s,;]+/gi, 'Bearer [redacted]')
  .replace(/\b(?:sk|rk|tok)[-_][A-Za-z0-9_-]{12,}\b/g, '[redacted]').slice(0, 300);
const writeJson = (path, value, flag = 'w') => writeFile(path, JSON.stringify(value, null, 2) + '\n', { flag, mode: 0o600 });

// Zen's live catalog can outpace OpenCode's bundled Models.dev catalog. Pin
// this documented OpenAI-compatible free model explicitly, and use it for
// OpenCode's background title task as well so no paid small-model default can
// be selected by an otherwise isolated CLI session.
export function buildFreeModelConfig(actionServer, actionLog, handshakeLog, modelID = 'step-5-preview-free') {
  const step5 = modelID === 'step-5-preview-free';
  const providerID = step5 ? 'zen-step5-free' : 'opencode';
  const modelAlias = `${providerID}/${modelID}`;
  const config = {
    '$schema': 'https://opencode.ai/config.json',
    model: modelAlias,
    small_model: modelAlias,
    // OpenCode's supported permission rules hide denied built-ins from the
    // model-facing tool list. The named Natlang MCP action remains available.
    permission: { '*': 'deny', natlang_action_bridge_submit_action: 'allow' },
    mcp: { natlang_action_bridge: { type: 'local', command: [process.execPath, actionServer], enabled: true,
      environment: { NATLANG_OPENCODE_ACTION_LOG: actionLog, NATLANG_OPENCODE_MCP_HANDSHAKE_LOG: handshakeLog } } },
    share: 'disabled', autoupdate: false
  };
  if (step5) {
    config.provider = {
      'zen-step5-free': {
        npm: '@ai-sdk/openai-compatible',
        name: 'OpenCode Zen Step 5 Preview Free',
        options: {
          baseURL: 'https://opencode.ai/zen/v1',
          apiKey: '{env:OPENCODE_API_KEY}'
        },
        models: {
          'step-5-preview-free': {
            name: 'Step 5 Preview Free'
          }
        }
      }
    };
  }
  return config;
}

export function buildToolSurfaceReceipt(installedToolIDs) {
  return {
    installed_tool_ids: [...installedToolIDs],
    model_tool_surface: { native_tools: 'disabled and omitted from model-facing requests by wildcard deny',
      natlang_action_bridge_submit_action: 'allowed; the only enabled action tool' },
    permission_policy: 'official wildcard deny disables and hides native built-in tools; exact Natlang action MCP tool allowed; permission requests outside this tool surface are rejected; session history is not an execution barrier'
  };
}

async function reservePort() {
  const probe = createServer();
  await new Promise((resolveListen, reject) => probe.once('error', reject).listen(0, '127.0.0.1', resolveListen));
  const port = probe.address().port;
  await new Promise(resolveClose => probe.close(resolveClose));
  return port;
}

async function waitChildClose(child, milliseconds) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return true;
  return new Promise(resolveClose => {
    let timer;
    const finish = result => { clearTimeout(timer); child.off('close', onClose); resolveClose(result); };
    const onClose = () => finish(true);
    child.once('close', onClose);
    timer = setTimeout(() => finish(false), milliseconds);
  });
}
async function stopChild(child) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  child.kill('SIGTERM');
  if (await waitChildClose(child, 1500)) return;
  child.kill('SIGKILL');
  await waitChildClose(child, 1500);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const sdkModule = resolve(args['--sdk-module']);
  const cliPath = resolve(args['--client-bin']);
  const output = resolve(args['--out']);
  if (!process.env.OPENCODE_API_KEY) throw new Error('OPENCODE_API_KEY must be present in child environment');
  for (const [path, label] of [[sdkModule, 'SDK module'], [cliPath, 'official CLI']]) {
    await accessFile(path, label, path === cliPath);
  }
  try { await mkdir(output, { recursive: false, mode: 0o700 }); }
  catch (error) {
    if (error?.code !== 'EEXIST' || (await readdir(output)).length !== 0) throw new Error('--out must be new or empty');
  }
  const scratch = resolve(output, 'scratch');
  const isolatedHome = resolve(output, 'opencode-home');
  const isolatedConfig = resolve(isolatedHome, 'config');
  const isolatedData = resolve(isolatedHome, 'data');
  const isolatedCache = resolve(isolatedHome, 'cache');
  const isolatedState = resolve(isolatedHome, 'state');
  await mkdir(isolatedHome, { recursive: false, mode: 0o700 });
  for (const path of [scratch, isolatedConfig, isolatedData, isolatedCache, isolatedState])
    await mkdir(path, { recursive: false, mode: 0o700 });
  const actionLog = resolve(output, 'action-mcp-calls.jsonl');
  const handshakeLog = resolve(output, 'mcp-handshake.jsonl');
  const bootstrapLog = resolve(output, 'bootstrap-events.jsonl');
  const stage = async (name, details = {}) => appendFile(bootstrapLog, JSON.stringify({ at: new Date().toISOString(), stage: name, ...details }) + '\n', { mode: 0o600 });
  await writeFile(bootstrapLog, JSON.stringify({ at: new Date().toISOString(), stage: 'output-ready' }) + '\n', { flag: 'wx', mode: 0o600 });
  await writeFile(actionLog, '', { flag: 'wx', mode: 0o600 });
  await writeFile(handshakeLog, '', { flag: 'wx', mode: 0o600 });
  const actionServer = resolve(dirname(fileURLToPath(import.meta.url)), 'opencode-natlang-action-mcp-server.mjs');
  const config = buildFreeModelConfig(actionServer, actionLog, handshakeLog, args['--model']);
  const configPath = resolve(isolatedConfig, 'opencode', 'opencode.json');
  await mkdir(dirname(configPath), { recursive: true, mode: 0o700 });
  await writeJson(configPath, config, 'wx');
  const cliEnvironment = { ...process.env, HOME: isolatedHome, OPENCODE_TEST_HOME: isolatedHome,
    XDG_CONFIG_HOME: isolatedConfig, XDG_DATA_HOME: isolatedData, XDG_CACHE_HOME: isolatedCache,
    XDG_STATE_HOME: isolatedState, OPENCODE_CONFIG_DIR: dirname(configPath),
    OPENCODE_DISABLE_PROJECT_CONFIG: '1', OPENCODE_AUTOUPDATE: '0' };
  const serverPort = await reservePort();
  const serverUrl = `http://127.0.0.1:${serverPort}`;
  await stage('cli-version-check-started');
  const cliVersion = await new Promise((resolveVersion, reject) => {
    const child = spawn(cliPath, ['--version'], { cwd: scratch, env: cliEnvironment, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '', err = '';
    child.stdout.on('data', chunk => { out += chunk.toString(); });
    child.stderr.on('data', chunk => { err += chunk.toString(); });
    child.once('error', reject);
    child.once('close', code => code === 0 ? resolveVersion(out.trim() || err.trim()) : reject(new Error('official CLI version check failed')));
  });
  await stage('cli-version-checked', { version: cliVersion });
  const serverChild = spawn(cliPath, ['serve', '--hostname', '127.0.0.1', '--port', String(serverPort), '--pure'],
    { cwd: scratch, env: cliEnvironment, stdio: ['ignore', 'ignore', 'ignore'] });
  let adapter;
  let sdkClient;
  await stage('serve-spawned', { pid: serverChild.pid, port: serverPort });
  let stopping, resolveStopped;
  const stopped = new Promise(resolveStop => { resolveStopped = resolveStop; });
  const startupController = new AbortController();
  const lifecycle = async status => writeJson(resolve(output, 'lifecycle.json'), {
    bootstrap_id: bootstrapId, status, at: new Date().toISOString(), provider_availability: 'not-probed', training_admission: false });
  const bootstrapId = randomUUID();
  const shutdownPath = resolve(output, 'shutdown.request');
  const stop = () => stopping ??= (async () => {
    await lifecycle('stopping').catch(() => {});
    await adapter?.close().catch(() => {});
    await stopChild(serverChild);
    await lifecycle('stopped').catch(() => {});
    resolveStopped();
  })();
  const onSignal = () => { startupController.abort(); void stop(); };
  process.once('SIGINT', onSignal); process.once('SIGTERM', onSignal);
  const shutdownPoll = setInterval(() => {
    void stat(shutdownPath).then(() => { startupController.abort(new Error('shutdown requested by supervisor')); void stop(); }, () => {});
  }, 100);
  shutdownPoll.unref?.();
  try {
    process.chdir(scratch);
    process.env.HOME = isolatedHome; process.env.OPENCODE_TEST_HOME = isolatedHome;
    process.env.XDG_CONFIG_HOME = isolatedConfig; process.env.XDG_DATA_HOME = isolatedData;
    process.env.XDG_CACHE_HOME = isolatedCache; process.env.XDG_STATE_HOME = isolatedState;
    process.env.OPENCODE_CONFIG_DIR = dirname(configPath); process.env.OPENCODE_DISABLE_PROJECT_CONFIG = '1';
    const { createOpencodeClient } = await import(pathToFileURL(sdkModule).href);
    sdkClient = createOpencodeClient({ baseUrl: serverUrl, directory: scratch });
    const deadline = Date.now() + 25_000;
    let ready = false, healthAttempts = 0, lastHealthError;
    while (Date.now() < deadline && !startupController.signal.aborted) {
      if (serverChild.exitCode !== null || serverChild.signalCode !== null) throw new Error('official CLI server exited before readiness');
      try {
        const healthResponse = await fetch(`${serverUrl}/global/health`, { signal: AbortSignal.any([startupController.signal, AbortSignal.timeout(1500)]) });
        const health = await healthResponse.json();
        if (healthResponse.ok && health?.healthy === true) { ready = true; await stage('health-ready', { attempts: healthAttempts, http_status: healthResponse.status, version: health.version }); break; }
        lastHealthError = `HTTP ${healthResponse.status}; healthy=${health?.healthy}`;
      } catch (error) { lastHealthError = safe(error?.message); }
      healthAttempts++;
      if (healthAttempts === 1 || healthAttempts % 5 === 0) await stage('health-poll', { attempts: healthAttempts, error: lastHealthError });
      await new Promise(resolveWait => { const timer = setTimeout(resolveWait, 200); timer.unref?.(); });
    }
    if (!ready) throw new Error(startupController.signal.aborted ? 'bootstrap stopped during startup' : 'official OpenCode CLI server did not become healthy');
    await stage('mcp-status-started');
    const mcpStatus = await sdkClient.mcp.status({ directory: scratch });
    if (mcpStatus?.error) throw new Error('official SDK MCP status failed');
    let statuses = mcpStatus.data;
    if (statuses?.natlang_action_bridge?.status !== 'connected') {
      const connected = await sdkClient.mcp.connect({ name: 'natlang_action_bridge', directory: scratch });
      if (connected?.error) throw new Error('official SDK MCP connect failed');
      statuses = (await sdkClient.mcp.status({ directory: scratch })).data;
    }
    if (statuses?.natlang_action_bridge?.status !== 'connected') throw new Error('action MCP server is not connected');
    await stage('mcp-connected', { status: statuses.natlang_action_bridge.status });
    const defaultTools = await sdkClient.tool.ids({ directory: scratch });
    if (defaultTools?.error || !Array.isArray(defaultTools?.data) || !defaultTools.data.length)
      throw new Error('official default tool inventory is missing');
    await stage('default-tool-inventory-checked', { count: defaultTools.data.length });
    const providerID = args['--model'] === 'step-5-preview-free' ? 'zen-step5-free' : 'opencode';
    adapter = await createOpenCodeCliChatAdapter({ cliPath, client: sdkClient, baseUrl: serverUrl,
      directory: scratch, outputDirectory: output, actionLogPath: actionLog, providerID,
      modelAlias: `${providerID}/${args['--model']}`, modelID: args['--model'], maxCliTurns: 384, contextTokens: 32768,
      maxRequestMs: args['--max-request-ms'], timeoutMs: args['--max-request-ms'], env: cliEnvironment });
    const receipt = { schema: 'natlang.opencode_cli_loopback_bootstrap/1', bootstrap_id: bootstrapId,
      official_cli: cliPath, official_cli_sha256: await shaFile(cliPath), official_cli_version: cliVersion,
      official_sdk_module: sdkModule, official_sdk_module_sha256: await shaFile(sdkModule), provider_id: providerID,
      model_id: args['--model'], model_alias: `${providerID}/${args['--model']}`, provider_availability: 'not-probed',
      model_endpoint: args['--model'] === 'step-5-preview-free' ? 'https://opencode.ai/zen/v1/chat/completions' : null,
      model_config_source: args['--model'] === 'step-5-preview-free' ? 'documented Zen OpenAI-compatible provider' : 'OpenCode built-in free provider catalog',
      main_model: `${providerID}/${args['--model']}`, small_model: `${providerID}/${args['--model']}`,
      credential_source: 'OPENCODE_API_KEY environment; value excluded', server_url: serverUrl,
      server_pid: serverChild.pid, adapter_url: adapter.url, scratch_directory: scratch, isolated_home: isolatedHome,
      ...buildToolSurfaceReceipt(defaultTools.data), mcp_status: statuses.natlang_action_bridge.status,
      mcp_handshake_path: handshakeLog, action_log_path: actionLog,
      containment: 'requires bwrap launcher; official CLI server and model subprocess run under isolated HOME/XDG, scratch cwd; provider network shared',
      max_request_ms: args['--max-request-ms'], training_admission: false };
    await writeJson(resolve(output, 'bootstrap-config.json'), receipt, 'wx');
    await writeJson(resolve(output, 'lifecycle.json'), { bootstrap_id: bootstrapId, status: 'listening-provider-not-probed',
      at: new Date().toISOString(), training_admission: false }, 'wx');
    await stage('ready', { server_pid: serverChild.pid, default_tool_count: defaultTools.data.length, mcp_status: statuses.natlang_action_bridge.status });
    process.stdout.write(JSON.stringify({ endpoint: adapter.url, model: receipt.model_alias,
      provider_availability: 'not-probed', config_receipt: resolve(output, 'bootstrap-config.json') }) + '\n');
    await stopped;
    await stopping;
  } catch (error) {
    const detail = safe(error?.message);
    await stage('failure', { error: detail }).catch(() => {});
    await stop();
    process.stderr.write(`OpenCode CLI bootstrap failed (${detail}).\n`);
    process.exitCode = 1;
  } finally {
    clearInterval(shutdownPoll);
    process.removeListener('SIGINT', onSignal); process.removeListener('SIGTERM', onSignal);
  }
}

async function accessFile(path, label, executable) {
  await access(path, executable ? fsConstants.R_OK | fsConstants.X_OK : fsConstants.R_OK)
    .catch(() => { throw new Error(`${label} is not accessible`); });
  if (!(await stat(path)).isFile()) throw new Error(`${label} is not a file`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  main().catch(error => { process.stderr.write(`OpenCode CLI bootstrap failed (${safe(error?.message)}).\n`); process.exitCode = 1; });
