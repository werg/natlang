#!/usr/bin/env node
/** Start the official OpenCode SDK server and the bounded Natlang loopback adapter. */
import { access, mkdir, stat, writeFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { constants as fsConstants } from 'node:fs';
import { dirname, resolve, basename, delimiter } from 'node:path';
import { pathToFileURL } from 'node:url';
import { randomUUID } from 'node:crypto';
import { createHash } from 'node:crypto';
import { createOpenCodeLoopbackChatAdapter } from './opencode-loopback-chat-adapter.mjs';

function parseArgs(argv) {
  const values = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith('--') || i + 1 >= argv.length) throw new Error(`invalid argument near ${token}`);
    const key = token.slice(2);
    if (!['sdk-module', 'client-bin', 'out', 'model', 'max-request-ms'].includes(key))
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

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const sdkModule = await requireFile(args['sdk-module'], 'SDK module');
  const clientBin = await requireFile(args['client-bin'], 'OpenCode client binary');
  if (basename(clientBin) !== 'opencode') throw new Error('--client-bin must point to the official opencode executable');
  await access(clientBin, fsConstants.X_OK);
  if (!process.env.OPENCODE_API_KEY) throw new Error('OPENCODE_API_KEY must be present in the environment');

  const output = resolve(args.out);
  await mkdir(output, { recursive: false });
  const scratch = resolve(output, 'scratch');
  await mkdir(scratch, { recursive: false });
  process.env.PATH = `${dirname(clientBin)}${delimiter}${process.env.PATH ?? ''}`;

  let official;
  let adapter;
  let stopping;
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
    output_directory: output,
    max_concurrency: 1,
    max_request_ms: args['max-request-ms'],
    response_mode: 'buffered JSON, including when stream=true',
    native_provider_tool_calls: false,
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
  const onSignal = () => { void stop(); };
  process.once('SIGINT', onSignal);
  process.once('SIGTERM', onSignal);

  try {
    const { createOpencode } = await import(pathToFileURL(sdkModule).href);
    if (typeof createOpencode !== 'function') throw new Error('SDK module does not export createOpencode');
    official = await createOpencode({ hostname: '127.0.0.1', port: 0, timeout: 20_000,
      config: { provider: { opencode: { options: { apiKey: '{env:OPENCODE_API_KEY}' } } },
        share: 'disabled', autoupdate: false } });
    if (!official?.client || !official?.server?.close) throw new Error('SDK did not return a client and server handle');
    adapter = await createOpenCodeLoopbackChatAdapter({ client: official.client, providerID: 'opencode',
      modelID: args.model, directory: scratch, maxConcurrency: 1, maxRequestMs: args['max-request-ms'] });
    const config = Object.freeze({ ...configReceipt, adapter_bind: { host: adapter.config.host, port: adapter.config.port },
      provider_availability: 'not-probed' });
    await writeFile(resolve(output, 'bootstrap-config.json'), JSON.stringify(config, null, 2) + '\n', { flag: 'wx' });
    await writeLifecycle('listening-provider-not-probed');
    process.stdout.write(JSON.stringify({ endpoint: adapter.url, model: config.model_alias,
      provider_availability: 'not-probed', config_receipt: resolve(output, 'bootstrap-config.json') }) + '\n');
    await new Promise(resolveDone => {
      const poll = setInterval(() => {
        if (stopping) { clearInterval(poll); void stopping.finally(resolveDone); }
      }, 100);
      poll.unref?.();
    });
  } catch (error) {
    await stop();
    process.stderr.write(`OpenCode loopback bootstrap failed (${error?.name ?? 'Error'}).\n`);
    process.exitCode = 1;
  } finally {
    process.removeListener('SIGINT', onSignal);
    process.removeListener('SIGTERM', onSignal);
  }
}

main().catch(error => {
  process.stderr.write(`OpenCode loopback bootstrap failed (${error?.name ?? 'Error'}).\n`);
  process.exitCode = 1;
});
