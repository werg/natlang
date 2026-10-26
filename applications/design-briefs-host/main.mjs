#!/usr/bin/env node
/** Launch the exact interface service or inspect a program's own native API. */
import { readFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { openDesignBriefStore } from './store.mjs';
import { loadApplications, nativeExecutor, describeApplication, ApplicationError } from './runtime.mjs';
import { createApplicationService } from './service.mjs';
import { serveApplications, authenticateBearer } from './server.mjs';
import { dispatchOnce } from './delivery.mjs';

async function configured(path) {
  const config = JSON.parse(await readFile(path, 'utf8'));
  if (!config.database || !config.capabilities || !Array.isArray(config.credentials)) throw new ApplicationError('invalid_configuration');
  config.database = resolve(dirname(path), config.database);
  await mkdir(dirname(config.database), { recursive: true, mode: 0o700 });
  return config;
}

export async function main(args = process.argv.slice(2)) {
  const [command, argument, ...rest] = args;
  if (command === 'describe') {
    const catalog = await loadApplications();
    const app = catalog.get(argument);
    if (!app) throw new ApplicationError('application_not_found');
    process.stdout.write(JSON.stringify(describeApplication(app), null, 2) + '\n');
    return;
  }
  if (!['serve','dispatch'].includes(command) || !argument)
    throw new ApplicationError('usage', 'main.mjs describe APP_ID | serve CONFIG.json | dispatch CONFIG.json APP_ID WORKER_ID');
  const configPath = resolve(argument), config = await configured(configPath);
  const store = openDesignBriefStore({ filename: config.database, capabilities: config.capabilities });
  if (command === 'dispatch') {
    try {
      const principal = authenticateBearer(`Bearer ${process.env.NATLANG_APPLICATION_TOKEN ?? ''}`, config.credentials);
      const results = await dispatchOnce({ store, principal, appId: rest[0], workerId: rest[1], routes: config.deliveryRoutes,
        ...(config.delivery ?? {}) });
      process.stdout.write(JSON.stringify(results) + '\n');
    } finally { store.close(); }
    return;
  }
  try {
    const catalog = await loadApplications();
    const effectAdapters = {};
    for (const [appId, entries] of Object.entries(config.effectAdapters ?? {})) {
      if (!catalog.has(appId)) throw new ApplicationError('unknown_adapter_application');
      effectAdapters[appId] = {};
      for (const [name, filename] of Object.entries(entries)) {
        if (!catalog.get(appId).interfaces.has(name)) throw new ApplicationError('unknown_adapter_interface');
        const adapter = (await import(pathToFileURL(resolve(dirname(configPath), filename)).href)).default;
        if (typeof adapter !== 'function') throw new ApplicationError('invalid_effect_adapter');
        effectAdapters[appId][name] = adapter;
      }
    }
    const model = { ...config.model };
    if (model.apiKeyEnv) { model.apiKey = process.env[model.apiKeyEnv]; delete model.apiKeyEnv; }
    const execute = nativeExecutor({ model, limits: config.executionLimits });
    const service = createApplicationService({ catalog, store, execute, capabilities: config.capabilities, effectAdapters });
    const server = await serveApplications({ service, credentials: config.credentials, ...(config.http ?? {}) });
    process.stdout.write(`Native application interfaces listening at ${server.url}\n`);
    const signal = await new Promise(resolve => {
      const stop = name => { process.off('SIGINT', onInt); process.off('SIGTERM', onTerm); resolve(name); };
      const onInt = () => stop('SIGINT'), onTerm = () => stop('SIGTERM');
      process.once('SIGINT', onInt); process.once('SIGTERM', onTerm);
    });
    await server.close();
    process.stdout.write(`Stopped after ${signal}\n`);
  } finally { store.close(); }
}

if (process.argv[1] && resolve(process.argv[1]) === import.meta.filename)
  main().catch(error => { process.stderr.write(`${error.code ?? 'application_failed'}${error.code === 'usage' ? ': '+error.detail : ''}\n`); process.exitCode = 1; });
