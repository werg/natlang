#!/usr/bin/env node
/** Verify the collector's requested model matches the official isolated OpenCode bridge. */
import { readFile } from 'node:fs/promises';

function modelIdFromArgv(argv) {
  if (!Array.isArray(argv) || argv.some(value => typeof value !== 'string'))
    throw new TypeError('collector argv must be an array of strings');
  const positions = argv.flatMap((value, index) => value === '--model-id' ? [index] : []);
  if (positions.length !== 1 || positions[0] + 1 >= argv.length)
    throw new Error('collector argv must contain exactly one --model-id value');
  return argv[positions[0] + 1];
}

export function verifyStep5ModelPair({ plan, bootstrapConfig, collectorArgv }) {
  const bridgeAlias = bootstrapConfig?.model_alias;
  if (typeof bridgeAlias !== 'string' || !bridgeAlias)
    throw new Error('bootstrap config lacks immutable model_alias');
  for (const [name, value] of [['main_model', plan?.model?.main_model],
    ['small_model', plan?.model?.small_model], ['collector_model_id', plan?.model?.collector_model_id]]) {
    if (value !== bridgeAlias) throw new Error(`plan ${name} does not match immutable bridge model_alias`);
  }
  const actualArgvAlias = modelIdFromArgv(collectorArgv);
  if (actualArgvAlias !== bridgeAlias)
    throw new Error('actual collector argv --model-id does not match immutable bridge model_alias');
  const template = plan?.command_templates?.collector;
  const match = typeof template === 'string' && template.match(/(?:^|\s)--model-id\s+([^\s]+)/);
  if (!match || match[1] !== bridgeAlias)
    throw new Error('collector command template --model-id does not match immutable bridge model_alias');
  if (bootstrapConfig.main_model !== bridgeAlias || bootstrapConfig.small_model !== bridgeAlias)
    throw new Error('immutable bridge main/small model do not match model_alias');
  const plannedVariant = plan.model.variant ?? 'catalog_default';
  if ((bootstrapConfig.model_variant ?? 'catalog_default') !== plannedVariant)
    throw new Error('plan model variant does not match immutable bridge variant');
  const bridgeTemplate = plan.command_templates?.bridge;
  const variantFlag = typeof bridgeTemplate === 'string' && bridgeTemplate.match(/(?:^|\s)--variant\s+([^\s]+)/);
  if ((plannedVariant === 'catalog_default' && variantFlag) ||
      (plannedVariant !== 'catalog_default' && variantFlag?.[1] !== plannedVariant))
    throw new Error('bridge command template variant does not match pinned model variant');
  const expectedToolSurface = plan?.provider?.tool_surface_mode;
  if (expectedToolSurface !== undefined) {
    if (!['standard', 'natlang-only'].includes(expectedToolSurface) ||
        bootstrapConfig.tool_surface_mode !== expectedToolSurface)
      throw new Error('bootstrap tool surface does not match pinned plan');
    if (expectedToolSurface === 'natlang-only') {
      const surface = bootstrapConfig.effective_agent_tool_surface;
      if (bootstrapConfig.cli_agent !== 'build' || bootstrapConfig.mcp_status !== 'connected' ||
          surface?.agent !== 'build' || surface?.tools?.['*'] !== false ||
          surface?.tools?.natlang_action_bridge_submit_action !== true ||
          Object.entries(surface?.tools ?? {}).some(([name, enabled]) =>
            name !== '*' && name !== 'natlang_action_bridge_submit_action' && enabled === true))
        throw new Error('effective OpenCode build-agent tools do not match the Natlang-only surface');
      const bridgeTemplate = plan.command_templates?.bridge;
      if (typeof bridgeTemplate !== 'string' || !/(?:^|\s)--tool-surface\s+natlang-only(?:\s|$)/.test(bridgeTemplate))
        throw new Error('bridge command template does not pin Natlang-only tool surface');
    }
  }
  return { ok: true, model_alias: bridgeAlias, collector_model_id: actualArgvAlias,
    ...(expectedToolSurface === undefined ? {} : { tool_surface_mode: expectedToolSurface }) };
}

async function main(argv) {
  const args = new Map();
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i], value = argv[i + 1];
    if (!['--plan', '--bootstrap-config', '--collector-argv-json'].includes(key) || !value || args.has(key))
      throw new Error(`invalid or duplicate option ${key}`);
    args.set(key, value);
  }
  for (const key of ['--plan', '--bootstrap-config', '--collector-argv-json'])
    if (!args.has(key)) throw new Error(`${key} is required`);
  const [plan, bootstrapConfig, collectorArgv] = await Promise.all([
    readFile(args.get('--plan'), 'utf8').then(JSON.parse),
    readFile(args.get('--bootstrap-config'), 'utf8').then(JSON.parse),
    readFile(args.get('--collector-argv-json'), 'utf8').then(JSON.parse)
  ]);
  process.stdout.write(`${JSON.stringify(verifyStep5ModelPair({ plan, bootstrapConfig, collectorArgv }))}\n`);
}

if (process.argv[1]?.endsWith('/opencode-step5-preflight.mjs'))
  main(process.argv.slice(2)).catch(error => { process.stderr.write(`Step 5 preflight failed: ${error.message}\n`); process.exitCode = 1; });
