#!/usr/bin/env node
/** Verify the collector's requested model matches the official isolated OpenCode bridge. */
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, sep } from 'node:path';

function modelIdFromArgv(argv) {
  if (!Array.isArray(argv) || argv.some(value => typeof value !== 'string'))
    throw new TypeError('collector argv must be an array of strings');
  const positions = argv.flatMap((value, index) => value === '--model-id' ? [index] : []);
  if (positions.length !== 1 || positions[0] + 1 >= argv.length)
    throw new Error('collector argv must contain exactly one --model-id value');
  return argv[positions[0] + 1];
}

function uniqueArgValue(argv, flag, { required = false } = {}) {
  const positions = argv.flatMap((value, index) => value === flag ? [index] : []);
  if (!positions.length && !required) return null;
  if (positions.length !== 1 || positions[0] + 1 >= argv.length)
    throw new Error(`collector argv must contain exactly one ${flag} value`);
  return argv[positions[0] + 1];
}

function isLoopbackAdapterUrl(value) {
  if (typeof value !== 'string') return false;
  try {
    const url = new URL(value);
    return url.protocol === 'http:' && url.hostname === '127.0.0.1' &&
      /^[1-9][0-9]{0,4}$/.test(url.port) && Number(url.port) <= 65535 &&
      url.pathname === '/v1' && !url.username && !url.password && !url.search && !url.hash;
  } catch { return false; }
}

export function verifyStep5ModelPair({ plan, bootstrapConfig, collectorArgv, bootstrapConfigText,
  bootstrapConfigPath }) {
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
  const providerPins = plan?.provider ?? {};
  let bootstrapConfigSha256Bound;
  if (providerPins.fresh_bootstrap_config_path !== undefined &&
      (typeof bootstrapConfigPath !== 'string' ||
       resolve(bootstrapConfigPath) !== resolve(providerPins.fresh_bootstrap_config_path)))
    throw new Error('actual bootstrap config path does not match the plan fresh config path');
  if (providerPins.bootstrap_config_sha256 === null) {
    if (providerPins.bootstrap_config_binding !== 'post-readiness' ||
        typeof providerPins.fresh_bootstrap_config_path !== 'string' || !providerPins.fresh_bootstrap_config_path ||
        typeof bootstrapConfigText !== 'string' || typeof bootstrapConfigPath !== 'string')
      throw new Error('null bootstrap config hash requires an explicit post-readiness binding and exact config bytes');
    bootstrapConfigSha256Bound = createHash('sha256').update(bootstrapConfigText).digest('hex');
  } else if (providerPins.bootstrap_config_sha256 !== undefined) {
    if (typeof providerPins.bootstrap_config_sha256 !== 'string' ||
        typeof bootstrapConfigText !== 'string' ||
        createHash('sha256').update(bootstrapConfigText).digest('hex') !== providerPins.bootstrap_config_sha256)
      throw new Error('bootstrap config bytes do not match the pinned plan hash');
    bootstrapConfigSha256Bound = providerPins.bootstrap_config_sha256;
  }
  for (const [pin, field] of [['official_cli_sha256', 'official_cli_sha256'],
    ['official_sdk_module_sha256', 'official_sdk_module_sha256'], ['official_cli', 'official_cli'],
    ['official_sdk_module', 'official_sdk_module']])
    if (providerPins[pin] !== undefined && providerPins[pin] !== bootstrapConfig[field])
      throw new Error(`bootstrap ${field} does not match its provider pin`);
  const plannedVariant = plan.model.variant ?? 'catalog_default';
  if ((bootstrapConfig.model_variant ?? 'catalog_default') !== plannedVariant)
    throw new Error('plan model variant does not match immutable bridge variant');
  const actualAdapterUrl = bootstrapConfig.adapter_url;
  if (actualAdapterUrl !== undefined) {
    if (!isLoopbackAdapterUrl(actualAdapterUrl))
      throw new Error('immutable bridge adapter_url is not a valid loopback /v1 endpoint');
    const actualServer = uniqueArgValue(collectorArgv, '--server', { required: true });
    if (actualServer !== actualAdapterUrl)
      throw new Error('actual collector argv --server does not match immutable bridge adapter_url');
    const plannedServer = plan?.collector?.server;
    const dynamicServer = 'http://127.0.0.1:{adapter-port}/v1';
    if (plannedServer !== undefined && plannedServer !== actualAdapterUrl &&
        !(providerPins.bootstrap_config_sha256 === null && plannedServer === dynamicServer))
      throw new Error('plan collector server does not match the exact bridge URL or approved post-readiness placeholder');
    const templateServer = uniqueArgValue(
      String(plan?.command_templates?.collector ?? '').split(/\s+/), '--server', { required: false });
    if (templateServer !== null && templateServer !== actualAdapterUrl &&
        !(providerPins.bootstrap_config_sha256 === null && templateServer === dynamicServer))
      throw new Error('collector command template server does not match the exact bridge URL or approved post-readiness placeholder');
  }
  const bridgeTemplate = plan.command_templates?.bridge;
  const variantFlag = typeof bridgeTemplate === 'string' && bridgeTemplate.match(/(?:^|\s)--variant\s+([^\s]+)/);
  if ((plannedVariant === 'catalog_default' && variantFlag) ||
      (plannedVariant !== 'catalog_default' && variantFlag?.[1] !== plannedVariant))
    throw new Error('bridge command template variant does not match pinned model variant');
  const expectedToolSurface = providerPins.tool_surface_mode;
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
    ...(bootstrapConfigSha256Bound === undefined ? {} : { bootstrap_config_sha256_bound: bootstrapConfigSha256Bound }),
    ...(expectedToolSurface === undefined ? {} : { tool_surface_mode: expectedToolSurface }) };
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function rowSha256(line) {
  return sha256(Buffer.from(line.replace(/\r$/, ''), 'utf8'));
}

/**
 * Bind an authored-root Step 5 plan and its exact-source fake proof to the
 * selected physical source JSONL row. This prevents a valid neighboring-row
 * proof from being copied into a new plan while leaving the model/provider
 * route checks independent.
 */
export function verifyStep5SourceBinding({ plan, sourceCasesText, sourceProofText }) {
  if (plan?.schema !== 'natlang.step5_authored_root_source_case_launch_plan/1')
    return { ok: true, skipped: true };
  const source = plan.source;
  if (!source || typeof source.path !== 'string' || typeof source.sha256 !== 'string' ||
      !Number.isSafeInteger(source.index) || source.index < 0 || typeof source.row_sha256 !== 'string' ||
      typeof source.id !== 'string' || typeof source.source_group !== 'string' || typeof source.split !== 'string')
    throw new Error('authored-root plan lacks an exact physical source-row binding');
  if (typeof sourceCasesText !== 'string' || sha256(Buffer.from(sourceCasesText, 'utf8')) !== source.sha256)
    throw new Error('source-cases bytes do not match the authored-root plan pin');

  const lines = sourceCasesText.split('\n');
  if (lines.at(-1) === '') lines.pop();
  if (source.index >= lines.length || !lines[source.index].trim())
    throw new Error('authored-root source row index is outside the pinned source file');
  const selectedLine = lines[source.index];
  if (rowSha256(selectedLine) !== source.row_sha256)
    throw new Error('physical source row bytes do not match the authored-root plan row hash');
  let record;
  try { record = JSON.parse(selectedLine.replace(/\r$/, '')); }
  catch { throw new Error('selected authored-root source row is not valid JSON'); }
  const sourceId = record?.id ?? record?.case_id;
  const groups = record?.source_groups ?? record?.groups;
  if (sourceId !== source.id || !Array.isArray(groups) || !groups.includes(source.source_group) ||
      record?.split !== source.split)
    throw new Error('selected authored-root source row ID/group/split does not match the plan');
  const rootCode = record?.curriculum?.reference?.root?.[0]?.[1]?.code;
  if (typeof source.authored_target_code_sha256 === 'string' &&
      (typeof rootCode !== 'string' || sha256(Buffer.from(rootCode, 'utf8')) !== source.authored_target_code_sha256))
    throw new Error('selected authored-root controller code does not match the plan target pin');

  const proofPin = plan?.reference_protocol?.exact_source_fake_proof;
  if (!proofPin || typeof proofPin.path !== 'string' || typeof proofPin.sha256 !== 'string' ||
      typeof sourceProofText !== 'string' || sha256(Buffer.from(sourceProofText, 'utf8')) !== proofPin.sha256 ||
      (Number.isSafeInteger(proofPin.bytes) && Buffer.byteLength(sourceProofText, 'utf8') !== proofPin.bytes))
    throw new Error('exact-source fake proof bytes do not match the plan pin');
  let proof;
  try { proof = JSON.parse(sourceProofText); }
  catch { throw new Error('exact-source fake proof is not valid JSON'); }
  const proofSourceHash = proof?.source_sha256 ?? proof?.source_file_sha256;
  const proofRowIndex = Number.isSafeInteger(proof?.source_row_index) ? proof.source_row_index :
    (Number.isSafeInteger(proof?.case_index) ? proof.case_index - 1 : null);
  const proofSourceId = proof?.source_id ?? proof?.case_id;
  const proofGroups = proof?.source_groups ?? (proof?.source_group ? [proof.source_group] : null);
  if (proofSourceHash !== source.sha256 || proofRowIndex !== source.index ||
      proof?.source_row_sha256 !== source.row_sha256 || proofSourceId !== source.id ||
      !Array.isArray(proofGroups) || !proofGroups.includes(source.source_group) || proof?.split !== source.split)
    throw new Error('exact-source fake proof does not bind the selected source file/row/ID/group/split');
  if (typeof proof.source_path === 'string' && resolve(proof.source_path) !== resolve(source.path))
    throw new Error('exact-source fake proof path does not match the plan source path');
  return { ok: true, source_path: source.path, source_index: source.index,
    source_id: source.id, source_group: source.source_group, split: source.split,
    source_row_sha256: source.row_sha256, exact_source_fake_proof_path: proofPin.path,
    exact_source_fake_proof_sha256: proofPin.sha256 };
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
  const [planText, bootstrapConfigText, collectorArgvText] = await Promise.all([
    readFile(args.get('--plan'), 'utf8'), readFile(args.get('--bootstrap-config'), 'utf8'),
    readFile(args.get('--collector-argv-json'), 'utf8')
  ]);
  const plan = JSON.parse(planText);
  const modelBinding = verifyStep5ModelPair({ plan,
    bootstrapConfig: JSON.parse(bootstrapConfigText), collectorArgv: JSON.parse(collectorArgvText),
    bootstrapConfigText, bootstrapConfigPath: args.get('--bootstrap-config') });
  let sourceBinding;
  if (plan?.schema === 'natlang.step5_authored_root_source_case_launch_plan/1') {
    const proofPin = plan?.reference_protocol?.exact_source_fake_proof;
    if (!proofPin?.path) throw new Error('authored-root plan lacks a pinned exact-source fake proof path');
    const repoRoot = resolve(process.cwd());
    const planPath = value => {
      const path = resolve(value);
      if (path !== repoRoot && !path.startsWith(`${repoRoot}${sep}`))
        throw new Error('authored-root source/proof path escapes the canonical repository');
      return path;
    };
    const [sourceCasesText, sourceProofText] = await Promise.all([
      readFile(planPath(plan.source.path), 'utf8'), readFile(planPath(proofPin.path), 'utf8')
    ]);
    sourceBinding = verifyStep5SourceBinding({ plan, sourceCasesText, sourceProofText });
  }
  process.stdout.write(`${JSON.stringify({ ...modelBinding, ...(sourceBinding ? { source_binding: sourceBinding } : {}) })}\n`);
}

if (process.argv[1]?.endsWith('/opencode-step5-preflight.mjs'))
  main(process.argv.slice(2)).catch(error => { process.stderr.write(`Step 5 preflight failed: ${error.message}\n`); process.exitCode = 1; });
