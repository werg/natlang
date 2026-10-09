import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { verifyStep5ArtifactClosure, verifyStep5ModelPair, verifyStep5SourceBinding, verifyStep5CollectorSourceSelection } from '../scripts/opencode-step5-preflight.mjs';

const alias = 'opencode/step-5-preview-free';
const bootstrapConfig = { model_alias: alias, main_model: alias, small_model: alias };
const plan = { model: { main_model: alias, small_model: alias, collector_model_id: alias },
  command_templates: { collector: `node cli.js input jobs output --model-id ${alias} --workers 1` } };

test('Step 5 preflight accepts an argv matching the immutable official bridge model', () => {
  assert.deepEqual(verifyStep5ModelPair({ plan, bootstrapConfig,
    collectorArgv: ['cli.js', 'input', 'jobs', 'output', '--model-id', alias, '--workers', '1'] }),
  { ok: true, model_alias: alias, collector_model_id: alias });
});

test('Step 5 preflight rejects a stale custom provider alias before collector launch', () => {
  assert.throws(() => verifyStep5ModelPair({ plan, bootstrapConfig,
    collectorArgv: ['cli.js', '--model-id', 'zen-step5-free/step-5-preview-free'] }),
  /actual collector argv --model-id does not match/);
});

test('Step 5 preflight rejects plan and bridge main/small model mismatches', () => {
  assert.throws(() => verifyStep5ModelPair({ plan: { ...plan,
    model: { ...plan.model, small_model: 'opencode/another-free-model' } }, bootstrapConfig,
  collectorArgv: ['cli.js', '--model-id', alias] }), /plan small_model does not match/);
  assert.throws(() => verifyStep5ModelPair({ plan, bootstrapConfig: { ...bootstrapConfig,
    small_model: 'opencode/another-free-model' }, collectorArgv: ['cli.js', '--model-id', alias] }),
  /immutable bridge main\/small model/);
});

test('Step 5 preflight pins the catalog-supported reasoning variant independently of model ID', () => {
  const lowPlan = { ...plan, model: { ...plan.model, variant: 'low' },
    command_templates: { ...plan.command_templates, bridge: `node bridge --model step-5-preview-free --variant low` } };
  const lowBootstrap = { ...bootstrapConfig, model_variant: 'low' };
  assert.deepEqual(verifyStep5ModelPair({ plan: lowPlan, bootstrapConfig: lowBootstrap,
    collectorArgv: ['cli.js', '--model-id', alias] }),
  { ok: true, model_alias: alias, collector_model_id: alias });
  assert.throws(() => verifyStep5ModelPair({ plan: lowPlan, bootstrapConfig,
    collectorArgv: ['cli.js', '--model-id', alias] }), /plan model variant does not match/);
  assert.throws(() => verifyStep5ModelPair({ plan: lowPlan,
    bootstrapConfig: { ...lowBootstrap, model_variant: 'high' }, collectorArgv: ['cli.js', '--model-id', alias] }),
  /plan model variant does not match/);
});

test('Step 5 preflight binds and verifies the effective Natlang-only agent tool surface', () => {
  const surfacePlan = { ...plan, provider: { tool_surface_mode: 'natlang-only' },
    model: { ...plan.model, variant: 'low' },
    command_templates: { ...plan.command_templates,
      bridge: 'node scripts/opencode-cli-loopback-bwrap-launch.mjs --model step-5-preview-free --variant low --tool-surface natlang-only' } };
  const surfaceConfig = { ...bootstrapConfig, model_variant: 'low', tool_surface_mode: 'natlang-only', cli_agent: 'build', mcp_status: 'connected',
    effective_agent_tool_surface: { agent: 'build', tools: { '*': false, natlang_action_bridge_submit_action: true } } };
  assert.equal(verifyStep5ModelPair({ plan: surfacePlan, bootstrapConfig: surfaceConfig,
    collectorArgv: ['cli.js', '--model-id', alias] }).tool_surface_mode, 'natlang-only');
  assert.throws(() => verifyStep5ModelPair({ plan: surfacePlan, bootstrapConfig: { ...bootstrapConfig, model_variant: 'low' },
    collectorArgv: ['cli.js', '--model-id', alias] }), /bootstrap tool surface/);
  assert.throws(() => verifyStep5ModelPair({ plan: surfacePlan,
    bootstrapConfig: { ...surfaceConfig, effective_agent_tool_surface: { agent: 'build',
      tools: { '*': false, natlang_action_bridge_submit_action: true, bash: true } } },
    collectorArgv: ['cli.js', '--model-id', alias] }), /do not match the Natlang-only/);
});

test('Step 5 preflight verifies exact bridge receipt bytes and official client pins', () => {
  const bridgeText = JSON.stringify({ model_alias: alias, main_model: alias, small_model: alias,
    official_cli: '/pinned/opencode', official_cli_sha256: 'cli-hash',
    official_sdk_module: '/pinned/sdk.mjs', official_sdk_module_sha256: 'sdk-hash' });
  const pinnedPlan = { ...plan, provider: { bootstrap_config_sha256: createHash('sha256').update(bridgeText).digest('hex'),
    official_cli: '/pinned/opencode', official_cli_sha256: 'cli-hash',
    official_sdk_module: '/pinned/sdk.mjs', official_sdk_module_sha256: 'sdk-hash' } };
  const pinnedConfig = JSON.parse(bridgeText);
  assert.equal(verifyStep5ModelPair({ plan: pinnedPlan, bootstrapConfig: pinnedConfig,
    bootstrapConfigText: bridgeText, collectorArgv: ['cli.js', '--model-id', alias] }).ok, true);
  assert.throws(() => verifyStep5ModelPair({ plan: pinnedPlan, bootstrapConfig: pinnedConfig,
    bootstrapConfigText: `${bridgeText}\n`, collectorArgv: ['cli.js', '--model-id', alias] }), /bytes do not match/);
  assert.throws(() => verifyStep5ModelPair({ plan: { ...pinnedPlan,
    provider: { ...pinnedPlan.provider, official_cli_sha256: 'wrong' } }, bootstrapConfig: pinnedConfig,
    bootstrapConfigText: bridgeText, collectorArgv: ['cli.js', '--model-id', alias] }), /official_cli_sha256/);
});

test('Step 5 preflight binds a post-readiness bridge port without weakening model or URL checks', () => {
  const dynamicServer = 'http://127.0.0.1:{adapter-port}/v1';
  const actualServer = 'http://127.0.0.1:33183/v1';
  const pendingPlan = { ...plan,
    provider: { bootstrap_config_sha256: null, bootstrap_config_binding: 'post-readiness',
      fresh_bootstrap_config_path: 'runs/fresh/bridge/bootstrap-config.json' },
    collector: { server: dynamicServer },
    model: { ...plan.model, variant: 'low' },
    command_templates: { bridge: 'node bridge --model step-5-preview-free --variant low',
      collector: `node cli.js input jobs output --model-id ${alias} --server ${dynamicServer}` } };
  const config = { ...bootstrapConfig, main_model: alias, small_model: alias, model_variant: 'low',
    adapter_url: actualServer, official_cli: '/pinned/opencode', official_cli_sha256: 'cli-hash',
    official_sdk_module: '/pinned/sdk.mjs', official_sdk_module_sha256: 'sdk-hash' };
  const configText = JSON.stringify(config);
  const argv = ['cli.js', 'input', 'jobs', 'output', '--model-id', alias, '--server', actualServer];
  const verified = verifyStep5ModelPair({ plan: pendingPlan, bootstrapConfig: config,
    bootstrapConfigText: configText, bootstrapConfigPath: 'runs/fresh/bridge/bootstrap-config.json', collectorArgv: argv });
  assert.equal(verified.ok, true);
  assert.equal(verified.bootstrap_config_sha256_bound, createHash('sha256').update(configText).digest('hex'));

  assert.throws(() => verifyStep5ModelPair({ plan: { ...pendingPlan,
    provider: { ...pendingPlan.provider, bootstrap_config_binding: undefined } }, bootstrapConfig: config,
    bootstrapConfigText: configText, bootstrapConfigPath: 'runs/fresh/bridge/bootstrap-config.json', collectorArgv: argv }), /explicit post-readiness binding/);
  assert.throws(() => verifyStep5ModelPair({ plan: pendingPlan, bootstrapConfig: config,
    bootstrapConfigText: configText, bootstrapConfigPath: 'runs/other/bridge/bootstrap-config.json', collectorArgv: argv }),
  /actual bootstrap config path does not match/);
  assert.throws(() => verifyStep5ModelPair({ plan: pendingPlan, bootstrapConfig: config,
    bootstrapConfigText: configText, bootstrapConfigPath: 'runs/fresh/bridge/bootstrap-config.json', collectorArgv: ['cli.js', '--model-id', 'opencode/paid-model', '--server', actualServer] }),
  /actual collector argv --model-id/);
  assert.throws(() => verifyStep5ModelPair({ plan: pendingPlan, bootstrapConfig: config,
    bootstrapConfigText: configText, bootstrapConfigPath: 'runs/fresh/bridge/bootstrap-config.json', collectorArgv: ['cli.js', '--model-id', alias, '--server', 'http://127.0.0.1:36833/v1'] }),
  /actual collector argv --server/);
  assert.throws(() => verifyStep5ModelPair({ plan: pendingPlan,
    bootstrapConfig: { ...config, adapter_url: 'http://example.com/v1' }, bootstrapConfigText: configText,
    bootstrapConfigPath: 'runs/fresh/bridge/bootstrap-config.json', collectorArgv: argv }), /valid loopback/);
  assert.throws(() => verifyStep5ModelPair({ plan: pendingPlan,
    bootstrapConfig: { ...config, small_model: 'opencode/paid-model' }, bootstrapConfigText: configText,
    bootstrapConfigPath: 'runs/fresh/bridge/bootstrap-config.json', collectorArgv: argv }), /immutable bridge main\/small model/);
  const exactNewlineConfig = `${configText}\n`;
  const newlineBound = verifyStep5ModelPair({ plan: pendingPlan, bootstrapConfig: config,
    bootstrapConfigText: exactNewlineConfig, bootstrapConfigPath: 'runs/fresh/bridge/bootstrap-config.json', collectorArgv: argv });
  assert.equal(newlineBound.bootstrap_config_sha256_bound,
    createHash('sha256').update(exactNewlineConfig).digest('hex'));
});

function authoredSourceBindingFixture() {
  const code = 'const file = folder.file("evidence.json"); return await nl<Fact>`Read it`(file);';
  const record = { id: 'source:microgrid:row1', split: 'train', source_groups: ['world:microgrid:2'],
    curriculum: { reference: { root: [[ 'eval', { code } ]] } } };
  const sourceCasesText = `${JSON.stringify(record)}\n`;
  const sourceRowSha256 = createHash('sha256').update(JSON.stringify(record)).digest('hex');
  const sourceSha256 = createHash('sha256').update(sourceCasesText).digest('hex');
  const source = { path: 'runs/source.cases.jsonl', sha256: sourceSha256, index: 0,
    row_sha256: sourceRowSha256, id: record.id, source_group: record.source_groups[0], split: record.split,
    authored_target_code_sha256: createHash('sha256').update(code).digest('hex') };
  const proof = { schema: 'test.exact-source-fake-proof/1', source_path: source.path,
    source_sha256: sourceSha256, source_row_index: 0, source_row_sha256: sourceRowSha256,
    source_id: record.id, source_groups: record.source_groups, split: record.split };
  const sourceProofText = JSON.stringify(proof);
  const plan = { schema: 'natlang.step5_authored_root_source_case_launch_plan/1', source,
    reference_protocol: { exact_source_fake_proof: { path: 'runs/fake-proof.json',
      bytes: Buffer.byteLength(sourceProofText),
      sha256: createHash('sha256').update(sourceProofText).digest('hex') } } };
  return { plan, sourceCasesText, sourceProofText, source, record, sourceSha256, sourceRowSha256 };
}

test('collector selects the exact zero-based authored source row before launch', () => {
  const { plan } = authoredSourceBindingFixture();
  const argv = ['cli.js', plan.source.path, 'jobs', 'results.jsonl', '--start', '0', '--limit', '1'];
  assert.equal(verifyStep5CollectorSourceSelection({ plan, collectorArgv: argv }).start, 0);
  for (const bad of [
    argv.map((v, i) => i === 5 ? '1' : v),
    argv.map((v, i) => i === 1 ? 'runs/other.jsonl' : v),
    [...argv, '--start', '0'], [...argv, '--all'],
    argv.map((v, i) => i === 7 ? '2' : v), argv.slice(0, 4)
  ]) assert.throws(() => verifyStep5CollectorSourceSelection({ plan, collectorArgv: bad }), /collector/);
});

test('authored-root preflight binds the exact physical source row to its fake proof', () => {
  const fixture = authoredSourceBindingFixture();
  const result = verifyStep5SourceBinding(fixture);
  assert.deepEqual(result, { ok: true, source_path: fixture.source.path, source_index: 0,
    source_id: fixture.source.id, source_group: fixture.source.source_group, split: 'train',
    source_row_sha256: fixture.sourceRowSha256, exact_source_fake_proof_path: 'runs/fake-proof.json',
    exact_source_fake_proof_sha256: fixture.plan.reference_protocol.exact_source_fake_proof.sha256 });
});

test('authored-root preflight accepts the proof schema zero-based source_index field', () => {
  const fixture = authoredSourceBindingFixture();
  const proof = JSON.parse(fixture.sourceProofText);
  delete proof.source_row_index;
  proof.source_index = 0;
  const sourceProofText = JSON.stringify(proof);
  const plan = { ...fixture.plan, reference_protocol: { exact_source_fake_proof: {
    path: 'runs/fake-proof-source-index.json', bytes: Buffer.byteLength(sourceProofText),
    sha256: createHash('sha256').update(sourceProofText).digest('hex') } } };
  assert.equal(verifyStep5SourceBinding({ ...fixture, plan, sourceProofText }).source_index, 0);
});

test('authored-root preflight rejects an exact-source proof copied from another row', () => {
  const fixture = authoredSourceBindingFixture();
  const other = { ...fixture.record, id: 'source:clinic:row0', source_groups: ['world:clinic:2'] };
  const otherLine = JSON.stringify(other);
  const proof = { schema: 'test.exact-source-fake-proof/1', source_path: fixture.source.path,
    source_sha256: fixture.sourceSha256, source_row_index: 0,
    source_row_sha256: createHash('sha256').update(otherLine).digest('hex'), source_id: other.id,
    source_groups: other.source_groups, split: 'train' };
  const sourceProofText = JSON.stringify(proof);
  const plan = { ...fixture.plan, reference_protocol: { exact_source_fake_proof: {
    path: 'runs/other-row-proof.json', bytes: Buffer.byteLength(sourceProofText),
    sha256: createHash('sha256').update(sourceProofText).digest('hex') } } };
  assert.throws(() => verifyStep5SourceBinding({ ...fixture, plan, sourceProofText }),
    /fake proof does not bind the selected source file\/row\/ID\/group\/split/);
});

test('authored-root preflight rejects source-byte, group, split and target-code drift', () => {
  const fixture = authoredSourceBindingFixture();
  assert.throws(() => verifyStep5SourceBinding({ ...fixture, sourceCasesText: `${fixture.sourceCasesText} ` }),
    /source-cases bytes do not match/);
  for (const [field, value] of [['source_group', 'world:other'], ['split', 'test'],
    ['authored_target_code_sha256', 'wrong']]) {
    const plan = { ...fixture.plan, source: { ...fixture.source, [field]: value } };
    assert.throws(() => verifyStep5SourceBinding({ ...fixture, plan }),
      field === 'authored_target_code_sha256' ? /controller code does not match/ : /ID\/group\/split/);
  }
});

async function artifactClosureFixture() {
  const root = await mkdtemp(join(tmpdir(), 'step5-artifact-closure-'));
  const names = ['adapter', 'bootstrap', 'collector_runner', 'invalid_tool_classifier',
    'launcher', 'mcp_server', 'preflight', 'structured_turn'];
  const closure = {}, files = {};
  for (const [index, name] of names.entries()) {
    const execution_path = `scripts/${name}.mjs`;
    const snapshot_path = `snapshot/scripts/${name}.mjs`;
    const bytes = Buffer.from(`immutable ${name} ${index}`);
    await mkdir(join(root, 'scripts'), { recursive: true });
    await mkdir(join(root, 'snapshot/scripts'), { recursive: true });
    await writeFile(join(root, execution_path), bytes);
    await writeFile(join(root, snapshot_path), bytes);
    files[name] = { execution_path, snapshot_path, bytes: bytes.length,
      sha256: createHash('sha256').update(bytes).digest('hex') };
    closure[name] = { path: execution_path, immutable_snapshot_path: snapshot_path,
      bytes: bytes.length, sha256: files[name].sha256 };
  }
  const snapshot = { schema: 'natlang.step5_bridge_code_snapshot/1', source_git_head: 'fixture', files };
  const snapshotBytes = Buffer.from(`${JSON.stringify(snapshot)}\n`);
  await writeFile(join(root, 'snapshot/manifest.json'), snapshotBytes);
  const entryPath = 'frozen/dist/teacher/cli.js';
  const entryBytes = Buffer.from('frozen collector entry');
  await mkdir(join(root, 'frozen/dist/teacher'), { recursive: true });
  await writeFile(join(root, entryPath), entryBytes);
  const runtimeManifest = { files: { 'dist/teacher/cli.js': createHash('sha256').update(entryBytes).digest('hex') } };
  const runtimeManifestBytes = Buffer.from(`${JSON.stringify(runtimeManifest)}\n`);
  await writeFile(join(root, 'frozen/frozen-runtime.json'), runtimeManifestBytes);
  const proofBytes = Buffer.from(JSON.stringify({ schema: 'fixture.fake-proof/1' }));
  await writeFile(join(root, 'proof.json'), proofBytes);
  const runtimeManifestSha = createHash('sha256').update(runtimeManifestBytes).digest('hex');
  const plan = {
    schema: 'natlang.step5_authored_root_source_case_launch_plan/1',
    bridge_code_import_closure: closure,
    bridge_code_import_closure_snapshot: { path: 'snapshot/manifest.json', bytes: snapshotBytes.length,
      sha256: createHash('sha256').update(snapshotBytes).digest('hex') },
    command_templates: { collector: `${entryPath} cases jobs results` },
    collector: { entry: entryPath,
      entry_sha256: runtimeManifest.files['dist/teacher/cli.js'], runtime_manifest_sha256: runtimeManifestSha },
    runtime: { path: 'frozen', manifest: { path: 'frozen/frozen-runtime.json', sha256: runtimeManifestSha,
      bytes: runtimeManifestBytes.length }, source_specific_proof: { path: 'proof.json',
      sha256: createHash('sha256').update(proofBytes).digest('hex'), runtime_manifest_sha256: runtimeManifestSha } },
    reference_protocol: { exact_source_fake_proof: { path: 'proof.json',
      sha256: createHash('sha256').update(proofBytes).digest('hex'), runtime_manifest_sha256: runtimeManifestSha } },
  };
  return { root, plan, runtimeManifestSha, names };
}

test('Step 5 artifact preflight verifies all imported files against immutable closure and runtime', async t => {
  const fixture = await artifactClosureFixture();
  t.after(() => rm(fixture.root, { recursive: true, force: true }));
  const result = await verifyStep5ArtifactClosure({ plan: fixture.plan, repoRoot: fixture.root });
  assert.equal(result.ok, true);
  assert.equal(result.closure_files_verified, 8);
  assert.equal(result.runtime_manifest_sha256, fixture.runtimeManifestSha);
  assert.equal(result.exact_source_fake_proof_runtime_sha256, fixture.runtimeManifestSha);
});

test('Step 5 artifact preflight rejects live bridge source drift from its pinned snapshot', async t => {
  const fixture = await artifactClosureFixture();
  t.after(() => rm(fixture.root, { recursive: true, force: true }));
  await writeFile(join(fixture.root, fixture.plan.bridge_code_import_closure.preflight.path), 'changed source');
  await assert.rejects(verifyStep5ArtifactClosure({ plan: fixture.plan, repoRoot: fixture.root }),
    /current bridge execution file preflight differs/);
});

test('Step 5 artifact preflight rejects stale closure metadata and collector entry drift', async t => {
  const fixture = await artifactClosureFixture();
  t.after(() => rm(fixture.root, { recursive: true, force: true }));
  const stale = structuredClone(fixture.plan);
  stale.bridge_code_import_closure.preflight.sha256 = 'stale';
  await assert.rejects(verifyStep5ArtifactClosure({ plan: stale, repoRoot: fixture.root }),
    /closure entry preflight differs/);
  const wrongEntry = structuredClone(fixture.plan);
  wrongEntry.collector.entry = 'frozen/other.js';
  await assert.rejects(verifyStep5ArtifactClosure({ plan: wrongEntry, repoRoot: fixture.root }),
    /collector entry\/runtime manifest do not match/);
});

test('Step 5 artifact preflight rejects a proof pinned to a different runtime', async t => {
  const fixture = await artifactClosureFixture();
  t.after(() => rm(fixture.root, { recursive: true, force: true }));
  const wrongProof = structuredClone(fixture.plan);
  wrongProof.reference_protocol.exact_source_fake_proof.runtime_manifest_sha256 = 'other-runtime';
  await assert.rejects(verifyStep5ArtifactClosure({ plan: wrongProof, repoRoot: fixture.root }),
    /exact-source fake proof runtime differs/);
});
