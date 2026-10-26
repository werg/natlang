/**
 * Bridge between `natlang tune` collection and the supervisor-managed generation pool (plans/HIERARCHICAL_LORA_TUNING.md
 * §4.2: "integrate with that pool; a program build must not launch a duplicate pool").
 *
 *   exportCases   root examples whose data is missing or stale → an immutable `natlang.program/2` case list in the form
 *                 the reviewed Luna dispatcher admits (source rows with source groups and split, a cases array with
 *                 seeds, a binding manifest). Root review and dispatch stay with the pool's operators.
 *   importCases   the pool's collector result rows → `natlang.tune-root/v1` records, each model turn mapped to its
 *                 invocation's binding through the site-aware scaffold, under the exact collector actor identity.
 *
 * The collector renders the same call messages as the ordinary runtime, but its system prompt appends the collection
 * guidance (`GENERATION_GUIDANCE`). Training rows present the serving prompt: the import removes exactly that block
 * from each system message and rejects a row whose prompt is anything else.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type { ProgramDescriptor } from '../adaptation/types.js';
import type { WeightManifest } from '../adaptation/weights/manifest.js';
import { WeightBinding } from '../adaptation/weights/binding.js';
import { tuneDirectory } from '../adaptation/weights/release.js';
import { GENERATION_GUIDANCE, GENERATION_GUIDANCE_AT_NL_DEPTH_LIMIT, TOOLS_PROMPT } from '../native/prompt.js';
import { loadNatlang } from '../runtime/node-files.js';
import type { CollectedTurn, SplitExample } from './collect.js';
import { sharedSourceHash, trainingScaffoldKey } from './keys.js';
import { canonicalJson } from '../package/archive.js';
import { importedTeachers, rootRecordPath, validRootRecord, type RootRecord } from './pipeline.js';

export const CASES_SCHEMA = 'natlang.tune-pool-cases/v1' as const;
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const json = (value: unknown) => JSON.stringify(value);
/** What a tune case binds: the program build, the site-aware scaffold and the split, checked again at import. */
export type TuneCaseBinding = { readonly schema: typeof CASES_SCHEMA; readonly program: string; readonly buildHash: string;
  readonly scaffold: string; readonly split: string; readonly root: string; readonly function: string; readonly input: unknown;
  readonly expected: unknown; readonly group: string | null; readonly component: string; readonly partition: string };
export type PoolContext = { readonly root: string; readonly program: ProgramDescriptor; readonly scaffold: WeightManifest;
  readonly examples: readonly SplitExample[]; readonly splitReceipt: unknown };

/** The serving system prompt the ordinary runtime renders for this program. */
function servingPromptSupported(program: ProgramDescriptor): void {
  // The collector runs the program's files only; program-level guidance, imports and services are not reproduced.
  if (program.sources['@natlang/program-policy.json'])
    throw new Error('this program declares program guidance, imported programs or services, which pool collection does not reproduce; ' +
      'collect it with `natlang tune` and a configured teacher instead');
}

const seedOf = (id: string) => Number.parseInt(sha(`natlang.tune-pool-seed/v1\0${id}`).slice(0, 8), 16) % 2 ** 31;

/** Arguments by the root function's parameter names (the collector binds inputs by name). */
function namedInputs(root: string, example: SplitExample): Record<string, unknown> {
  const fn = loadNatlang(resolve(root, example.function), root);
  const params = (fn as unknown as Record<symbol, { definition: { params: { name: string }[] } }>)[Symbol.for('natlang.callable')]!.definition.params;
  const input = example.input;
  if (input && typeof input === 'object' && !Array.isArray(input)) return input as Record<string, unknown>;
  const values = Array.isArray(input) ? input : [input];
  return Object.fromEntries(params.map((param, index) => [param.name, values[index]]));
}

/**
 * Write the case list for every collectable root whose data is missing or stale. Protected roots never leave the
 * program; confirmation roots only when asked; roots without `expected` are skipped (the pool needs an exact oracle).
 */
export function exportCases(context: PoolContext, out: string, options: { includeConfirmation?: boolean; teacherId?: string } = {}):
  { cases: number; skipped: { root: string; reason: string }[]; manifest: string } {
  const { root, program, scaffold, examples } = context;
  servingPromptSupported(program);
  if (existsSync(out) && readdirSync(out).length) throw new Error(`${out} is not empty; case lists are immutable, choose a new directory`);
  const partitions = new Set(['train', 'development', ...(options.includeConfirmation ? ['confirmation'] : [])]);
  const shared = sharedSourceHash(program);
  const files = Object.fromEntries(Object.entries(program.sources).filter(([path]) => !path.startsWith('@natlang/')));
  const scaffoldKey = trainingScaffoldKey(scaffold), split = sha(json(context.splitReceipt));
  const rows: string[] = [], cases: { index: number; seed: number }[] = [], skipped: { root: string; reason: string }[] = [];
  for (const example of examples) {
    if (!partitions.has(example.partition)) continue;
    if (validRootRecord(root, example, scaffold, shared, options.teacherId ?? '')) { skipped.push({ root: example.id, reason: 'valid data exists' }); continue; }
    if (!('expected' in example)) { skipped.push({ root: example.id, reason: 'no expected result (the pool needs an exact oracle)' }); continue; }
    const tune: TuneCaseBinding = { schema: CASES_SCHEMA, program: program.id, buildHash: program.buildHash, scaffold: scaffoldKey, split,
      root: example.id, function: example.function, input: example.input, expected: example.expected, group: example.group ?? null,
      component: example.component, partition: example.partition };
    cases.push({ index: rows.length, seed: seedOf(example.id) });
    rows.push(json({ version: 'natlang.program/2', id: `tune:${program.id}:${example.id}`, kind: 'lambda_source',
      semantics: { root: example.function, files, inputs: namedInputs(root, example), expected: example.expected },
      source_ids: [`tune:${program.id}:${example.id}`], source_groups: [`tune:${program.id}:${example.component}`],
      split: example.partition, tune }));
  }
  if (!rows.length) throw new Error('no root needs pool collection (all have valid data, or none has an expected result)');
  mkdirSync(out, { recursive: true });
  const sourceText = rows.join('\n') + '\n';
  writeFileSync(join(out, 'source.jsonl'), sourceText, { flag: 'wx' });
  writeFileSync(join(out, 'cases.json'), json({ cases }) + '\n', { flag: 'wx' });
  const manifest = { schema: CASES_SCHEMA, program: program.id, buildHash: program.buildHash, scaffold: scaffoldKey, split,
    source: { path: 'source.jsonl', sha256: sha(sourceText), rows: rows.length }, cases: { path: 'cases.json', sha256: sha(readFileSync(join(out, 'cases.json'))) },
    partitions: [...partitions], skipped };
  writeFileSync(join(out, 'manifest.json'), JSON.stringify(manifest, null, 1) + '\n', { flag: 'wx' });
  return { cases: rows.length, skipped, manifest: join(out, 'manifest.json') };
}

/** Collector result files below a path: a merged output, per-case `*.results.jsonl`, or a whole campaign root. */
function resultFiles(path: string): string[] {
  if (!statSync(path).isDirectory()) return [path];
  return readdirSync(path, { withFileTypes: true }).flatMap(entry => entry.isDirectory() ? resultFiles(join(path, entry.name))
    : entry.name.endsWith('.jsonl') && !entry.name.endsWith('.trace.jsonl') ? [join(path, entry.name)] : []).sort();
}

/** The exact actor and runtime identity of a collector row, as the tuner's `teacher` field. */
export function collectorTeacherId(provenance: Record<string, unknown>, reviewedActor?: Record<string, unknown>): string {
  const keys = ['model', 'provider', 'pi_options', 'transport', 'tool_schema', 'tool_surface_sha256', 'runtime', 'runtime_contract_version',
    'collector_version', 'execution_policy_version', 'system_prompt_sha256', 'prompt_rendering', 'execution_plans', 'temperature',
    'context_tokens', 'action_max_tokens', 'chat_request_controls', 'provider_request_controls', 'neuralese_server', 'collection_role'];
  const identity = { ...Object.fromEntries(keys.filter(key => key in provenance).map(key => [key, provenance[key]])),
    ...(reviewedActor ? { reviewed_dispatch_actor: reviewedActor } : {}) };
  return `collector:${String(provenance.provider ?? provenance.transport ?? 'unknown')}:${String(provenance.model)}:${sha(json(identity)).slice(0, 16)}`;
}

/** The serving system message for a collector system message: exactly one collection-guidance block removed. */
function servingSystem(content: string): string | null {
  for (const block of [GENERATION_GUIDANCE, GENERATION_GUIDANCE_AT_NL_DEPTH_LIMIT]) {
    const at = content.indexOf(block);
    if (at >= 0 && content.indexOf(block, at + 1) < 0 && content.startsWith(TOOLS_PROMPT.slice(0, 400)))
      return content.slice(0, at) + content.slice(at + block.length);
  }
  return null;
}

type Row = { task?: { program_ir?: { semantics?: { root?: string }; tune?: TuneCaseBinding } }; provenance?: Record<string, unknown>;
  outcome?: { status?: string; accepted?: boolean; oracle?: { accepted?: boolean; level?: string };
    checks?: Record<string, unknown>;
    invocation_ledger?: { invocation_id: string; parent_invocation_id?: string | null; completion_status?: string | null;
      completion_source?: string }[];
    action_ledger?: { call_id?: string; name?: string; outcome?: string; arguments?: unknown; seq?: number }[];
    execution_graph?: { kind?: string; phase?: string; call_id?: string; definition?: { id?: string }; context?: string; inputs?: { port?: string }[] }[] };
  trajectory?: { phase?: string; invocation_id?: string; context?: { role: string; content?: unknown }[]; tools_offered?: unknown[];
    model_response?: { calls?: unknown; text?: string; protocol_failure?: unknown };
    assistant?: { reasoning?: unknown; calls?: { source_tool?: string; tool?: string; arguments?: Record<string, unknown> }[] } }[] };

const blockedChecks = ['seeded_failure_requirement', 'expected_status', 'honest_stop', 'effects', 'files', 'world',
  'file_return_consistency', 'answer'];
export function honestTuneBlockedRoot(row: Row, expected: unknown): boolean {
  if (!expected || typeof expected !== 'object' || Array.isArray(expected) ||
      (expected as Record<string, unknown>).status !== 'blocked') return false;
  const outcome = row.outcome, checks = outcome?.checks, oracle = outcome?.oracle;
  if (outcome?.status !== 'quiesced' || outcome.accepted !== true || oracle?.accepted !== true || oracle.level !== 'exact' ||
      !checks || blockedChecks.some(key => checks[key] !== true) || Object.values(checks).some(value => value !== true)) return false;
  const ledger = outcome.invocation_ledger;
  if (!ledger) return false;
  const roots = ledger.filter(item => item.parent_invocation_id === null);
  if (roots.length !== 1) return false;
  const root = roots[0]!;
  if (root.completion_status !== 'quiesced' || root.completion_source !== 'execution_graph' || !root.invocation_id) return false;
  const children = ledger.filter(item => item !== root);
  if (children.some(item => item.completion_status !== 'done' || item.completion_source !== 'execution_graph')) return false;
  const rootCalls = (row.trajectory ?? []).filter(item => item.invocation_id === root.invocation_id)
    .flatMap(item => item.assistant?.calls ?? []);
  const terminal = rootCalls.at(-1), args = terminal?.arguments;
  if ((terminal?.source_tool ?? terminal?.tool) !== 'return_result' || !args || args.status !== 'blocked' ||
      typeof args.reason !== 'string' || !args.reason.trim() || canonicalJson(args) !== canonicalJson(expected)) return false;
  const rootEvents = (outcome.action_ledger ?? []).filter(event => event.call_id === root.invocation_id);
  const terminalEvents = rootEvents.filter(event => event.name === 'return_result');
  if (terminalEvents.length !== 1 || terminalEvents[0]?.outcome !== 'blocked' ||
      canonicalJson(terminalEvents[0].arguments) !== canonicalJson(args) || !Number.isSafeInteger(terminalEvents[0].seq) ||
      rootEvents.some(event => !Number.isSafeInteger(event.seq)) ||
      terminalEvents[0].seq !== Math.max(...rootEvents.map(event => Number(event.seq)))) return false;
  const childIds = new Set(children.map(item => item.invocation_id));
  for (const step of row.trajectory ?? []) if (step.invocation_id && childIds.has(step.invocation_id))
    for (const call of step.assistant?.calls ?? []) if ((call.source_tool ?? call.tool) === 'return_result' && call.arguments?.status !== 'success') return false;
  return true;
}

export type ImportSummary = { rows: number; imported: number; failures: number; rejected: { row: string; reason: string }[];
  unboundTurns: number; teachers: string[] };

export type TunePoolImportCustody = {
  schema: 'natlang.tune-pool-import-custody/2';
  input: { kind: 'file'; path: string; bytes: number; sha256: string } |
    { kind: 'directory'; path: string; files: { path: string; bytes: number; sha256: string }[] };
  plan: { path: string; bytes: number; sha256: string };
  source: { path: string; bytes: number; sha256: string };
  claim_ledger: { observed: { path: string; bytes: number; sha256: string; stat: Record<string, number>;
    prefix: { bytes: number; sha256: string } }; snapshot: { path: string; bytes: number; sha256: string } };
  launch_record: { observed: { path: string; bytes: number; sha256: string; stat: Record<string, number>;
    prefix: { bytes: number; sha256: string } }; snapshot: { path: string; bytes: number; sha256: string } };
  dispatcher_identity_sha256: string;
  reviewed_artifacts: { path: string; bytes: number; sha256: string }[];
  runtime: { root: string; receipt_sha256: string; receipt: { path: string; bytes: number; sha256: string } };
  source_inventory?: unknown;
  selected_rows: { case_index: number; input: { path: string; line: number }; result_row_sha256: string; result_line_sha256: string;
    result_file: { path: string; bytes: number; sha256: string }; result_manifest: { path: string; bytes: number; sha256: string };
    queue: { path: string; bytes: number; sha256: string }; source_row_sha256: string;
    provenance_sha256: string; execution_identity: Record<string, unknown>;
    actor: { provider: string; model: string; dispatch_plan_sha256: string; runtime_receipt_sha256: string;
      source_inventory_sha256?: string | null } }[];
};

function verifyImportInputBinding(path: string, custody: TunePoolImportCustody): void {
  const inputPath = resolve(path), input = custody.input;
  if (resolve(input.path) !== inputPath) throw new Error('reviewed campaign custody names a different import input');
  if (input.kind === 'file') {
    const bytes = readFileSync(inputPath);
    if (statSync(inputPath).isDirectory() || sha(bytes) !== input.sha256 || bytes.byteLength !== input.bytes)
      throw new Error('reviewed campaign import file changed after verification');
    return;
  }
  if (!statSync(inputPath).isDirectory()) throw new Error('reviewed campaign import directory is no longer present');
  const expected = [...input.files].sort((a, b) => a.path.localeCompare(b.path));
  if (new Set(expected.map(item => item.path)).size !== expected.length)
    throw new Error('reviewed campaign cohort repeats a result file');
  for (const binding of expected) verifyCustodyFile(binding, 'selected closed-case result file');
}

function verifyCustodyFile(binding: { path: string; bytes: number; sha256: string }, label: string): void {
  const bytes = readFileSync(binding.path);
  if (bytes.byteLength !== binding.bytes || sha(bytes) !== binding.sha256)
    throw new Error(`reviewed dispatcher ${label} changed after verification`);
}

function verifyCampaignSnapshot(binding: TunePoolImportCustody['claim_ledger'], label: string): void {
  const { observed, snapshot } = binding;
  if (!observed || !snapshot || !observed.path || observed.bytes !== observed.stat?.bytes ||
      observed.prefix?.bytes !== observed.bytes || observed.prefix?.sha256 !== observed.sha256 ||
      observed.sha256 !== snapshot.sha256 || observed.bytes !== snapshot.bytes)
    throw new Error(`reviewed dispatcher ${label} observation and snapshot bindings differ`);
  verifyCustodyFile(snapshot, `${label} snapshot`);
}

/**
 * Import collector result rows produced from an exported case list. A row is rejected unless its tune binding names
 * this program build, this scaffold and this split; a root that the collector could not finish is not recorded (it is
 * collected again later). Turns of invocations without a binding of their own keep `binding: null` and never train.
 */
export function importCases(context: PoolContext, path: string, custody: TunePoolImportCustody): ImportSummary {
  const { root, program, scaffold, examples } = context;
  if (custody?.schema !== 'natlang.tune-pool-import-custody/2') throw new Error('tune result import requires immutable reviewed dispatcher custody');
  verifyImportInputBinding(path, custody);
  for (const [label, binding] of [['plan', custody.plan], ['source', custody.source],
    ['runtime receipt', custody.runtime.receipt]] as const)
    verifyCustodyFile(binding, label);
  verifyCampaignSnapshot(custody.claim_ledger, 'claim ledger');
  verifyCampaignSnapshot(custody.launch_record, 'launch record');
  for (const binding of custody.reviewed_artifacts) verifyCustodyFile(binding, 'reviewed plan artifact');
  if (custody.runtime.receipt.sha256 !== custody.runtime.receipt_sha256)
    throw new Error('reviewed dispatcher runtime receipt identity changed');
  const scaffoldKey = trainingScaffoldKey(scaffold), split = sha(json(context.splitReceipt)), shared = sharedSourceHash(program);
  const sourceRows = readFileSync(custody.source.path, 'utf8').split('\n').map((line, index) => line.trim()
    ? { index, row: JSON.parse(line) } : null).filter((item): item is { index: number; row: unknown } => item !== null);
  const byId = new Map(examples.map(example => [example.id, example]));
  const binding = new WeightBinding(scaffold);
  const components = new Map(program.components.map(component => [component.key, component]));
  const semantic = new Map(scaffold.bindings.map(entry => [entry.id, entry.semanticHash]));
  const summary: ImportSummary = { rows: 0, imported: 0, failures: 0, rejected: [], unboundTurns: 0, teachers: [] };
  const teachers = new Set(importedTeachers(root));
  const importFiles = custody.input.kind === 'file' ? [resolve(path)] : custody.input.files.map(item => resolve(item.path));
  if (custody.input.kind === 'directory') {
    const selectedResultFiles = [...new Set(custody.selected_rows.map(item => resolve(item.result_file.path)))].sort();
    if (selectedResultFiles.length !== importFiles.length || selectedResultFiles.some((file, index) => file !== [...importFiles].sort()[index]))
      throw new Error('reviewed directory input does not exactly name the closed-case result-file cohort');
  }
  for (const file of importFiles) {
    const fileText = readFileSync(file, 'utf8'), fileBytes = Buffer.from(fileText), lines = fileText.split('\n');
    for (const [index, line] of lines.entries()) {
    if (!line.trim()) continue;
    let row: Row;
    // A campaign root also holds ledgers, queues and journals; only collector result rows carry a tune binding.
    try { row = JSON.parse(line) as Row; } catch { continue; }
    const where = `${file}:${index + 1}`;
    const tune = row.task?.program_ir?.tune;
    if (tune?.schema !== CASES_SCHEMA) continue; // not a tune case
    summary.rows++;
    const reject = (reason: string) => { summary.rejected.push({ row: where, reason }); };
    if (tune.program !== program.id || tune.buildHash !== program.buildHash) { reject('collected from another build of the program'); continue; }
    if (tune.scaffold !== scaffoldKey) { reject('collected against another scope tree or site assignment'); continue; }
    if (tune.split !== split) { reject('collected under another split receipt'); continue; }
    const example = byId.get(tune.root);
    if (!example || example.partition !== tune.partition) { reject('the root is no longer a collectable example in that partition'); continue; }
    const provenance = row.provenance ?? {}, outcome = row.outcome ?? {};
    if (!outcome.status) { reject('the collector recorded no outcome'); continue; }
    const sourceMatches = sourceRows.filter(item => canonicalJson(item.row) === canonicalJson(row.task?.program_ir));
    if (sourceMatches.length !== 1) { reject('complete tune task does not identify exactly one reviewed source case'); continue; }
    const caseIndex = sourceMatches[0]!.index;
    const matchingCustody = custody.selected_rows.filter(item => item.case_index === caseIndex &&
      resolve(item.input.path) === resolve(file) && item.input.line === index + 1);
    const rowCustody = matchingCustody[0];
    if (matchingCustody.length !== 1 || !rowCustody || !/^[0-9a-f]{64}$/.test(rowCustody.result_row_sha256) ||
        !/^[0-9a-f]{64}$/.test(rowCustody.result_line_sha256)) {
      reject('result row is not bound to this exact reviewed campaign case and result file'); continue;
    }
    const exactOutput = fileBytes;
    if (exactOutput.byteLength !== rowCustody.result_file.bytes || sha(exactOutput) !== rowCustody.result_file.sha256) {
      reject('result output changed after campaign verification'); continue;
    }
    const rawLine = Buffer.from(line + (index < lines.length - 1 ? '\n' : ''));
    if (sha(rawLine) !== rowCustody.result_line_sha256) { reject('result row bytes differ from the exact reviewed campaign line'); continue; }
    for (const [label, fileBinding] of [['result manifest', rowCustody.result_manifest], ['case queue', rowCustody.queue]] as const)
      verifyCustodyFile(fileBinding, label);
    if (canonicalJson(sourceRows.find(item => item.index === caseIndex)?.row) !== canonicalJson(row.task?.program_ir)) {
      reject('result provenance or complete source case differs from reviewed dispatcher custody'); continue;
    }
    const teacher = collectorTeacherId(provenance, rowCustody.actor);
    // Invocation → definition (the root is named by its source path) → the binding resolved exactly as at serving time.
    const starts = (outcome.execution_graph ?? []).filter(event => event.kind === 'invocation' && event.phase === 'start' && event.call_id);
    const definitionOf = new Map<string, { definition: string | null; context: string | null }>(starts.map(event => {
      const isRoot = !(event.inputs ?? []).some(input => input.port === 'caller');
      return [event.call_id!, { definition: isRoot ? `nl:${row.task!.program_ir!.semantics!.root}` : event.definition?.id ?? null,
        context: !event.context || event.context === 'ctx:files' ? null : event.context }];
    }));
    const callOutcomes = new Map((outcome.invocation_ledger ?? []).map(item => [item.invocation_id, item.completion_status ?? null]));
    const ok = outcome.status === 'done', oracle = 'expected' in example ? outcome.oracle?.accepted === true : null;
    const blockedRoot = honestTuneBlockedRoot(row, example.expected);
    const rootAccepted = ok && outcome.accepted === true && oracle !== false || blockedRoot;
    const turns: CollectedTurn[] = [], counters = new Map<string, number>();
    let bad: string | undefined;
    for (const step of row.trajectory ?? []) {
      if (step.phase !== 'action') continue;
      const messages = structuredClone(step.context ?? []);
      const system = messages[0];
      const served = system?.role === 'system' && typeof system.content === 'string' ? servingSystem(system.content) : null;
      if (served === null) { bad = 'a system message is not the collector prompt with exactly one collection-guidance block'; break; }
      messages[0] = { ...system!, content: served };
      const call = step.invocation_id ?? null, turn = (counters.get(call ?? '') ?? 0) + 1;
      counters.set(call ?? '', turn);
      const site = call ? definitionOf.get(call) : undefined;
      const resolved = site?.definition ? binding.resolve(site.definition, undefined, site.context) : undefined;
      const entry = resolved?.binding ? scaffold.bindings.find(item => item.id === resolved.binding) : undefined;
      if (!entry) summary.unboundTurns++;
      const component = entry ? components.get(entry.component) : undefined;
      const instructions = component?.baseline.kind === 'lambda.instructions' && component.origin === 'named'
        ? component.baseline.template.segments.join('') : undefined;
      const response = step.model_response ?? {};
      turns.push({ schema: 'natlang.tune-turn/v1', root: example.id, component: example.component, partition: example.partition,
        binding: entry?.id ?? null, definition: site?.definition ?? null, call, turn, messages, tools: structuredClone(step.tools_offered ?? []),
        reply: { ...(Array.isArray(response.calls) && response.calls.length ? { calls: response.calls as never } : {}),
          ...(response.text !== undefined ? { text: response.text } : {}),
          ...(typeof step.assistant?.reasoning === 'string' && step.assistant.reasoning ? { reasoning: step.assistant.reasoning } : {}),
          // A turn the collector flagged as a protocol failure is never a training target.
          ...(response.protocol_failure ? { truncated: true } : {}) },
        teacher, source: example.source ?? 'real', texts: instructions !== undefined ? { instructions } : {},
        evidence: { rootOracle: oracle, enclosingSuccess: rootAccepted, callOutcome: call ? callOutcomes.get(call) ?? null : null } });
    }
    if (bad) { reject(bad); continue; }
    const bindings = Object.fromEntries([...new Set(turns.map(turn => turn.binding).filter((id): id is string => !!id))].map(id => [id, semantic.get(id) ?? '']));
    const record: RootRecord = { schema: 'natlang.tune-root/v1', root: example.id, teacher, sharedSources: shared, bindings,
      outcome: { ok: rootAccepted, oracle }, turns, imported: { from: resolve(file), line: index + 1,
        result_row_sha256: rowCustody.result_row_sha256, result_line_sha256: rowCustody.result_line_sha256,
        campaign: { plan: custody.plan, source: custody.source, claim_ledger: custody.claim_ledger, launch_record: custody.launch_record,
          runtime_receipt: custody.runtime.receipt, dispatcher_identity_sha256: custody.dispatcher_identity_sha256,
          reviewed_artifacts: custody.reviewed_artifacts, source_inventory: custody.source_inventory ?? null },
        result_file: rowCustody.result_file, result_manifest: rowCustody.result_manifest, queue: rowCustody.queue,
        source_row_sha256: rowCustody.source_row_sha256, provenance_sha256: rowCustody.provenance_sha256,
        execution_identity: rowCustody.execution_identity, actor: rowCustody.actor,
        program_ir_sha256: provenance.program_ir_sha256 ?? null,
        model: provenance.model ?? null, provider: provenance.provider ?? null, system_prompt_sha256: provenance.system_prompt_sha256 ?? null,
        rerendered: 'collector system prompt with its collection-guidance block removed' } };
    const target = rootRecordPath(root, example, teacher);
    mkdirSync(join(tuneDirectory(root), 'data', 'roots'), { recursive: true });
    const encoded = JSON.stringify(record);
    if (existsSync(target)) {
      const existing = JSON.parse(readFileSync(target, 'utf8')) as RootRecord;
      const oldImport = existing.imported as Record<string, unknown> | undefined;
      const newImport = record.imported as Record<string, unknown> | undefined;
      const oldActor = oldImport?.actor, newActor = newImport?.actor;
      const sameClosedResult = existing.root === record.root && existing.teacher === record.teacher &&
        oldImport?.result_row_sha256 === newImport?.result_row_sha256 &&
        oldImport?.result_line_sha256 === newImport?.result_line_sha256 &&
        oldImport?.source_row_sha256 === newImport?.source_row_sha256 &&
        oldActor !== undefined && newActor !== undefined &&
        JSON.stringify(oldActor) === JSON.stringify(newActor);
      if (!sameClosedResult) {
        reject('this campaign actor/root already has different imported bytes'); continue;
      }
      // Campaign supervision files advance as other cases close. Keep the first
      // immutable terminal-evidence snapshot for this exact row and make retries idempotent.
      teachers.add(teacher);
      summary.imported++;
      if (!rootAccepted) summary.failures++;
      continue;
    }
    writeFileSync(target, encoded);
    teachers.add(teacher);
    summary.imported++;
    if (!rootAccepted) summary.failures++;
    }
  }
  summary.teachers = [...teachers].sort();
  mkdirSync(join(tuneDirectory(root), 'data'), { recursive: true });
  writeFileSync(join(tuneDirectory(root), 'data', 'imports.json'), JSON.stringify({ teachers: summary.teachers }, null, 1) + '\n');
  return summary;
}
