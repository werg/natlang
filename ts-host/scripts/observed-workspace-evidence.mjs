import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, resolve, posix } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { graphCustodyMatches } from '../dist/calls/graph-custody.js';

export function verifyApplicationReturn(binding, records, roots) {
  if (!binding || !Array.isArray(roots) || roots.length !== 1 || binding.root_call_id !== roots[0] || typeof binding.valueJson !== 'string')
    throw new Error('application return does not identify its actual authored root');
  const value = JSON.parse(binding.valueJson), output = records.get(binding.root_call_id)?.output;
  if (output?.complete !== true || output.bytes !== Buffer.byteLength(binding.valueJson) || output.hash !== digest(binding.valueJson))
    throw new Error('application return differs from original root output');
  if (!Object.hasOwn(binding,'selectedResult') || !(isDeepStrictEqual(value,binding.selectedResult) ||
      value?.results?.length === 1 && Object.hasOwn(value.results[0],'report') && isDeepStrictEqual(value.results[0].report,binding.selectedResult)))
    throw new Error('selected domain return differs from actual application result');
}

export async function validateReassessmentCheckRoles(proof, originalChecks, root) {
  const checks = proof.checks;
  if (!checks || Array.isArray(checks) || !Object.keys(checks).length || Object.values(checks).some(value=>value!==true) ||
      !originalChecks || !Object.keys(originalChecks).length)
    throw new Error('completed reassessment needs positive independent required checks');
  if (isDeepStrictEqual(Object.keys(checks).sort(),Object.keys(originalChecks).sort())) return;
  const contract=proof.outcomeContract, dimensions=proof.outcomeDimensions;
  if (!contract || !dimensions) throw new Error('omitted reassessment checks need source-bound outcome roles');
  const bindings=(proof.checker_sources??[]).filter(item=>item.role==='outcome-contract');
  if(bindings.length!==1 || !isDeepStrictEqual(JSON.parse(await readBound(bindings[0],root)),contract))
    throw new Error('reassessment outcome roles differ from the bound checker contract');
  const names=[],required=[];
  for(const role of ['effects','returnedValue','incidentalValues']) {
    const declaration=contract[role],observed=dimensions[role];
    if(role==='incidentalValues' && declaration===undefined && observed===undefined)continue;
    if(!declaration || !observed)throw new Error('reassessment check roles must account for every original observation');
    const roleNames=declaration.checks,needed=role==='effects'||role==='returnedValue'&&declaration.required===true;
    if(!Array.isArray(roleNames)||roleNames.some(name=>typeof name!=='string'||!name)||
       typeof declaration.rationale!=='string'||!declaration.rationale.trim()||role==='effects'&&!roleNames.length||
       role==='returnedValue'&&(typeof declaration.required!=='boolean'||observed.required!==declaration.required))
      throw new Error('reassessment check roles need explicit requirements and rationales');
    const expected=Object.fromEntries(roleNames.map(name=>[name,needed?checks[name]:originalChecks[name]]));
    if(Object.values(expected).some(value=>typeof value!=='boolean')||!isDeepStrictEqual(observed.checks,expected)||
       observed.passed!==Object.values(expected).every(Boolean))
      throw new Error('reassessment must preserve optional observations and prove required roles');
    names.push(...roleNames);if(needed)required.push(...roleNames);
  }
  if(new Set(names).size!==names.length||!isDeepStrictEqual([...names].sort(),Object.keys(originalChecks).sort())||
     !isDeepStrictEqual([...required].sort(),Object.keys(checks).sort()))
    throw new Error('reassessment roles must partition the exact original check set');
}

function declaredReturnSelectors(proof) {
  const incidental=proof.outcomeDimensions?.incidentalValues;
  if(incidental?.passed!==false)return [];
  if(!Array.isArray(incidental.incorrectReturns))throw new Error('incorrect incidental returns need exact original value bindings');
  const contains=(value,wanted)=>isDeepStrictEqual(value,wanted)||value!==null&&typeof value==='object'&&Object.values(value).some(item=>contains(item,wanted));
  for(const item of incidental.incorrectReturns){
    if(!item.definition?.name||!item.definition?.source||typeof item.valueJson!=='string'||createHash('sha256').update(item.valueJson).digest('hex')!==item.sha256)throw new Error('incorrect return value identity/hash differs');
    if(!contains(proof.actualResult,JSON.parse(item.valueJson)))throw new Error('incorrect return value is outside the actual application result');
  }
  return incidental.incorrectReturns;
}

async function readBound(binding, root) {
  if (!binding?.path || !/^[a-f\d]{64}$/u.test(binding.sha256 ?? ''))
    throw new Error('invalid observed workspace artifact binding');
  const bytes = await readFile(resolve(root, binding.path));
  if (createHash('sha256').update(bytes).digest('hex') !== binding.sha256)
    throw new Error(`observed workspace artifact changed: ${binding.path}`);
  return bytes;
}

export function verifyObservedCallTree(records, root) {
  const byId = new Map();
  const children = new Map();
  for (const record of records) {
    if (byId.has(record.call_id)) throw new Error('observed call graph repeats an invocation');
    byId.set(record.call_id, record);
    const group = children.get(record.parent_call_id) ?? [];
    group.push(record.call_id);
    children.set(record.parent_call_id, group);
  }
  const hasCustody = records.some(record => record.graph_custody != null);
  const pending = [root];
  const seen = new Set();
  while (pending.length) {
    const id = pending.pop();
    const call = byId.get(id);
    const ids = children.get(id) ?? [];
    if (!call || seen.has(id) || (id === root && call.parent_call_id !== null))
      throw new Error('observed call graph has invalid parent links');
    if (hasCustody && (!graphCustodyMatches(call, ids) ||
        ids.some(child => byId.get(child).task_id !== call.task_id)))
      throw new Error('observed call graph differs from recorded child inventory');
    seen.add(id);
    pending.push(...ids);
  }
  if (seen.size !== records.length) throw new Error('observed call graph contains unreachable invocations');
}

const digest = bytes => createHash('sha256').update(bytes).digest('hex');

function verifyEntry(interaction, record, partial) {
  const entry = interaction.entry ?? {program: 'skills/open-user-goal/runUserGoal.nl',
    callable: 'runUserGoal', arguments: {goal: interaction.goal}};
  if (typeof entry.program !== 'string' || !/\.(nl|ts)$/u.test(entry.program) ||
      entry.program.startsWith('/') || entry.program.split('/').includes('..') ||
      interaction.program !== entry.program || typeof entry.callable !== 'string' || !entry.callable ||
      !entry.arguments || typeof entry.arguments !== 'object' || Array.isArray(entry.arguments) ||
      record.definition?.name !== entry.callable || (!partial && record.outcome !== 'done') ||
      !isDeepStrictEqual(Object.keys(record.inputs ?? {}).sort(), Object.keys(entry.arguments).sort()))
    throw new Error('observed root differs from its actual application entry');
  if (interaction.entry) {
    const {program_root: programRoot, definition: {source}} = record;
    if (typeof programRoot !== 'string' || typeof source !== 'string' ||
        programRoot.startsWith('/') || source.startsWith('/') ||
        programRoot.split('/').includes('..') || source.split('/').includes('..') ||
        posix.join(programRoot, source) !== entry.program ||
        !entry.argumentJson || !isDeepStrictEqual(Object.keys(entry.argumentJson).sort(), Object.keys(entry.arguments).sort()))
      throw new Error('observed application entry needs its actual source and exact native argument JSON');
  }
  for (const [name, value] of Object.entries(entry.arguments)) {
    const text = interaction.entry ? entry.argumentJson[name] : JSON.stringify(value);
    if (typeof text !== 'string' || !isDeepStrictEqual(JSON.parse(text), value))
      throw new Error('observed application argument JSON differs from its declared value');
    const bytes = Buffer.from(text), input = record.inputs[name];
    if (input?.complete !== true || input.bytes !== bytes.length || input.hash !== digest(bytes))
      throw new Error('observed application input differs from the captured invocation');
  }
}

async function trainingScope(receipt, row, root) {
  const scope = receipt.training_scope;
  if (!scope) {
    if (receipt.status !== 'passed' || row.outcome.training_scope != null)
      throw new Error('partial observed decisions require independently attested training scope');
    return null;
  }
  if (receipt.status !== 'partial' || scope.schema !== 'natlang.training-scope/1' ||
      !isDeepStrictEqual(scope, row.outcome.training_scope) ||
      JSON.parse(await readBound(scope.original_outcome, root)).passed !== false ||
      !Array.isArray(scope.invocations) || !scope.invocations.length)
    throw new Error('invalid independent partial training scope');
  if(scope.assessor)await readBound(scope.assessor,root);
  const selected = new Map();
  for (const invocation of scope.invocations) {
    if (typeof invocation.call_id !== 'string' || selected.has(invocation.call_id) ||
        !invocation.checks || !Object.keys(invocation.checks).length ||
        Object.values(invocation.checks).some(value => value !== true) ||
        !Array.isArray(invocation.evidence) || !invocation.evidence.length ||
        (invocation.ordinals != null && (!Array.isArray(invocation.ordinals) || !invocation.ordinals.length ||
          new Set(invocation.ordinals).size !== invocation.ordinals.length ||
          invocation.ordinals.some(index => !Number.isSafeInteger(index) || index < 0))))
      throw new Error('partial invocation lacks independently verified decision evidence');
    for (const evidence of invocation.evidence) {
      if (!['workspace', 'cdp'].includes(evidence.kind)) throw new Error('partial decisions require native workspace evidence');
      await readBound(evidence, root);
    }
    selected.set(invocation.call_id, invocation.ordinals == null ? null : new Set(invocation.ordinals));
  }
  return selected;
}

async function completedReassessment(receipt, manifest, row, root) {
  const reassessment=receipt.reassessment;
  if (!reassessment) {
    if (row.outcome.reassessment) throw new Error('unattested completed reassessment');
    return;
  }
  if (receipt.status!=='passed' || receipt.training_scope || !isDeepStrictEqual(reassessment,row.outcome.reassessment))
    throw new Error('completed reassessment cannot relabel partial decisions');
  const original=JSON.parse(await readBound(reassessment.original_outcome,root));
  const assessment=JSON.parse(await readBound(reassessment.assessment,root));
  const validator=JSON.parse(await readBound(receipt.validator,root));
  const raw=await readBound(validator.original_packet,root);
  const packet=JSON.parse(validator.original_packet.path.endsWith('.gz')?gunzipSync(raw):raw);
  const native=packet.execution.diagnostics.natlangTeacher;
  const prior=packet.execution.diagnostics.openGoalVerification;
  const captured=await Promise.all(manifest.calls.map(async binding=>JSON.parse(await readBound(binding,root))));
  const byId=calls=>Object.fromEntries(calls.map(item=>[item.callId,item]));
  if(native.calls?.length!==captured.length || !isDeepStrictEqual(byId(native.calls),byId(captured)) ||
      !isDeepStrictEqual(native.rootCallIds,receipt.root_call_ids) ||
      !isDeepStrictEqual(native.entry??null,manifest.interaction?.entry??null) || prior.goal!==manifest.interaction?.goal)
    throw new Error('completed reassessment does not bind exact original application graph');
  const workspace=receipt.evidence.filter(item=>item.kind==='workspace');
  if(workspace.length!==1)throw new Error('completed reassessment requires one independent whole-outcome proof');
  const proof=JSON.parse(await readBound(workspace[0],root)).proof;
  const originalCompleted=JSON.parse(await readBound(validator.completed_outcome,root));
  const normalized=value=>({...value,...Object.fromEntries(['evidence','checker_sources'].map(field=>[field,(value[field]??[]).map(({path,...binding})=>binding)]))});
  if(assessment.completed_outcome?.sha256!==validator.completed_outcome.sha256 || !isDeepStrictEqual(normalized(proof),normalized(originalCompleted)))
    throw new Error('completed reassessment differs from bound independent verdict');
  if(original.passed!==false || !isDeepStrictEqual(original,packet.result) ||
      assessment.schema!=='natlang.live-reassessment/1' || assessment.task_id!==manifest.id || assessment.original_sha256!==digest(raw) ||
      proof?.schema!=='natlang.live-outcome-reassessment/1' || proof.passed!==true || proof.task_id!==manifest.id ||
      proof.original_sha256!==assessment.original_sha256 || proof.goal!==prior.goal ||
      !isDeepStrictEqual(proof.entry??null,native.entry??null) || !isDeepStrictEqual(proof.actualResult??null,native.report??null) ||
      !proof.checks || !Object.keys(proof.checks).length ||
      Object.values(proof.checks).some(value=>value!==true) || !proof.rationale ||
      !Array.isArray(proof.evidence) || !proof.evidence.length || !Array.isArray(proof.checker_sources) || !proof.checker_sources.length)
    throw new Error('completed reassessment differs from actual original execution or independent outcome');
  await validateReassessmentCheckRoles(proof,prior.checks,root);
  for(const artifact of [...proof.evidence,...proof.checker_sources])await readBound(artifact,root);
}

function decisionEligibility(record, detail, selected, incorrectReturn) {
  const actual = record.model_turns ?? [];
  const events = detail.events ?? [];
  const failed = new Set();
  for (const event of events.filter(event => event.kind === 'action' && ['error', 'rejected', 'blocked', 'budget'].includes(event.outcome))) {
    const owners = actual.flatMap((exchange, index) => exchange.kind === 'generation' &&
      exchange.reply?.raw_calls?.some(call => call.id === event.tool_call_id) ? [index] : []);
    if (!event.tool_call_id || owners.length !== 1) throw new Error('failed action lacks exact captured generation ownership');
    failed.add(owners[0]);
  }
  const actionEvents = Map.groupBy(events.filter(event => event.kind === 'action'), event => event.tool_call_id);
  const ordinals = selected?.get(record.call_id);
  if (selected?.has(record.call_id) && ((ordinals === null && record.outcome !== 'done') ||
      (ordinals !== null && [...ordinals].some(index => actual[index]?.kind !== 'generation'))))
    throw new Error('partial scope names an invalid invocation or generation');
  const executor = Boolean(record.executor?.kind !== 'crisp' && (record.outcome === 'done' ||
    selected?.has(record.call_id) && ordinals !== null));
  return actual.map((exchange, index) => {
    if (exchange.schema !== 'natlang.model-exchange/1') throw new Error('unsupported captured model exchange');
    if (exchange.kind !== 'generation' || exchange.error) return false;
    const reply = exchange.reply, gateway = reply?.raw_response?.vibestudio;
    const content = gateway?.response?.content;
    if (!Array.isArray(exchange.request?.messages) || !gateway?.model || !gateway.api || !gateway.turnId ||
        !gateway.attribution?.callerId || !gateway.attribution?.callerKind || !gateway.context ||
        gateway.response?.role !== 'assistant' || !Array.isArray(content) || !Array.isArray(reply.raw_calls))
      throw new Error('observed generation lacks its original gateway exchange');
    const originalCalls = content.filter(block => block.type === 'toolCall');
    if (reply.raw_calls.length !== originalCalls.length || originalCalls.some((call, offset) => {
      const raw = reply.raw_calls[offset];
      return !call.id || raw.id !== call.id || raw.function?.name !== call.name ||
        !isDeepStrictEqual(JSON.parse(raw.function?.arguments ?? 'null'), call.arguments);
    }) || !isDeepStrictEqual(reply.calls ?? [], originalCalls.map(call => [call.name, call.arguments])) ||
        (reply.text ?? '') !== content.filter(block => block.type === 'text').map(block => block.text).join('') ||
        (reply.reasoning ?? '') !== content.filter(block => block.type === 'thinking').map(block => block.thinking).join('\n'))
      throw new Error('observed normalized model reply differs from original gateway response');
    return (!selected || selected.has(record.call_id) && (ordinals === null || ordinals.has(index))) &&
      index !== incorrectReturn && executor && !failed.has(index) &&
      originalCalls.every(call => actionEvents.get(call.id)?.length === 1 &&
        ['ok', 'completed'].includes(actionEvents.get(call.id)[0].outcome)) &&
      Boolean(reply.text || originalCalls.length);
  });
}

/** Verify the on-disk artifacts before allowing the generic native exporter to use an observed capture. */
export async function verifyObservedWorkspaceEvidence(row) {
  const binding = row.provenance.source_manifest;
  const root = dirname(binding.path);
  const manifest = JSON.parse(await readBound(binding, process.cwd()));
  const attestation = manifest.verification;
  const receipt = JSON.parse(await readBound(attestation, root));
  const equal = isDeepStrictEqual;
  if (row.provenance.execution_kind !== 'observed-live-workspace' || row.task.source !== 'observed-live-workspace' ||
      row.outcome.accepted !== true || manifest.split !== 'train' || !equal(manifest.learner_kinds, ['natlang-executor']) ||
      manifest.schema !== 'natlang.live-teacher-task/1' || receipt.schema !== 'natlang.live-task-verification/1' ||
      !equal(attestation, row.provenance.verification) || receipt.task_id !== row.task.id ||
      (receipt.split ?? manifest.split) !== manifest.split ||
      receipt.run_id !== row.outcome.world.run_id || !['passed', 'partial'].includes(receipt.status) ||
      receipt.status !== row.outcome.status || (receipt.status === 'partial' ? row.outcome.task_completed !== false : row.outcome.task_completed === false) ||
      !receipt.checks || !Object.keys(receipt.checks).length || Object.values(receipt.checks).some(value => value !== true) ||
      !['id', 'family', 'split', 'source_namespace', 'source_ids', 'source_groups', 'source_revisions', 'interaction'].every(
        key => equal(manifest[key], row.task[key])) ||
      !equal(receipt.interaction, row.task.interaction) || !equal(manifest.world, receipt.world) ||
      !equal(receipt.world, row.outcome.world) || !equal(receipt.checks, row.outcome.checks) ||
      !equal(receipt.root_call_ids, row.outcome.root_call_ids) ||
      !equal(manifest.code, row.provenance.code) || manifest.code?.dirty !== false)
    throw new Error('observed workspace trajectory differs from the independently verified manifest');
  const evidenceKinds = new Set();
  let incorrectOptionalReturn = false;
  const returnSelectors=[];
  const applicationReturns=[];
  const matchedReturnSelectors=new Set();
  for (const evidence of receipt.evidence ?? []) {
    const evidenceBytes = await readBound(evidence, root);
    evidenceKinds.add(evidence.kind);
    if (evidence.kind === 'workspace') {
      const payload=JSON.parse(evidenceBytes);
      if(Object.hasOwn(payload,"applicationReturn"))applicationReturns.push(payload.applicationReturn);
      const proof=payload.proof??{};
      returnSelectors.push(...declaredReturnSelectors(proof));
      const returned = proof.outcomeDimensions?.returnedValue;
      incorrectOptionalReturn ||= returned?.required === false && returned.passed === false;
    }
  }
  if (!Array.isArray(manifest.required_effect_kinds) || !manifest.required_effect_kinds.length ||
      manifest.required_effect_kinds.some(kind => !['workspace', 'cdp'].includes(kind) || !evidenceKinds.has(kind)))
    throw new Error('observed workspace outcome evidence is incomplete');
  await readBound(receipt.validator, root);

  await completedReassessment(receipt, manifest, row, root);
  const selected = await trainingScope(receipt, row, root);
  const decisions = Map.groupBy(row.trajectory, item => item.invocation_id);
  const returnExclusions = [];
  const capturedRecords = new Map();
  const captureScopes = new Map(receipt.calls.map(scope => [scope.call_id, scope]));
  const callFiles = new Map(manifest.calls.map(artifact => [artifact.path, artifact]));
  if (captureScopes.size !== receipt.calls.length || callFiles.size !== manifest.calls.length)
    throw new Error('observed capture repeats an attested call or artifact');
  const roots = new Set(receipt.root_call_ids);
  const goal = row.task.interaction?.goal;
  const records = [];
  if (row.task.interaction?.kind !== 'open-user-goal' || roots.size !== 1 || typeof goal !== 'string' || !goal.trim())
    throw new Error('observed open goal requires one independently attested application root');
  for (const call of row.outcome.invocation_ledger) {
    const original = JSON.parse(await readBound(call.artifact, root));
    const record = original.record;
    capturedRecords.set(record.call_id, record);
    records.push({
      call_id: record.call_id, parent_call_id: record.parent_call_id, task_id: record.task_id,
      graph_custody: record.graph_custody, effects: record.effects,
    });
    if (record.call_id !== call.call_id || record.version !== 'natlang.calls/1' ||
        original.summary.definition_id !== record.definition?.id ||
        !['agent', 'crisp', 'crisp-agent'].includes(record.executor?.kind) ||
        !equal(call.artifact, callFiles.get(call.artifact.path)) ||
        original.callId !== call.call_id || !equal(call.scope, captureScopes.get(call.call_id)) ||
        ['call_id', 'owner_kind', 'owner_key', 'run_id'].some(key => original.summary[key] !== call.scope[key]) ||
        original.summary.parent_call_id !== call.parent_call_id ||
        !equal(record.definition, call.definition) || !equal(record.executor, call.executor) ||
        !equal(record.parent_call_id, call.parent_call_id) || record.outcome !== call.outcome ||
        !equal(record.inputs ?? null, call.inputs) || !equal(record.output ?? null, call.output) ||
        !equal(record.graph_custody ?? null, call.graph_custody) ||
        ['parent_action_index', 'started_at', 'ended_at'].some(key => !equal(record[key] ?? null, call[key] ?? null)) ||
        !equal(original.events ?? [], call.events) || !equal(record.effects ?? [], call.effects))
      throw new Error('observed invocation differs from its captured call or verified scope');
    if (record.parent_call_id === null) {
      if (!roots.delete(call.call_id)) throw new Error('observed root was not independently attested');
      verifyEntry(row.task.interaction, record, selected !== null);
    }

    const actual = record.model_turns ?? [];
    const projected = decisions.get(call.call_id) ?? [];
    let selectorMatch=false;
    returnSelectors.forEach((item,index)=>{
      const matches=record.definition?.name===item.definition.name&&record.definition?.source===item.definition.source&&record.output?.complete===true&&record.output.hash===item.sha256&&record.output.bytes===Buffer.byteLength(item.valueJson);
      if(matches){matchedReturnSelectors.add(index);selectorMatch=true;}
    });
    const lastGeneration = selectorMatch || incorrectOptionalReturn && record.parent_call_id === null
      ? actual.findLastIndex(exchange => exchange.kind === 'generation') : -1;
    if (lastGeneration >= 0) returnExclusions.push({invocation_id: call.call_id, ordinal: lastGeneration});
    const eligible = decisionEligibility(record, original, selected, lastGeneration);
    if (projected.length !== actual.length)
      throw new Error('observed trajectory omitted captured exchanges');
    for (const [index, decision] of projected.entries()) {
      const exchange = actual[index];
      if (decision.training_admission?.approved !== eligible[index] ||
          index !== decision.live_exchange.ordinal ||
          !equal(decision.live_exchange.call_artifact, call.artifact) ||
          decision.parent_invocation_id !== record.parent_call_id ||
          decision.function !== record.definition.name || decision.definition_id !== record.definition.id ||
          !equal(decision.context, exchange.request?.messages ?? null) ||
          !equal(decision.tools_offered, exchange.request?.tools ?? []) ||
          !equal(decision.live_exchange.gateway, exchange.reply?.raw_response?.vibestudio ?? null) ||
          decision.phase !== (exchange.kind === 'generation' ? 'action' : exchange.kind) ||
          !equal(decision.error, exchange.error ?? null))
        throw new Error('observed decision differs from the recorded model exchange');
      if (exchange.kind === 'generation' && !exchange.error &&
          (!decision.assistant || !Array.isArray(exchange.reply?.raw_calls) ||
           !equal(decision.assistant.raw_calls, exchange.reply.raw_calls) ||
           decision.assistant.content !== (exchange.reply.text ?? '') ||
           decision.assistant.reasoning !== (exchange.reply.reasoning ?? '') ||
           !equal(decision.assistant.calls, exchange.reply.raw_response.vibestudio.response.content
             .filter(block => block.type === 'toolCall')
             .map(call => ({id: call.id, source_tool: call.name, arguments: call.arguments})))))
        throw new Error('observed decision target differs from the model reply');
    }
    captureScopes.delete(call.call_id);
    callFiles.delete(call.artifact.path);
    decisions.delete(call.call_id);
  }
  if (captureScopes.size || callFiles.size || decisions.size || roots.size)
    throw new Error('observed trajectory does not cover the full verified call tree');
  if (returnSelectors.some((item,index)=>item.required!==false&&!matchedReturnSelectors.has(index)))throw new Error('incorrect return binding does not match its actual original invocation');
  if ((selected && [...selected.keys()].some(id => !capturedRecords.has(id))) ||
      row.outcome.excluded_return_decisions !== undefined && !equal(row.outcome.excluded_return_decisions, returnExclusions))
    throw new Error('observed decision admission differs from independent training scope or return evidence');
  for(const binding of applicationReturns)verifyApplicationReturn(binding,capturedRecords,receipt.root_call_ids);
  verifyObservedCallTree(records, receipt.root_call_ids[0]);
  for (const decision of row.trajectory) {
    const chain = [];
    let record = capturedRecords.get(decision.invocation_id);
    while (record) { chain.unshift(record.call_id); record = capturedRecords.get(record.parent_call_id); }
    if (!equal(chain, decision.live_exchange.call_ancestry) || decision.live_exchange.root_call_id !== chain[0] ||
        decision.live_exchange.target_source !== 'actual-model-reply' || decision.live_exchange.student_sampling_claimed !== false)
      throw new Error('observed decision ancestry differs from the complete captured tree');
  }
}
