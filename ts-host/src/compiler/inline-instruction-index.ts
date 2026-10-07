import { hexDigest } from '../native/hash.js';
import { canonical, fingerprint } from '../adaptation/identity.js';
import { sourceWithLiteralCalls } from '../native/neuralese.js';
import { desugarNlCalls } from './nl-call.js';
import * as ts from 'typescript';

type Dict = Record<string, unknown>;

export type InlineInstructionWriter = {
  kind: 'inline_instruction_writer';
  writer_id: string;
  site_id: string;
  trajectory_id: string;
  decision_id: string;
  decision_index: number;
  parent_invocation_id: string;
  /** Actual ledger call ID used to prove causality. */
  tool_call_id: string;
  /** Synthetic materialized target call ID used to attach the writer in training output. */
  target_tool_call_id: string;
  definition_id: string;
  code_sha256: string;
  code: string;
  code_span: { start: number; end: number };
  template_source: string;
  template_segments: readonly [string];
  plan: {
    definition_id: string;
    parameters: unknown[];
    returns: unknown;
    captures: unknown[];
    explicit_captures: boolean;
    capture_binding_plan?: CaptureBindingPlan;
  };
  realized_instruction: string;
  /** Plain text recovered only from the paired raw model call that wrote the soft block. */
  body_source?: string;
  /** The exact sentinel carried by the original eval source, used for lossless reconstruction. */
  body_code_source?: string;
};

export type CaptureBindingPlan = {
  schema: 'natlang.inline-capture-binding-plan/1' | 'natlang.inline-capture-binding-plan/2';
  syntax: 'nl.with';
  body_block_id?: string;
  body_kind?: 'literal' | 'neuralese_block';
  creation?: Dict;
  body_source_sha256: string;
  captures: { name: string; type: 'string' | 'number' | 'boolean'; mode: 'snapshot'; value: string | number | boolean;
    source?: 'input' | 'local' | 'block'; host_snapshot?: Dict }[];
  parent_invocation_id: string;
  parent_scope_sha256: string;
  child_scope_sha256: string;
};

export type InlineInstructionRead = {
  kind: 'inline_instruction_read';
  writer_id: string;
  site_id: string;
  trajectory_id: string;
  decision_id: string;
  decision_index: number;
  invocation_id: string;
  parent_invocation_id: string;
  opening_source: string;
  realized_instruction: string;
  body_block_id?: string;
  body_source?: string;
  capture_binding_plan?: CaptureBindingPlan;
};

export type InlineInstructionHold = { trajectory_id: string; decision_id: string; reason: string };
export type InlineInstructionIndex = {
  writers: InlineInstructionWriter[];
  reads: InlineInstructionRead[];
  held: InlineInstructionHold[];
};

/**
 * Build action-scoped inline instruction writer/read edges from materialized teacher decisions.
 * This is indexing metadata only; it does not mutate records, convert tool arguments, or grant admission.
 */
export function buildInlineInstructionIndex(records: readonly unknown[]): InlineInstructionIndex {
  const rows = records.flatMap((item): Row[] => isDict(item) ? [item as Row] : []);
  const groups = new Map<string, Row[]>();
  for (const row of rows) {
    const trajectory = stringAt(row.source_ref, 'trajectory_id');
    if (!trajectory) continue;
    const group = groups.get(trajectory) ?? [];
    group.push(row);
    groups.set(trajectory, group);
  }

  const writers: InlineInstructionWriter[] = [];
  const reads: InlineInstructionRead[] = [];
  const held: InlineInstructionHold[] = [];
  const hold = (row: Row, reason: string) => held.push({ trajectory_id: stringAt(row.source_ref, 'trajectory_id') ?? '',
    decision_id: typeof row.id === 'string' ? row.id : '', reason });

  for (const [trajectoryId, runRows] of groups) {
    const observations = new Map<string, { row: Row; site: Dict; siteId: string; origin: Dict; invocationId: string; realized: string; code: string; span: { start: number; end: number } }>();
    const conflictedObservations = new Set<string>();
    for (const row of runRows) {
      const ref = asDict(row.source_ref), wrapped = asDict(ref?.inline_instruction_site);
      if (!wrapped) continue;
      const invocationId = stringAt(ref, 'invocation_id');
      const site = asDict(wrapped.site), origin = asDict(site?.origin);
      if (!invocationId || !site || !origin) { hold(row, 'missing-site-or-origin'); continue; }
      if (asDict(wrapped.validation)?.valid !== true) { hold(row, 'invalid-compiler-site'); continue; }
      if (site.schema !== 'natlang.inline_instruction_site/1') { hold(row, 'unsupported-site-schema'); continue; }
      const parentInvocationId = stringAt(origin, 'parentInvocationId');
      const toolCallId = stringAt(origin, 'toolCallId');
      const definitionId = stringAt(site, 'definition_id');
      if (!parentInvocationId || !toolCallId || !definitionId || parentInvocationId === invocationId) {
        hold(row, 'missing-or-invalid-origin'); continue;
      }
      if (!Number.isSafeInteger(origin.actionOrdinal) || Number(origin.actionOrdinal) < 0) {
        hold(row, 'missing-action-ordinal'); continue;
      }
      const codeHash = stringAt(origin, 'writtenCodeSha256');
      const bodyId = stringAt(site, 'soft_body_id');
      const mappedSoftBody = bodyId !== undefined && checkedSoftBodyMapping(site, origin);
      if (!codeHash || (codeHash !== stringAt(origin, 'checkedCodeSha256') && !mappedSoftBody)) {
        hold(row, 'checked-source-hash-mismatch'); continue;
      }
      const segments = site.template_segments;
      const holes = site.interpolations;
      const captures = Array.isArray(site.captures) ? site.captures : undefined;
      if (!Array.isArray(segments) || segments.length !== 1 || typeof segments[0] !== 'string' ||
          !Array.isArray(holes) || holes.length !== 0) { hold(row, 'unsupported-interpolation'); continue; }
      const softBodyId = bodyId;
      const softCaptureSite = softBodyId !== undefined && site.explicit_captures === true && !!captures?.length;
      const explicitCaptureSite = site.explicit_captures === true && !!captures?.length;
      if (!captures || (captures.length !== 0 && !explicitCaptureSite) || (site.explicit_captures === true && !explicitCaptureSite)) {
        hold(row, 'unsupported-capture-contract'); continue;
      }
      if (!Array.isArray(site.parameters) || site.returns === undefined) { hold(row, 'typed-plan-incomplete'); continue; }
      const realized = stringAt(site, 'realized_instruction');
      const segment = segments[0];
      const expectedRealized = segment.endsWith('\n') ? segment : `${segment}\n`;
      if (!softCaptureSite && (!realized || realized !== expectedRealized)) { hold(row, 'realized-source-mismatch'); continue; }
      const spanValue = asDict(site.template_span);
      const start = spanValue?.start, end = spanValue?.end;
      if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || Number(start) < 0 || Number(end) <= Number(start)) {
        hold(row, 'invalid-template-span'); continue;
      }

      const siteId = `inline-site:${hexDigest(JSON.stringify([trajectoryId, parentInvocationId, toolCallId, definitionId, start, end])).slice(0, 24)}`;
      const key = `${invocationId}:${siteId}`;
      const existing = observations.get(key);
      const observation = { row, site, siteId, origin, invocationId, realized: realized ?? '', code: '', span: { start: Number(start), end: Number(end) } };
      if (existing && stable(existing.site) !== stable(site)) {
        conflictedObservations.add(key);
        hold(row, 'duplicate-site-metadata-conflict');
        continue;
      }
      if (!existing) observations.set(key, observation);
    }

    const sitesByChild = new Map<string, Set<string>>();
    const signaturesBySite = new Map<string, Set<string>>();
    for (const item of observations.values()) {
      const ids = sitesByChild.get(item.invocationId) ?? new Set<string>();
      ids.add(item.siteId);
      sitesByChild.set(item.invocationId, ids);
      const signatures = signaturesBySite.get(item.siteId) ?? new Set<string>();
      signatures.add(stable(item.site));
      signaturesBySite.set(item.siteId, signatures);
    }
    const ambiguousChildren = new Set([...sitesByChild].filter(([, ids]) => ids.size > 1).map(([invocation]) => invocation));
    const ambiguousSites = new Set([...signaturesBySite].filter(([, signatures]) => signatures.size > 1).map(([siteId]) => siteId));
    for (const item of observations.values()) {
      if (ambiguousChildren.has(item.invocationId)) hold(item.row, 'ambiguous-child-site');
      else if (ambiguousSites.has(item.siteId)) hold(item.row, 'conflicting-writer-site-metadata');
    }

    const processedSites = new Set<string>();
    for (const [observationKey, observation] of observations) {
      if (conflictedObservations.has(observationKey) || ambiguousChildren.has(observation.invocationId) ||
          ambiguousSites.has(observation.siteId)) continue;
      const { row, site, siteId, origin, invocationId, realized, span } = observation;
      const parentInvocationId = String(origin.parentInvocationId);
      const toolCallId = String(origin.toolCallId);
      const decisionIndex = decisionOrder(row);
      const parentCandidates = runRows.flatMap(candidate => {
        if (stringAt(candidate.source_ref, 'invocation_id') !== parentInvocationId) return [];
        const index = decisionOrder(candidate);
        const calls = decisionCalls(candidate);
        const targetCalls = targetToolCalls(candidate);
        return calls.flatMap((call, callIndex) => stringAt(call.outcome, 'tool_call_id') === toolCallId &&
          call.source_tool === 'eval' ? [{ candidate, call, callIndex, targetCall: targetCalls[callIndex], index }] : []);
      });
      if (parentCandidates.length !== 1) { hold(row, parentCandidates.length ? 'duplicate-origin-action' : 'missing-parent-action'); continue; }
      const { candidate: parent, call, targetCall } = parentCandidates[0]!;
      if (asDict(parent.outcome)?.accepted !== true || asDict(parent.training_admission)?.approved !== true ||
          asDict(parent.decision)?.training_approved !== true ||
          !['ok', 'completed'].includes(String(asDict(call.outcome)?.status))) { hold(row, 'parent-action-not-successful-and-approved'); continue; }
      if (decisionIndex === undefined || parentCandidates[0]!.index === undefined || decisionIndex <= parentCandidates[0]!.index) {
        hold(row, 'source-order-invalid-or-unknown'); continue;
      }
      const args = asDict(call.arguments);
      const code = args?.code;
      if (typeof code !== 'string' || hexDigest(code) !== origin.writtenCodeSha256) { hold(row, 'parent-code-hash-mismatch'); continue; }
      const targetId = targetCall && typeof targetCall.id === 'string' ? targetCall.id : undefined;
      const targetFunction = asDict(targetCall?.function);
      const targetArgs = parseArguments(targetFunction?.arguments);
      if (!targetId || targetFunction?.name !== 'eval' || stable(targetArgs) !== stable(args)) {
        hold(row, 'target-action-identity-or-arguments-mismatch'); continue;
      }
      const bodyId = stringAt(site, 'soft_body_id');
      let bindingPlan: CaptureBindingPlan | undefined;
      let bodySource: string | undefined;
      let bodyCodeSource: string | undefined;
      let readSource: string;
      if (bodyId || site.explicit_captures === true) {
        const attested = attestSnapshotBody(site, code, span, parent, row, bodyId);
        if (!attested.valid) { hold(row, attested.reason!); continue; }
        bindingPlan = attested.plan;
        bodySource = attested.bodySource;
        bodyCodeSource = attested.bodyCodeSource;
        readSource = `${bodySource}\n`;
      } else {
        if (!validTemplateSource(code, span, String((site.template_segments as unknown[])[0]))) {
          hold(row, 'template-span-or-cooked-text-mismatch'); continue;
        }
        const opening = openingText(row);
        if (!opening || !opening.includes(realized)) { hold(row, 'realized-instruction-not-visible-in-child'); continue; }
        readSource = realized;
      }

      const writerKey = `${siteId}`;
      if (!processedSites.has(writerKey)) {
        const codeSpan = code.slice(span.start, span.end);
        const parameters = Array.isArray(site.parameters) ? site.parameters : [];
        writers.push({ kind: 'inline_instruction_writer', writer_id: writerKey, site_id: siteId,
          trajectory_id: trajectoryId, decision_id: String(parent.id ?? ''), decision_index: parentCandidates[0]!.index!,
          parent_invocation_id: parentInvocationId, tool_call_id: toolCallId, target_tool_call_id: targetId,
          definition_id: String(site.definition_id),
          code_sha256: String(origin.writtenCodeSha256), code, code_span: span, template_source: codeSpan,
          template_segments: bodySource === undefined ? [String((site.template_segments as unknown[])[0])] : [bodySource],
          plan: { definition_id: String(site.definition_id), parameters, returns: site.returns,
            captures: Array.isArray(site.captures) ? site.captures : [], explicit_captures: site.explicit_captures === true,
            ...(bindingPlan ? { capture_binding_plan: bindingPlan } : {}) },
          realized_instruction: bodySource === undefined ? realized : readSource,
          ...(bodySource !== undefined ? { body_source: bodySource, body_code_source: bodyCodeSource } : {}) });
        processedSites.add(writerKey);
      }
      const existingRead = reads.find(item => item.trajectory_id === trajectoryId && item.invocation_id === invocationId);
      if (existingRead) {
        if (existingRead.writer_id !== writerKey || existingRead.realized_instruction !== realized) hold(row, 'ambiguous-child-site');
        continue;
      }
      reads.push({ kind: 'inline_instruction_read', writer_id: writerKey, site_id: siteId, trajectory_id: trajectoryId,
        decision_id: String(row.id ?? ''), decision_index: decisionIndex!, invocation_id: invocationId,
        parent_invocation_id: parentInvocationId, opening_source: bodyId ? '' : openingText(row) ?? '',
        realized_instruction: bodyId ? readSource : realized,
        ...(bodyId ? { body_block_id: bodyId } : {}),
        ...(bindingPlan ? { body_source: bodySource, capture_binding_plan: bindingPlan } : {}) });
    }
  }

  return { writers, reads, held };
}

type Row = Dict & { id?: string; source_ref?: Dict; target?: Dict; messages?: unknown[]; decision?: Dict; outcome?: Dict; training_admission?: Dict };
function isDict(value: unknown): value is Dict { return !!value && typeof value === 'object' && !Array.isArray(value); }
function asDict(value: unknown): Dict | undefined { return isDict(value) ? value : undefined; }
function stringAt(value: unknown, key: string): string | undefined {
  const item = asDict(value)?.[key]; return typeof item === 'string' && item.length ? item : undefined;
}
function decisionOrder(row: Row): number | undefined {
  const value = asDict(row.decision)?.index;
  return Number.isSafeInteger(value) && Number(value) >= 0 ? Number(value) : undefined;
}
function decisionCalls(row: Row): Dict[] {
  const calls = asDict(asDict(row.decision)?.assistant)?.calls;
  return Array.isArray(calls) ? calls.filter(isDict) : [];
}
function targetToolCalls(row: Row): Dict[] {
  const calls = asDict(row.target)?.tool_calls;
  return Array.isArray(calls) ? calls.filter(isDict) : [];
}
function parseArguments(value: unknown): unknown {
  if (typeof value !== 'string') return value;
  try { return JSON.parse(value); } catch { return undefined; }
}
function checkedSoftBodyMapping(site: Dict, origin: Dict): boolean {
  const bodyId = stringAt(site, 'soft_body_id');
  const written = stringAt(origin, 'writtenCodeSha256');
  const checkedHash = stringAt(origin, 'checkedCodeSha256');
  const rawSpan = asDict(site.template_span), checkedSpan = asDict(site.checked_template_span);
  const sourceSpan = asDict(origin.sourceTemplateSpan);
  if (!bodyId || !written || !checkedHash || !rawSpan || !checkedSpan || !sourceSpan ||
      stable(rawSpan) !== stable(sourceSpan)) return false;
  const parentTemplate = Number(rawSpan.start), parentEnd = Number(rawSpan.end);
  const checkedStart = Number(checkedSpan.start), checkedEnd = Number(checkedSpan.end);
  if (![parentTemplate, parentEnd, checkedStart, checkedEnd].every(Number.isSafeInteger) ||
      parentEnd <= parentTemplate || checkedEnd <= checkedStart) return false;
  // The raw call text is checked against its exact paired authored marker by materialization. Here we confirm the
  // compiler's deterministic lowering and that the two attested template slices are the corresponding literals.
  return /^nz1_[a-z2-7]{20,}$/.test(bodyId) && !!written && !!checkedHash &&
    site.raw_body_source_sha256 === (typeof site.raw_body_source === 'string' ? hexDigest(site.raw_body_source) : undefined);
}
function validTemplateSource(code: string, span: { start: number; end: number }, cooked: string): boolean {
  if (span.end > code.length || span.start < 0) return false;
  const source = ts.createSourceFile('inline-eval.ts', code, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  let matches = 0, valid = false;
  const visit = (node: ts.Node) => {
    if (node.getStart(source) === span.start && node.getEnd() === span.end) {
      matches++;
      valid = ts.isNoSubstitutionTemplateLiteral(node) && node.text === cooked && code.slice(span.start, span.end) === node.getText(source);
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  return matches === 1 && valid;
}
function attestSnapshotBody(site: Dict, code: string, span: { start: number; end: number }, parent: Row, child: Row,
  bodyId?: string): { valid: true; plan: CaptureBindingPlan; bodySource: string; bodyCodeSource: string } |
    { valid: false; reason: string } {
  const captures = Array.isArray(site.captures) ? site.captures.map(asDict) : [];
  if (!captures.length || captures.some(capture => !capture)) return { valid: false, reason: 'unsupported-capture-contract' };
  const names = captures.map(capture => stringAt(capture, 'name'));
  if (names.some(name => !name) || new Set(names).size !== names.length) return { valid: false, reason: 'capture-descriptors-ambiguous' };
  const runtime = asDict(site.runtime_captures);
  if (!runtime) return { valid: false, reason: 'runtime-capture-attestation-missing' };
  const bodySource = bodyId ? stringAt(site, 'raw_body_source') : (site.template_segments as string[])[0];
  const bodyHash = bodyId ? stringAt(site, 'raw_body_source_sha256') : typeof bodySource === 'string' ? hexDigest(bodySource) : undefined;
  if (!bodySource || bodyHash !== hexDigest(bodySource) || /[\\`]|\$\{/.test(bodySource))
    return { valid: false, reason: 'soft-body-source-invalid-or-escaped' };
  const bodyCodeSource = bodyId ? `${bodyId}` : bodySource;
  if (!validTemplateSource(code, span, bodyCodeSource)) return { valid: false, reason: 'template-span-or-cooked-text-mismatch' };
  const checkedCode = desugarNlCalls(sourceWithLiteralCalls(code));
  const checkedSpan = asDict(site.checked_template_span);
  const checkedStart = Number(checkedSpan?.start), checkedEnd = Number(checkedSpan?.end);
  const expectedCheckedTemplate = bodyId ? `\`\${__neuralese.body(${JSON.stringify(bodyId)})}\`` : code.slice(span.start, span.end);
  if (hexDigest(checkedCode) !== stringAt(asDict(site.origin), 'checkedCodeSha256') ||
      !Number.isSafeInteger(checkedStart) || !Number.isSafeInteger(checkedEnd) ||
      checkedCode.slice(checkedStart, checkedEnd) !== expectedCheckedTemplate)
    return { valid: false, reason: 'checked-soft-body-span-mismatch' };
  const parsed = ts.createSourceFile('inline-capture-eval.ts', code, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  let template: ts.TaggedTemplateExpression | undefined;
  const visit = (node: ts.Node) => {
    if (ts.isTaggedTemplateExpression(node) && node.template.getStart(parsed) === span.start && node.template.getEnd() === span.end)
      template = node;
    ts.forEachChild(node, visit);
  };
  visit(parsed);
  const tag = template?.tag;
  const withCall = tag && ts.isCallExpression(tag) && ts.isPropertyAccessExpression(tag.expression) &&
    ts.isIdentifier(tag.expression.expression) && tag.expression.expression.text === 'nl' && tag.expression.name.text === 'with' ? tag : undefined;
  const object = withCall?.arguments.length === 1 && ts.isObjectLiteralExpression(withCall.arguments[0]!) ? withCall.arguments[0] : undefined;
  if (!object) return { valid: false, reason: 'explicit-with-syntax-mismatch' };
  const properties = object.properties;
  const propertyNames = properties.map(property => ts.isShorthandPropertyAssignment(property) ? property.name.text : undefined);
  if (propertyNames.some(name => !name) || stable(propertyNames) !== stable(names))
    return { valid: false, reason: 'explicit-capture-bindings-mismatch' };
  const parentText = rowText(parent), childText = rowText(child);
  const parentScopeDigest = hexDigest(parentText), childScopeDigest = hexDigest(childText);
  const snapshotEnvelope = asDict(site.runtime_capture_snapshots);
  const snapshots = snapshotEnvelope && snapshotEnvelope.schema === 'natlang.runtime_capture_snapshots/1' &&
    Array.isArray(snapshotEnvelope.captures) ? snapshotEnvelope.captures.map(asDict) : undefined;
  if (snapshotEnvelope && (!snapshots || snapshots.length !== captures.length || snapshots.some(item => !item) ||
      new Set(snapshots.map(item => item!.name)).size !== captures.length))
    return { valid: false, reason: 'runtime-capture-snapshots-incomplete-or-ambiguous' };
  if (!bodyId && !snapshots) return { valid: false, reason: 'literal-capture-snapshot-attestation-missing' };
  const bindingRows: CaptureBindingPlan['captures'] = [];
  for (const capture of captures as Dict[]) {
    const name = stringAt(capture, 'name')!;
    const target = asDict(capture.type);
    const type = stringAt(target, 'natlang') ?? stringAt(target, 'text');
    if (capture.mode !== 'snapshot' || capture.mutable !== false || !['input', 'local', 'block'].includes(String(capture.source)) ||
        !['string', 'number', 'boolean'].includes(type ?? ''))
      return { valid: false, reason: 'capture-not-portable-input-snapshot' };
    const runtimeCapture = asDict(runtime[name]);
    if (!runtimeCapture || runtimeCapture.mode !== 'snapshot' || runtimeCapture.type !== type ||
        Object.keys(runtime).length !== captures.length)
      return { valid: false, reason: 'runtime-capture-plan-mismatch' };
    const childValue = visiblePrimitive(childScopeDeclarations(child), name, type!, true);
    if (!childValue.found) return { valid: false, reason: 'capture-scope-visibility-unproven' };
    const snapshot = snapshots?.find(item => item!.name === name);
    let value: unknown;
    if (snapshot) {
      const creation = asDict(snapshot.creation), origin = asDict(site.origin);
      const fields = ['parentInvocationId', 'toolCallId', 'actionOrdinal', 'writtenCodeSha256', 'checkedCodeSha256'];
      if (snapshot.type !== type || snapshot.source !== capture.source || snapshot.mode !== 'snapshot' ||
          typeof snapshot.value !== type || (type === 'number' && (!Number.isFinite(snapshot.value) || Object.is(snapshot.value, -0))) ||
          snapshot.value_canonical !== canonical({ type, value: snapshot.value }) ||
          snapshot.value_sha256 !== fingerprint({ type, value: snapshot.value }, 'natlang.inline-capture-snapshot/v1') ||
          !creation || !origin || fields.some(field => creation[field] !== origin[field]) ||
          creation.definitionId !== site.definition_id || stable(creation.templateSpan) !== stable(site.template_span) ||
          stable(creation.checkedTemplateSpan) !== stable(site.checked_template_span) ||
          stable(creation.sourceSpan) !== stable(site.source_span))
        return { valid: false, reason: 'runtime-capture-snapshot-origin-or-value-mismatch' };
      value = snapshot.value;
    } else if (capture.source !== 'input') return { valid: false, reason: 'capture-local-snapshot-attestation-missing' };
    if (capture.source === 'input') {
      const parentValue = visiblePrimitive(parentInputScope(parent), name, type!, false);
      if (!parentValue.found) return { valid: false, reason: 'capture-scope-visibility-unproven' };
      if (snapshot && !Object.is(value, parentValue.value)) return { valid: false, reason: 'capture-snapshot-value-mismatch' };
      value = parentValue.value;
    }
    if (!Object.is(value, childValue.value)) return { valid: false, reason: 'capture-snapshot-value-mismatch' };
    bindingRows.push({ name, type: type as 'string' | 'number' | 'boolean', mode: 'snapshot', value: value as string | number | boolean,
      ...(snapshot ? { source: capture.source as 'input' | 'local' | 'block', host_snapshot: snapshot } : {}) });
  }
  const plan: CaptureBindingPlan = { schema: snapshots || !bodyId ? 'natlang.inline-capture-binding-plan/2' : 'natlang.inline-capture-binding-plan/1',
    syntax: 'nl.with', ...(bodyId ? { body_block_id: bodyId } : {}),
    ...(snapshots || !bodyId ? { body_kind: bodyId ? 'neuralese_block' : 'literal', creation: {
      parentInvocationId: asDict(site.origin)!.parentInvocationId, toolCallId: asDict(site.origin)!.toolCallId,
      actionOrdinal: asDict(site.origin)!.actionOrdinal, writtenCodeSha256: asDict(site.origin)!.writtenCodeSha256,
      checkedCodeSha256: asDict(site.origin)!.checkedCodeSha256, definitionId: site.definition_id, sourceSpan: site.source_span,
      templateSpan: site.template_span, checkedTemplateSpan: site.checked_template_span } } : {}),
    body_source_sha256: bodyHash, captures: bindingRows, parent_invocation_id: stringAt(asDict(parent.source_ref), 'invocation_id') ?? '',
    parent_scope_sha256: parentScopeDigest, child_scope_sha256: childScopeDigest };
  if (!plan.parent_invocation_id) return { valid: false, reason: 'capture-parent-invocation-missing' };
  if (bodyId ? !bodyInChildInstructions(child, bodyId) : !openingText(child)?.includes(`${bodySource}\n`))
    return { valid: false, reason: 'instruction-body-not-visible-in-child' };
  return { valid: true, plan, bodySource, bodyCodeSource };
}

function rowText(row: Row): string {
  const values: string[] = [];
  for (const message of row.messages ?? []) {
    const item = asDict(message);
    if (!item) continue;
    const content = item.content;
    if (typeof content === 'string') values.push(content);
    else if (Array.isArray(content)) for (const part of content) {
      const record = asDict(part);
      if (record?.type === 'text' && typeof record.text === 'string') values.push(record.text);
      else if (record?.type === 'neuralese' && typeof record.id === 'string') values.push(`${record.id}`);
    }
    for (const call of Array.isArray(item.tool_calls) ? item.tool_calls : []) {
      const args = asDict(asDict(call)?.function)?.arguments;
      if (typeof args === 'string') values.push(args);
    }
  }
  return values.join('\n');
}

function visiblePrimitive(text: string, name: string, type: string, declaration: boolean): { found: boolean; value?: unknown } {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const literal = type === 'string' ? '"(?:\\\\.|[^"\\\\])*"' :
    type === 'boolean' ? '(?:true|false)' : '-?(?:0|[1-9]\\d*)(?:\\.\\d+)?(?:[eE][+-]?\\d+)?';
  const prefix = declaration ? `const\\s+${escaped}:\\s*${type}\\s*=` : `${escaped}:\\s*${type}\\s*=`;
  const regex = new RegExp(`(?:^|\\n)\\s*${prefix}\\s*(${literal})\\s*;?\\s*(?=\\n|$)`, 'g');
  const matches = [...text.matchAll(regex)];
  if (!matches.length) return { found: false };
  const values: unknown[] = [];
  for (const match of matches) {
    try { values.push(JSON.parse(match[1]!)); } catch { return { found: false }; }
  }
  if (values.some(value => typeof value !== type || (type === 'number' && !Number.isFinite(value)) || !Object.is(value, values[0])))
    return { found: false };
  return { found: true, value: values[0] };
}

function parentInputScope(row: Row): string {
  for (const message of row.messages ?? []) {
    const item = asDict(message);
    if (item?.role !== 'tool' || item.tool_call_id !== 'scope_0' || typeof item.content !== 'string') continue;
    return item.content.split(/\nDeclared\s/)[0] ?? '';
  }
  return '';
}

function childScopeDeclarations(row: Row): string {
  for (const message of row.messages ?? []) {
    const item = asDict(message);
    if (item?.role !== 'assistant') continue;
    for (const call of Array.isArray(item.tool_calls) ? item.tool_calls : []) {
      const toolCall = asDict(call), fn = asDict(toolCall?.function);
      if (toolCall?.id !== 'scope_0' || fn?.name !== 'eval' || typeof fn.arguments !== 'string') continue;
      const args = parseArguments(fn.arguments), code = asDict(args)?.code;
      if (typeof code === 'string') return code;
    }
  }
  return '';
}

function bodyInChildInstructions(row: Row, id: string): boolean {
  const opening = (row.messages ?? []).map(asDict).find(item => item?.role === 'user');
  if (!opening) return false;
  const content = opening.content;
  const parts = Array.isArray(content) ? content.map(part => {
    const value = asDict(part);
    return value?.type === 'neuralese' && typeof value.id === 'string' ? `${value.id}` :
      value?.type === 'text' && typeof value.text === 'string' ? value.text : '';
  }).join('') : typeof content === 'string' ? content : '';
  const marker = parts.indexOf('Instructions:\n');
  if (marker < 0) return false;
  const tail = parts.slice(marker + 'Instructions:\n'.length);
  const boundary = /\n\n(?:In eval\b|Eval also\b)/.exec(tail);
  const section = boundary ? tail.slice(0, boundary.index + 1) : '';
  const needle = `${id}`;
  return !!section && section.split(needle).length === 2;
}

function openingText(row: Row): string | undefined {
  if (!Array.isArray(row.messages)) return undefined;
  const opening = row.messages.find(item => asDict(item)?.role === 'user');
  const rawContent = asDict(opening)?.content;
  const content = typeof rawContent === 'string' ? rawContent : Array.isArray(rawContent) &&
    rawContent.every(part => asDict(part)?.type === 'text' && typeof asDict(part)?.text === 'string') ?
    rawContent.map(part => asDict(part)!.text).join('') : undefined;
  if (typeof content !== 'string') return undefined;
  const marker = content.indexOf('Instructions:\n');
  if (marker < 0) return undefined;
  const start = marker + 'Instructions:\n'.length;
  const tail = content.slice(start);
  const boundary = /\n\n(?:In eval\b|Eval also\b)/.exec(tail);
  // Keep the first separator newline: it is the final newline of the instruction string itself.
  return boundary ? tail.slice(0, boundary.index + 1) : undefined;
}
function stable(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stable).join(',')}]`;
  if (isDict(value)) return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stable(value[key])}`).join(',')}}`;
  return JSON.stringify(value) ?? 'undefined';
}
