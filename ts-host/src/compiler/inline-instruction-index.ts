import { hexDigest } from '../native/hash.js';
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
  };
  realized_instruction: string;
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
      if (!codeHash || codeHash !== stringAt(origin, 'checkedCodeSha256')) { hold(row, 'checked-source-hash-mismatch'); continue; }
      const segments = site.template_segments;
      const holes = site.interpolations;
      const captures = Array.isArray(site.captures) ? site.captures : undefined;
      if (!Array.isArray(segments) || segments.length !== 1 || typeof segments[0] !== 'string' ||
          !Array.isArray(holes) || holes.length !== 0) { hold(row, 'unsupported-interpolation'); continue; }
      if (!captures || captures.length !== 0 || site.explicit_captures === true) { hold(row, 'unsupported-capture-contract'); continue; }
      if (!Array.isArray(site.parameters) || site.returns === undefined) { hold(row, 'typed-plan-incomplete'); continue; }
      const realized = stringAt(site, 'realized_instruction');
      const segment = segments[0];
      const expectedRealized = segment.endsWith('\n') ? segment : `${segment}\n`;
      if (!realized || realized !== expectedRealized) { hold(row, 'realized-source-mismatch'); continue; }
      const spanValue = asDict(site.template_span);
      const start = spanValue?.start, end = spanValue?.end;
      if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || Number(start) < 0 || Number(end) <= Number(start)) {
        hold(row, 'invalid-template-span'); continue;
      }

      const siteId = `inline-site:${hexDigest(JSON.stringify([trajectoryId, parentInvocationId, toolCallId, definitionId, start, end])).slice(0, 24)}`;
      const key = `${invocationId}:${siteId}`;
      const existing = observations.get(key);
      const observation = { row, site, siteId, origin, invocationId, realized, code: '', span: { start: Number(start), end: Number(end) } };
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
      if (!validTemplateSource(code, span, String((site.template_segments as unknown[])[0]))) {
        hold(row, 'template-span-or-cooked-text-mismatch'); continue;
      }
      const opening = openingText(row);
      if (!opening || !opening.includes(realized)) { hold(row, 'realized-instruction-not-visible-in-child'); continue; }

      const writerKey = `${siteId}`;
      if (!processedSites.has(writerKey)) {
        const codeSpan = code.slice(span.start, span.end);
        const parameters = Array.isArray(site.parameters) ? site.parameters : [];
        writers.push({ kind: 'inline_instruction_writer', writer_id: writerKey, site_id: siteId,
          trajectory_id: trajectoryId, decision_id: String(parent.id ?? ''), decision_index: parentCandidates[0]!.index!,
          parent_invocation_id: parentInvocationId, tool_call_id: toolCallId, target_tool_call_id: targetId,
          definition_id: String(site.definition_id),
          code_sha256: String(origin.writtenCodeSha256), code, code_span: span, template_source: codeSpan,
          template_segments: [String((site.template_segments as unknown[])[0])],
          plan: { definition_id: String(site.definition_id), parameters, returns: site.returns,
            captures: [], explicit_captures: site.explicit_captures === true }, realized_instruction: realized });
        processedSites.add(writerKey);
      }
      const existingRead = reads.find(item => item.trajectory_id === trajectoryId && item.invocation_id === invocationId);
      if (existingRead) {
        if (existingRead.writer_id !== writerKey || existingRead.realized_instruction !== realized) hold(row, 'ambiguous-child-site');
        continue;
      }
      reads.push({ kind: 'inline_instruction_read', writer_id: writerKey, site_id: siteId, trajectory_id: trajectoryId,
        decision_id: String(row.id ?? ''), decision_index: decisionIndex!, invocation_id: invocationId,
        parent_invocation_id: parentInvocationId, opening_source: opening, realized_instruction: realized });
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
function openingText(row: Row): string | undefined {
  if (!Array.isArray(row.messages)) return undefined;
  const opening = row.messages.find(item => asDict(item)?.role === 'user');
  const content = asDict(opening)?.content;
  return typeof content === 'string' ? content : undefined;
}
function stable(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stable).join(',')}]`;
  if (isDict(value)) return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stable(value[key])}`).join(',')}}`;
  return JSON.stringify(value) ?? 'undefined';
}
