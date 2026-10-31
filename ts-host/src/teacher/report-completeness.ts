export type ReportCompletenessSpec = {
  schema: 'natlang.report-completeness/1' | 'natlang.report-completeness/2';
  mutating_methods: string[];
  effect_receipt_path?: string;
  report_effects_path: string;
  checkpoint_effects_path: string;
  report_diagnosis_path: string;
  diagnosis_function: string;
};

type Dict = Record<string, unknown>;

function atPath(value: unknown, path: string): unknown {
  let current = value;
  for (const part of path.split('.')) {
    if (!current || typeof current !== 'object' || !Object.hasOwn(current, part)) return undefined;
    current = (current as Dict)[part];
  }
  return current;
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') {
    const object = value as Dict;
    return `{${Object.keys(object).sort().map(key => `${JSON.stringify(key)}:${canonical(object[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function sameReceipt(actual: Dict, reported: unknown, nestedReceipt = false): boolean {
  if (!reported || typeof reported !== 'object' || Array.isArray(reported)) return false;
  const candidate = reported as Dict;
  const result = actual.result as Dict | undefined;
  if (!result || typeof result !== 'object') return false;
  const expected = { receiptId: result.receiptId, sequence: result.sequence, toolName: result.toolName,
    requestor: result.requestor, arguments: result.arguments, result: nestedReceipt ? result.result : result.value };
  const normalized = { receiptId: candidate.receiptId, sequence: candidate.sequence, toolName: candidate.toolName,
    requestor: candidate.requestor, arguments: candidate.arguments, result: candidate.result };
  return typeof candidate.actionId === 'string' && candidate.actionId.length > 0 &&
    canonical(normalized) === canonical(expected);
}

/** Reconcile a root's asserted repair report with authenticated execution receipts and child results. */
export function checkReportCompleteness(spec: ReportCompletenessSpec | undefined, value: unknown,
  executionGraph: Dict[], invocationLedger: Dict[]): { required: false } | {
    required: true; accepted: boolean; actual_effect_receipts: number; report_effect_receipts: number;
    checkpoint_effect_receipts: number; diagnosis_observed: boolean; diagnosis_matches: boolean; reasons: string[];
  } {
  if (!spec) return { required: false };
  if (!['natlang.report-completeness/1', 'natlang.report-completeness/2'].includes(spec.schema) ||
    (spec.schema === 'natlang.report-completeness/2' &&
      (typeof spec.effect_receipt_path !== 'string' || !spec.effect_receipt_path.length)) ||
    !Array.isArray(spec.mutating_methods) ||
    !spec.mutating_methods.every(method => typeof method === 'string' && method.length > 0) ||
    new Set(spec.mutating_methods).size !== spec.mutating_methods.length ||
    ![spec.report_effects_path, spec.checkpoint_effects_path, spec.report_diagnosis_path,
      spec.diagnosis_function, ...(spec.effect_receipt_path ? [spec.effect_receipt_path] : [])]
      .every(path => typeof path === 'string' && path.length > 0))
    throw new TypeError('invalid report completeness contract');
  const methods = new Set(spec.mutating_methods);
  const actual = executionGraph.filter(event => event.kind === 'effect' && event.phase === 'completed' &&
    typeof event.method === 'string' && methods.has(event.method)).flatMap(event => {
      if (spec.schema !== 'natlang.report-completeness/2') return [event];
      const wrapper = event.result as Dict | undefined;
      if (wrapper && wrapper.applied === false && wrapper.receipt === null) return [];
      const receipt = atPath(event.result, spec.effect_receipt_path!);
      return [{ ...event, result: receipt }];
    });
  const reportEffects = atPath(value, spec.report_effects_path);
  const checkpointEffects = atPath(value, spec.checkpoint_effects_path);
  const reportIsArray = Array.isArray(reportEffects), checkpointIsArray = Array.isArray(checkpointEffects);
  const report = reportIsArray ? reportEffects as unknown[] : [];
  const checkpoint = checkpointIsArray ? checkpointEffects as unknown[] : [];
  const exact = (entries: unknown[], shapeOk: boolean) => {
    if (!shapeOk || entries.length !== actual.length) return false;
    const actionIds = entries.map(entry => entry && typeof entry === 'object' ? (entry as Dict).actionId : undefined);
    const receiptIds = entries.map(entry => entry && typeof entry === 'object' ? (entry as Dict).receiptId : undefined);
    return actionIds.every(id => typeof id === 'string' && id.length > 0) && new Set(actionIds).size === actionIds.length &&
      receiptIds.every(id => typeof id === 'string' && id.length > 0) && new Set(receiptIds).size === receiptIds.length &&
      actual.every(event => entries.filter(entry => sameReceipt(event, entry,
        spec.schema === 'natlang.report-completeness/2')).length === 1);
  };
  const diagnosisEntries = invocationLedger.filter(entry => {
    const result = entry.host_result as Dict | undefined;
    return result?.capture_kind === 'invocation_output' && result.complete === true &&
      result.definition_source === spec.diagnosis_function;
  });
  const observedDiagnosis = diagnosisEntries.length > 0;
  const actualDiagnosis = diagnosisEntries.at(-1)?.host_result as Dict | undefined;
  const reportedDiagnosis = atPath(value, spec.report_diagnosis_path);
  const diagnosisMatches = observedDiagnosis && reportedDiagnosis !== undefined &&
    canonical(actualDiagnosis?.value) === canonical(reportedDiagnosis);
  const reasons: string[] = [];
  if (!exact(report, reportIsArray)) reasons.push('reported_effects_do_not_match_completed_mutations');
  if (!exact(checkpoint, checkpointIsArray)) reasons.push('checkpoint_effects_do_not_match_completed_mutations');
  if (!observedDiagnosis) reasons.push('diagnosis_invocation_missing');
  else if (!diagnosisMatches) reasons.push('reported_diagnosis_does_not_match_completed_diagnosis');
  return { required: true, accepted: reasons.length === 0, actual_effect_receipts: actual.length,
    report_effect_receipts: report.length, checkpoint_effect_receipts: checkpoint.length,
    diagnosis_observed: observedDiagnosis, diagnosis_matches: diagnosisMatches, reasons };
}
