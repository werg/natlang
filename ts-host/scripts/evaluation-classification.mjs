/** Classify only trusted collector outcomes and typed runner/provider errors. */
export function classifyReturnedOutcome(outcome = {}) {
  if (outcome.oracle?.needs_review || (Array.isArray(outcome.quality_pending) && outcome.quality_pending.length))
    return 'review_required';
  if (outcome.status === 'done') {
    if (outcome.accepted === true && outcome.checks?.answer === true) return 'complete_success';
    if (outcome.checks?.answer === false) return 'semantic_failure';
    return 'contract_failure';
  }
  return 'incomplete_task';
}

/**
 * Case deadlines are runner-owned resource limits. Provider/network failures remain
 * infrastructure failures only when a typed code identifies them; generic TimeoutError
 * and AbortError names are ambiguous and are not enough to establish an outage.
 */
export function classifyCaughtError(error, { caseDeadlineExceeded = false } = {}) {
  if (caseDeadlineExceeded) return 'resource_failure';
  const code = error && typeof error.code === 'string' ? error.code : '';
  if (['ETIMEDOUT', 'ECONNRESET', 'ECONNREFUSED', 'EPIPE', 'ECONNABORTED',
    'NATLANG_PROVIDER_REQUEST_TIMEOUT', 'NATLANG_PROVIDER_ACTION_CYCLE_TIMEOUT'].includes(code))
    return 'infrastructure_failure';
  if (['ERR_CONTEXT_LIMIT', 'ERR_REQUEST_BUDGET', 'ERR_TURN_LIMIT'].includes(code))
    return 'resource_failure';
  return 'incomplete_task';
}
