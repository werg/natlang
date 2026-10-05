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
  const code = error?.code ?? error?.cause?.code ?? '';
  if ([400,401,403,404,408,429,500,502,503,504].includes(error?.status))
    return 'infrastructure_failure';
  if (['ETIMEDOUT', 'ECONNRESET', 'ECONNREFUSED', 'EPIPE', 'ECONNABORTED',
    'UND_ERR_SOCKET', 'UND_ERR_CONNECT_TIMEOUT', 'UND_ERR_HEADERS_TIMEOUT', 'UND_ERR_BODY_TIMEOUT',
    'NATLANG_PROVIDER_REQUEST_TIMEOUT', 'NATLANG_PROVIDER_ACTION_CYCLE_TIMEOUT'].includes(code))
    return 'infrastructure_failure';
  if (['ERR_CONTEXT_LIMIT', 'ERR_REQUEST_BUDGET', 'ERR_TURN_LIMIT'].includes(code))
    return 'resource_failure';
  return 'incomplete_task';
}
