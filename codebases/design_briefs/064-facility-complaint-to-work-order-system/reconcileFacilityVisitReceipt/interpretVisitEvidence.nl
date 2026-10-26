---
args: { caseId: ID, receipt: FacilityVisitReceipt, plannedLocationId: ID, plannedIssueIds: ID[], requiredCheckIds: ID[], authorizedVisit: VisitAccessPermission | null, consentStatusAtAttempt: "valid" | "expired" | "unknown" }
returns: FacilityVisitAssessment
---
1. Bind the receipt to the exact case, official location, issue set, and consented access window. The authorization must name that location, include that window, contain an actual consentRef, and have consentStatusAtAttempt "valid". A mismatch is held evidence; it does not authorize entry.
2. A "no-access" or "cancelled" attempt is not completed work. Request a fresh consented window; do not retry the old window. If no current permission is available, return consent-needed.
3. An "unknown" attempt stays unresolved until the site contact confirms whether access or work occurred. Do not treat missing receipt data as success or failure.
4. For "completed" or "partial", require actual technicianEvidence, every required check ID, and the work-order IDs that were completed. Missing any required check leaves work unverified. A residual symptom requires mechanism review.
5. Return resolved only when the work and every required technical check are evidenced, no residual symptom remains, and either the occupant confirms the condition or the required objective checks are all present. For recurrence-prone issues, choose monitor-recurrence with a named follow-up rather than immediate closure.
