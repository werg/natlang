const p = (name, evidence_path, allowed_fields, constraint, source_scope) =>
  ({ name, evidence_path, allowed_fields, constraint, source_scope });

export const worlds = [
  {
    slug: 'service_amendment_payment', group: 'v15:service_amendment_payment:world',
    domain: 'municipal service agreement payment reconciliation',
    instruction: 'Reconcile the payable amount and release decision for one service agreement across its signed agreement, later amendment, accepted-work ledger, and finance authorization.',
    fields: {
      agreementId: 'complete service agreement identifier',
      payableAmount: 'USD amount with a dollar sign, comma grouping, and exactly two decimal places',
      calculation: 'brief arithmetic showing accepted units multiplied by the controlling unit rate',
      decision: 'exactly lowercase release or hold',
    },
    value_policy: 'Preserve identifiers and dollar formatting exactly. Recompute amounts from the current controlling rate and accepted quantity; do not retain a superseded calculation.',
    decision_rule: 'Release only if the latest signed rate amendment is effective for the accepted work and finance authorization plus appropriation are both recorded; otherwise hold.',
    initial: { agreementId: 'UNKNOWN', payableAmount: '$0.00', calculation: 'not calculated', decision: 'hold' },
    evidence: {
      'pass-01-agreement.md': 'Service agreement SV-408 was signed 1 April 2028. It covers 12 completed service visits at the original rate of $120.00 per visit. The agreement is for the municipal north depot.',
      'pass-02-amendment.md': 'Signed amendment AM-408, effective 1 April 2028, replaces the original rate with $95.00 per visit for all SV-408 work. It does not change the 12-visit service scope.',
      'pass-03-accepted-work.md': 'The depot supervisor accepted 11 of the 12 SV-408 visits. Visit 12 was cancelled and is not billable. The 11 accepted visits are covered by AM-408.',
      'pass-04-finance.md': 'Finance authorization FA-408 was signed 19 April 2028 for payment of accepted SV-408 work. The $1,500.00 appropriation is available and has not been exhausted.',
    },
    passes: [
      p('service agreement', 'pass-01-agreement.md', ['agreementId', 'payableAmount', 'calculation'], 'Identify the agreement and calculate the provisional amount using the agreement rate and listed visit count.', 'the complete SV-408 signed agreement'),
      p('rate amendment', 'pass-02-amendment.md', ['payableAmount', 'calculation'], 'Apply any signed effective amendment to the agreement rate. Revise the provisional amount if the controlling rate changed.', 'the complete signed AM-408 amendment'),
      p('accepted-work ledger', 'pass-03-accepted-work.md', ['payableAmount', 'calculation'], 'Recalculate using only accepted billable visits and the controlling rate carried forward.', 'the accepted-work entries for SV-408'),
      p('finance authorization', 'pass-04-finance.md', ['decision'], 'Use release only if the signed payment authorization and available appropriation satisfy the task rule.', 'the finance authorization and appropriation entry for SV-408'),
    ],
    passStates: [
      { agreementId: 'SV-408', payableAmount: '$1,440.00', calculation: '12 visits × $120.00 = $1,440.00', decision: 'hold' },
      { agreementId: 'SV-408', payableAmount: '$1,140.00', calculation: '12 visits × $95.00 = $1,140.00', decision: 'hold' },
      { agreementId: 'SV-408', payableAmount: '$1,045.00', calculation: '11 accepted visits × $95.00 = $1,045.00', decision: 'hold' },
      { agreementId: 'SV-408', payableAmount: '$1,045.00', calculation: '11 accepted visits × $95.00 = $1,045.00', decision: 'release' },
    ],
    justified_revision: { pass: 2, field: 'payableAmount', reason: 'The signed amendment supersedes the original $120.00 rate with $95.00 for all covered work, changing the provisional amount.' },
    derivation: '12×120=1440; the signed amendment changes the rate, so 12×95=1140; the work ledger excludes one cancelled visit, so 11×95=1045. Finance approval exists and the $1,500 appropriation covers $1,045; release.',
  },
  {
    slug: 'community_grant_match_cap', group: 'v15:community_grant_match_cap:world',
    domain: 'community grant eligible-cost and matching-cap calculation',
    instruction: 'Calculate the grant amount for one application from its budget, the program allowability rule, the match record, the award ceiling, and the signed grant authorization.',
    fields: {
      grantId: 'complete grant application identifier',
      candidateAmount: 'USD amount currently under review, with a dollar sign, comma grouping, and exactly two decimal places',
      calculation: 'brief expression showing allowable costs, match cap, and award ceiling applied',
      decision: 'exactly lowercase approve or hold',
    },
    value_policy: 'Exclude disallowed budget items before applying the match ratio and award ceiling. Carry forward the actual computed amount and use USD with two decimals.',
    decision_rule: 'Approve only when the calculated grant is within both the carried match-based cap and carried award ceiling and the grant officer signed the award authorization.',
    initial: { grantId: 'UNKNOWN', candidateAmount: '$0.00', calculation: 'not calculated', decision: 'hold' },
    evidence: {
      'pass-01-application.md': 'Application CG-52 requests $52,000.00 for the Cedar Youth Workshop. Its itemized budget is $22,000.00 for instructor wages, $15,000.00 for venue and materials, $11,000.00 for accessible transportation, and $4,000.00 for fundraising events.',
      'pass-02-allowability.md': 'Program rule PR-52 allows instructor wages, venue and materials, and access transportation. Fundraising-event costs are disallowed. The CG-52 budget total after removing disallowed cost is the allowable-cost base.',
      'pass-03-match-and-ceiling.md': 'CG-52 has $16,000.00 in verified local cash match. The program may grant no more than $1.50 for each $1.00 of local cash match and has an award ceiling of $26,000.00. The grant is the lesser of allowable costs, the match-based cap, and the award ceiling.',
      'pass-04-award.md': 'Grant officer L. Pereira signed award authorization GA-52 on 8 May 2028. The authorization amount is the amount calculated under PR-52 and remains available in the grant account.',
    },
    passes: [
      p('application budget', 'pass-01-application.md', ['grantId', 'candidateAmount', 'calculation'], 'Identify CG-52, calculate the unfiltered requested budget total, and extract the fundraising-event line amount without classifying its allowability yet.', 'the four itemized CG-52 application budget lines'),
      p('allowability rule', 'pass-02-allowability.md', ['candidateAmount', 'calculation'], 'Remove disallowed line items and revise the amount to the allowable-cost base.', 'the full program rule and its application to CG-52'),
      p('match and award cap', 'pass-03-match-and-ceiling.md', ['candidateAmount', 'calculation'], 'Compute the match cap, then take the minimum of allowable costs, match cap, and award ceiling.', 'the verified local match, ratio, and cap for CG-52'),
      p('signed award', 'pass-04-award.md', ['decision'], 'Approve only if the authorized amount matches the computed amount and the grant officer signed.', 'the complete signed GA-52 authorization'),
    ],
    passStates: [
      { grantId: 'CG-52', candidateAmount: '$52,000.00', calculation: '$22,000 + $15,000 + $11,000 + $4,000 = $52,000 requested', decision: 'hold' },
      { grantId: 'CG-52', candidateAmount: '$48,000.00', calculation: '$22,000 + $15,000 + $11,000 = $48,000 allowable; $4,000 fundraising excluded', decision: 'hold' },
      { grantId: 'CG-52', candidateAmount: '$24,000.00', calculation: 'min($48,000, $16,000 × 1.50, $26,000) = $24,000.00', decision: 'hold' },
      { grantId: 'CG-52', candidateAmount: '$24,000.00', calculation: 'min($48,000, $16,000 × 1.50, $26,000) = $24,000.00', decision: 'approve' },
    ],
    justified_revision: { pass: 2, field: 'candidateAmount', reason: 'The program allowability rule excludes $4,000.00 of fundraising costs from the application total.' },
    derivation: 'Allowable costs are 22,000+15,000+11,000=48,000. Match cap is 16,000×1.5=24,000; award ceiling is 26,000; minimum is 24,000. Signed GA-52 authorizes the computed amount, so approve.',
  },
  {
    slug: 'archive_channel_consent', group: 'v15:archive_channel_consent:world',
    domain: 'community archive recording consent and publication channel',
    instruction: 'Determine whether one recording may be published on the requested channel by reconciling the signed release, later addendum, publication request, and controlling rights register.',
    fields: {
      recordingId: 'complete archive recording identifier',
      permittedChannels: 'exact lowercase channel list joined with comma-space',
      finding: 'brief statement comparing requested channel with current consent scope',
      decision: 'exactly lowercase publish or withhold',
    },
    value_policy: 'A later signed addendum controls over the earlier release. Preserve the exact channel names and do not infer permission for a channel that is not listed.',
    decision_rule: 'Publish only if the current signed consent explicitly permits the requested web channel and the rights register shows no later restriction.',
    initial: { recordingId: 'UNKNOWN', permittedChannels: 'unknown', finding: 'not evaluated', decision: 'withhold' },
    evidence: {
      'pass-01-release.md': 'Signed release AR-218 identifies recording ARC-218 and permits public web and reading-room access. The release was signed by the speaker on 2 February 2028.',
      'pass-02-addendum.md': 'Speaker-signed addendum AA-218, dated 12 February 2028, replaces the channel paragraph of AR-218. It permits reading-room access only and expressly withdraws public web permission for ARC-218.',
      'pass-03-request.md': 'The archive requested public website publication of recording ARC-218 on 20 February 2028. Reading-room playback is not the requested action.',
      'pass-04-rights-register.md': 'Rights register RR-218 lists AA-218 as the latest controlling signed consent for ARC-218. No later web-consent instrument is recorded as of 20 February 2028.',
    },
    passes: [
      p('original release', 'pass-01-release.md', ['recordingId', 'permittedChannels', 'finding'], 'Identify the recording and record the channels in its signed release as the initial consent state.', 'the complete signed AR-218 release'),
      p('later addendum', 'pass-02-addendum.md', ['permittedChannels', 'finding'], 'Apply the later speaker-signed replacement clause and revise the permitted channels; identify what changed.', 'the complete speaker-signed AA-218 replacement clause'),
      p('requested publication', 'pass-03-request.md', ['finding'], 'Compare the requested action with the carried current consent channels; do not broaden consent.', 'the ARC-218 publication request'),
      p('rights register', 'pass-04-rights-register.md', ['decision'], 'Use publish only if the current controlling consent includes public web access and no later restriction exists.', 'the rights history for ARC-218 through 20 February 2028'),
    ],
    passStates: [
      { recordingId: 'ARC-218', permittedChannels: 'public web, reading room', finding: 'initial release permits public web', decision: 'withhold' },
      { recordingId: 'ARC-218', permittedChannels: 'reading room', finding: 'AA-218 withdraws public web permission', decision: 'withhold' },
      { recordingId: 'ARC-218', permittedChannels: 'reading room', finding: 'requested web publication is outside current consent', decision: 'withhold' },
      { recordingId: 'ARC-218', permittedChannels: 'reading room', finding: 'requested web publication is outside current consent', decision: 'withhold' },
    ],
    justified_revision: { pass: 2, field: 'permittedChannels', reason: 'The later speaker-signed addendum replaces the initial broad release with reading-room-only permission.' },
    derivation: 'The original release permits web and reading room. AA-218 is later and expressly supersedes it, leaving reading room only. The request is for web, outside that scope; RR-218 confirms no later web permission. Withhold.',
  },
  {
    slug: 'accessible_route_substitution', group: 'v15:accessible_route_substitution:world',
    domain: 'accessible ferry terminal route substitution',
    instruction: 'Resolve the route for one assistance booking from the booking, lift inspection, accessible-route standard, and operations clearance.',
    fields: {
      bookingId: 'complete passenger booking identifier',
      route: 'exact approved route name',
      finding: 'brief numeric comparison of route slope and clear width with both limits',
      decision: 'exactly lowercase confirm or hold',
    },
    value_policy: 'Use the route that is available and satisfies both stated accessibility measurements. Preserve route names and decimal precision.',
    decision_rule: 'Confirm only if the selected route is operational, its slope is at most 8.0%, its clear width is at least 1.2 m, and operations signed clearance before departure.',
    initial: { bookingId: 'UNKNOWN', route: 'unassigned', finding: 'not evaluated', decision: 'hold' },
    evidence: {
      'pass-01-booking.md': 'Assistance booking PAX-8841 is for sailing ISLE-28-204, departing 24 September 2028. The booking assigns the passenger to Lift A at North Pier.',
      'pass-02-inspection.md': 'Inspection report IR-204 for sailing ISLE-28-204 and booking PAX-8841 says Lift A is out of service for the 24 September sailing. The approved alternate Route Ramp B is operational, has a 7.5% slope, and has 1.4 m clear width.',
      'pass-03-access-standard.md': 'Terminal standard AS-4 requires an operational step-free route with slope no greater than 8.0% and clear width no less than 1.2 m. Both measurements apply to the same selected route.',
      'pass-04-operations-clearance.md': 'Terminal operations supervisor signed clearance OC-204 for operational Route Ramp B on 23 September 2028, before ISLE-28-204 departure. The clearance is for booking PAX-8841.',
    },
    passes: [
      p('assistance booking', 'pass-01-booking.md', ['bookingId', 'route'], 'Identify the booking and its initially assigned route.', 'the complete booking record for PAX-8841'),
      p('lift inspection', 'pass-02-inspection.md', ['route', 'finding'], 'If the assigned lift is unavailable, revise the route to the named approved operational alternate and carry its measurements.', 'the inspection and approved alternate for the same sailing'),
      p('accessibility standard', 'pass-03-access-standard.md', ['finding'], 'Evaluate both numeric limits against the selected route in the carried draft.', 'the two route measurements and their inclusive limits'),
      p('operations clearance', 'pass-04-operations-clearance.md', ['decision'], 'Confirm only if the selected route and booking match a signed pre-departure operations clearance.', 'the complete OC-204 clearance for PAX-8841'),
    ],
    passStates: [
      { bookingId: 'PAX-8841', route: 'Lift A', finding: 'not evaluated', decision: 'hold' },
      { bookingId: 'PAX-8841', route: 'Ramp B', finding: 'Ramp B operational; slope 7.5%; width 1.4 m', decision: 'hold' },
      { bookingId: 'PAX-8841', route: 'Ramp B', finding: '7.5% ≤ 8.0%; 1.4 m ≥ 1.2 m', decision: 'hold' },
      { bookingId: 'PAX-8841', route: 'Ramp B', finding: '7.5% ≤ 8.0%; 1.4 m ≥ 1.2 m', decision: 'confirm' },
    ],
    justified_revision: { pass: 2, field: 'route', reason: 'Inspection shows the assigned Lift A is out of service and names Ramp B as the operational approved alternate.' },
    derivation: 'Lift A is unavailable, so choose Ramp B. Ramp B is operational; 7.5 is no greater than 8.0 and 1.4 is no less than 1.2. OC-204 is signed before departure for this booking; confirm.',
  },
  {
    slug: 'procurement_eligible_bid', group: 'v15:procurement_eligible_bid:world',
    domain: 'municipal procurement bid eligibility and award',
    instruction: 'Identify the winning bid under the mandatory qualification, conflict-screening, and signature rules for one request for quotation.',
    fields: {
      requestId: 'complete request for quotation identifier',
      vendor: 'exact vendor name of the lowest eligible bidder',
      finding: 'brief statement of qualification and conflict-screen result',
      decision: 'exactly lowercase award or hold',
    },
    value_policy: 'A low price does not cure a missing mandatory certificate. A declared conflict is eligible only after documented recusal of the conflicted reviewer.',
    decision_rule: 'Award the lowest-priced bid that has the required current ISO certificate, a completed conflict screen with recusal where needed, and a purchasing-chair signature before the deadline.',
    initial: { requestId: 'UNKNOWN', vendor: 'unselected', finding: 'not evaluated', decision: 'hold' },
    evidence: {
      'pass-01-bids.md': 'Request RFQ-44 asks for 500 standard supply kits. Vendor Alder Works bid $33,000.00; vendor Beacon Supply bid $34,000.00. These are the two responsive bids received by the deadline.',
      'pass-02-qualification.md': 'For RFQ-44, the two responsive bids are Alder Works at $33,000.00 and Beacon Supply at $34,000.00. RFQ-44 requires a current ISO 9001 certificate. Alder Works certificate expired 31 December 2027. Beacon Supply certificate ISO-9001-BS-9 is current through 31 December 2028. Beacon disclosed that consultant J. Rhee advised on the solicitation.',
      'pass-03-conflict-screen.md': 'Conflict officer CO-44 recorded that J. Rhee did not evaluate bids and was recused from all RFQ-44 scoring on 3 March 2028. CO-44 closed the conflict screen with no remaining participation by J. Rhee.',
      'pass-04-award-signature.md': 'Purchasing chair M. Holt signed award approval PA-44 for Beacon Supply on 6 March 2028. The bid deadline was 7 March 2028. Approval identifies RFQ-44 and $34,000.00.',
    },
    passes: [
      p('responsive bids', 'pass-01-bids.md', ['requestId', 'vendor', 'finding'], 'Identify RFQ-44 and provisionally select the lowest-priced responsive bid before checking mandatory qualifications.', 'the two responsive RFQ-44 bids and prices'),
      p('mandatory qualification', 'pass-02-qualification.md', ['vendor', 'finding'], 'Reject a bidder missing the mandatory current certificate and revise to the lowest-priced remaining qualified bidder.', 'the current certificate status of each RFQ-44 bidder'),
      p('conflict screen', 'pass-03-conflict-screen.md', ['finding'], 'Determine whether the selected bidder’s disclosed adviser was fully recused from bid evaluation and scoring.', 'the conflict officer record for RFQ-44'),
      p('award signature', 'pass-04-award-signature.md', ['decision'], 'Award only if the signed approval matches the selected eligible vendor and precedes the bid deadline.', 'the complete signed PA-44 and RFQ-44 bid deadline'),
    ],
    passStates: [
      { requestId: 'RFQ-44', vendor: 'Alder Works', finding: 'lowest responsive bid: $33,000.00; qualifications pending', decision: 'hold' },
      { requestId: 'RFQ-44', vendor: 'Beacon Supply', finding: 'Alder certificate expired; Beacon current ISO 9001 and lowest qualified at $34,000.00', decision: 'hold' },
      { requestId: 'RFQ-44', vendor: 'Beacon Supply', finding: 'Beacon current ISO 9001; disclosed adviser fully recused from scoring', decision: 'hold' },
      { requestId: 'RFQ-44', vendor: 'Beacon Supply', finding: 'Beacon current ISO 9001; disclosed adviser fully recused from scoring', decision: 'award' },
    ],
    justified_revision: { pass: 2, field: 'vendor', reason: 'Alder Works is the lowest bid but its required ISO certificate expired; Beacon Supply is the next-lowest bidder with a current certificate.' },
    derivation: 'Alder Works is lowest at 33,000 but unqualified due expired certification. Beacon is 34,000, has current certification, and its adviser was recused before scoring. Chair signed before deadline; award to Beacon.',
  },
  {
    slug: 'public_notice_correction_deadline', group: 'v15:public_notice_correction_deadline:world',
    domain: 'public hearing notice correction and minimum notice period',
    instruction: 'Determine whether a corrected public hearing notice met the required full-calendar-day notice period, accounting for the correction date and any lawful waiver.',
    fields: {
      docketId: 'complete hearing docket identifier',
      controllingPostDate: 'date in YYYY-MM-DD format of the latest corrected notice',
      noticeFinding: 'brief exact day-count comparison with the required minimum and waiver condition',
      decision: 'exactly lowercase accept or reject',
    },
    value_policy: 'A corrected notice resets the notice period. Count full calendar days from the correction date to the hearing date; do not use the superseded first posting date.',
    decision_rule: 'Accept only if at least 7 full intervening calendar days lie after the latest correction and before the hearing, or the clerk records a lawful waiver.',
    initial: { docketId: 'UNKNOWN', controllingPostDate: '2000-01-01', noticeFinding: 'not evaluated', decision: 'reject' },
    evidence: {
      'pass-01-hearing-docket.md': 'Docket PLAN-51 schedules a public hearing for 22 May 2028. The notice period is measured in full calendar days before the hearing.',
      'pass-02-publication-log.md': 'The original PLAN-51 notice was posted 10 May 2028. A corrected notice replaced it on 15 May 2028. The correction is the operative notice. The hearing is scheduled for 22 May 2028.',
      'pass-03-notice-rule.md': 'Notice rule NR-7 requires at least 7 full intervening calendar days after the latest corrected notice and before the hearing, excluding both event dates. A correction resets the count. Fewer than 7 days is invalid unless a lawful waiver is recorded.',
      'pass-04-clerk-register.md': 'Clerk register CR-51 records no lawful waiver for PLAN-51. Clerk E. Doss signed the completeness determination on 19 May 2028.',
    },
    passes: [
      p('hearing docket', 'pass-01-hearing-docket.md', ['docketId'], 'Identify the docket and hearing date that controls the later notice calculation.', 'the complete PLAN-51 docket entry'),
      p('publication history', 'pass-02-publication-log.md', ['controllingPostDate', 'noticeFinding'], 'Use the latest corrected publication date, replacing any provisional count based on the original posting.', 'the ordered original and corrected posting events for PLAN-51'),
      p('notice rule', 'pass-03-notice-rule.md', ['noticeFinding'], 'Count only full intervening dates, excluding correction and hearing dates, then compare with the seven-day minimum.', 'the correction-reset rule and full-intervening-day convention'),
      p('clerk register', 'pass-04-clerk-register.md', ['decision', 'noticeFinding'], 'Use the clerk register to resolve whether a lawful waiver exists; accept only if the notice count qualifies or a waiver is recorded.', 'the waiver register for PLAN-51'),
    ],
    passStates: [
      { docketId: 'PLAN-51', controllingPostDate: '2000-01-01', noticeFinding: 'not evaluated', decision: 'reject' },
      { docketId: 'PLAN-51', controllingPostDate: '2028-05-15', noticeFinding: 'corrected notice controls; 6 full intervening days before hearing', decision: 'reject' },
      { docketId: 'PLAN-51', controllingPostDate: '2028-05-15', noticeFinding: '6 full intervening days < 7 required; accept only with lawful waiver', decision: 'reject' },
      { docketId: 'PLAN-51', controllingPostDate: '2028-05-15', noticeFinding: '6 full intervening days < 7 required; no lawful waiver recorded', decision: 'reject' },
    ],
    justified_revision: { pass: 2, field: 'controllingPostDate', reason: 'The publication log identifies 15 May as the operative correction, replacing the earlier 10 May posting date.' },
    derivation: 'Hearing May 22 and latest corrected posting May 15 leave six full intervening dates, May 16–21, after excluding both event dates. Six is less than seven and the clerk register has no waiver; reject.',
  },
  {
    slug: 'course_accommodation_version', group: 'v15:course_accommodation_version:world',
    domain: 'course accommodation plan and current course version',
    instruction: 'Determine whether one learner accommodation applies to the current course version and can be approved before the scheduled assessment.',
    fields: {
      planId: 'complete accommodation plan identifier',
      courseVersion: 'exact current course version identifier',
      coveredVersion: 'exact course version covered by the latest signed plan instrument',
      support: 'exact accommodation package separated by semicolon-space',
      leadTime: 'business-day count and threshold, formatted as a short comparison',
      decision: 'exactly lowercase approve or hold',
    },
    value_policy: 'Use the latest signed plan addendum for course-version scope. Apply the stated lead-time rule to business days, not calendar days.',
    decision_rule: 'Approve only if the latest signed plan and addendum cover the current version, include the listed support, at least the carried minimum business days remain after the applicable addendum, and the coordinator signed approval for that same plan/addendum/version.',
    initial: { planId: 'UNKNOWN', courseVersion: 'unknown', coveredVersion: 'unknown', support: 'not established', leadTime: 'not evaluated', decision: 'hold' },
    evidence: {
      'pass-01-course-record.md': 'Course BIO-42 assessment is scheduled for 18 October 2028. The current course version is C8, released 1 October 2028.',
      'pass-02-plan.md': 'Learner accommodation plan AC-73 is signed and covers BIO-42 version C7. It specifies accessible electronic text and time-and-a-half assessment duration.',
      'pass-03-addendum-and-calendar.md': 'Signed addendum AC-73A, issued 9 October 2028, extends AC-73 to current version C8 without changing its listed support. The assessment remains 18 October; six business days remain after the addendum date.',
      'pass-04-coordinator.md': 'Accessibility coordinator N. Bell signed approval CA-73 for AC-73A and BIO-42 version C8 on 10 October 2028, before the 18 October assessment.',
    },
    passes: [
      p('current course record', 'pass-01-course-record.md', ['courseVersion'], 'Record the current version that an eligible plan must cover.', 'the current BIO-42 release and assessment schedule'),
      p('signed plan', 'pass-02-plan.md', ['planId', 'coveredVersion', 'support'], 'Identify the signed plan, its covered version, and exact support package; do not assume version coverage.', 'the full AC-73 plan scope and supports'),
      p('version addendum and calendar', 'pass-03-addendum-and-calendar.md', ['coveredVersion', 'leadTime'], 'Apply the later signed addendum to covered-version scope and verify its business-day lead time meets five days.', 'the signed addendum, current version, and lead-time facts'),
      p('coordinator approval', 'pass-04-coordinator.md', ['decision'], 'Approve only when coordinator approval names the same plan/addendum and current course version.', 'the complete signed CA-73 approval'),
    ],
    passStates: [
      { planId: 'UNKNOWN', courseVersion: 'C8', coveredVersion: 'unknown', support: 'not established', leadTime: 'not evaluated', decision: 'hold' },
      { planId: 'AC-73', courseVersion: 'C8', coveredVersion: 'C7', support: 'accessible electronic text; time-and-a-half duration', leadTime: 'not evaluated', decision: 'hold' },
      { planId: 'AC-73', courseVersion: 'C8', coveredVersion: 'C8', support: 'accessible electronic text; time-and-a-half duration', leadTime: '6 business days; 6 ≥ 5 required', decision: 'hold' },
      { planId: 'AC-73', courseVersion: 'C8', coveredVersion: 'C8', support: 'accessible electronic text; time-and-a-half duration', leadTime: '6 business days; 6 ≥ 5 required', decision: 'approve' },
    ],
    justified_revision: { pass: 3, field: 'coveredVersion', reason: 'AC-73A is a later signed addendum extending the plan from C7 to current course version C8.' },
    derivation: 'Current course is C8. Original plan covers C7, but signed AC-73A extends it to C8. Six business days remain, meeting the five-day minimum; coordinator approves the same plan/addendum and version; approve.',
  },
  {
    slug: 'protected_species_field_permit', group: 'v15:protected_species_field_permit:world',
    domain: 'field research permit for a protected species and seasonal closure',
    instruction: 'Determine whether one field permit may proceed after reconciling survey identification, the protected-species schedule, the seasonal closure, and the permit-board exception record.',
    fields: {
      permitId: 'complete field permit identifier',
      species: 'exact current survey species name',
      finding: 'brief statement joining listing status, field date, and closure rule',
      decision: 'exactly lowercase issue or deny',
    },
    value_policy: 'Use the latest verified species identification. Apply any seasonal closure to the same species, site, and requested date.',
    decision_rule: 'Deny only when the verified species is listed, the seasonal closure applies to the same site and field date, and no signed board exception covers that same species, site, and date; issue otherwise.',
    initial: { permitId: 'UNKNOWN', species: 'unverified', finding: 'not evaluated', decision: 'deny' },
    evidence: {
      'pass-01-application.md': 'Field permit FP-209 requests sampling at Marsh Reserve Site M-4 on 11 July 2028. The applicant requests entry from 09:00 to 11:00.',
      'pass-02-survey.md': 'Initial field survey FS-209 provisionally identified the target as silver marsh violet. The survey states identification is pending laboratory confirmation.',
      'pass-03-verification-and-rule.md': 'Laboratory correction LC-209 for permit FP-209 verifies the requested target at Site M-4 as blue marsh orchid, a listed protected species. FP-209 requests 11 July 2028 sampling. Reserve rule MR-6 closes Site M-4 to blue marsh orchid field sampling from 1 May through 31 August 2028 inclusive.',
      'pass-04-board-register.md': 'Permit-board register PB-209 contains no signed exception for blue marsh orchid at Site M-4 on 11 July 2028. Board secretary certified the search on 9 July 2028.',
    },
    passes: [
      p('permit application', 'pass-01-application.md', ['permitId'], 'Identify the permit, site, and requested date to scope later species and closure facts.', 'the complete FP-209 request'),
      p('initial survey', 'pass-02-survey.md', ['species'], 'Record the provisional species as provisional and do not treat pending identification as verified.', 'the full provisional FS-209 identification'),
      p('verified identification and rule', 'pass-03-verification-and-rule.md', ['species', 'finding'], 'Replace the provisional species with the lab-verified identification and test the same-site date against the inclusive closure.', 'the verified species, listing status, and site-specific closure period'),
      p('permit-board register', 'pass-04-board-register.md', ['decision', 'finding'], 'Use the exception search for this same species, site, and date; deny when the closure applies and no signed exception exists.', 'the exception search for FP-209'),
    ],
    passStates: [
      { permitId: 'FP-209', species: 'unverified', finding: 'not evaluated', decision: 'deny' },
      { permitId: 'FP-209', species: 'silver marsh violet (provisional)', finding: 'not evaluated', decision: 'deny' },
      { permitId: 'FP-209', species: 'blue marsh orchid', finding: 'listed; 11 July is within 1 May–31 August closure at Site M-4', decision: 'deny' },
      { permitId: 'FP-209', species: 'blue marsh orchid', finding: 'listed; 11 July is within 1 May–31 August closure at Site M-4; no exception', decision: 'deny' },
    ],
    justified_revision: { pass: 3, field: 'species', reason: 'Laboratory correction LC-209 supersedes the provisional field identification with blue marsh orchid.' },
    derivation: 'The verified target is listed blue marsh orchid. July 11 falls inside the inclusive May 1–August 31 closure for Site M-4. PB-209 has no matching signed exception; deny.',
  },
  {
    slug: 'drought_household_allocation', group: 'v15:drought_household_allocation:world',
    domain: 'monthly household water allocation under a drought tier',
    instruction: 'Compute a household monthly water allocation from its normal baseline, drought reduction, certified hardship allowance, and signed allocation record.',
    fields: {
      accountId: 'complete water account identifier',
      allowedLiters: 'whole-number liters followed by one space and L',
      calculation: 'brief arithmetic showing the ordinary tier and any qualified hardship allowance',
      decision: 'exactly lowercase issue or hold',
    },
    value_policy: 'The ordinary amount is baseline × 70%. Add 250 L only when the current signed medical hardship certificate is present. Use whole liters.',
    decision_rule: 'Issue only if the household qualifies for the calculated allowance, the same amount is entered in the signed account authorization, and the district cap is not exceeded.',
    initial: { accountId: 'UNKNOWN', allowedLiters: '0 L', calculation: 'not calculated', decision: 'hold' },
    evidence: {
      'pass-01-account.md': 'Water account H-18 has a normal monthly baseline of 4,000 L and requests 3,600 L for November 2028. The household is within the district service area.',
      'pass-02-drought-tier.md': 'For H-18, the normal monthly baseline is 4,000 L. Drought tier D3 authorizes 70% of normal monthly baseline for ordinary residential use. The district cap is 3,100 L per account per month.',
      'pass-03-hardship-certificate.md': 'Current physician certificate HC-18, signed 28 October 2028, qualifies H-18 for the tier hardship allowance of 250 L for November 2028. The allowance is added to the ordinary tier amount.',
      'pass-04-authorization.md': 'District water officer signed WA-18 for 3,050 L to H-18 for November 2028. The authorization is within the 3,100 L district cap.',
    },
    passes: [
      p('water account', 'pass-01-account.md', ['accountId'], 'Identify H-18; do not treat requested quantity as an approved allowance.', 'the H-18 account baseline and November request'),
      p('drought tier', 'pass-02-drought-tier.md', ['allowedLiters', 'calculation'], 'Compute the ordinary allocation as 70% of the carried 4,000 L baseline and check the district cap.', 'the D3 reduction and cap'),
      p('hardship certificate', 'pass-03-hardship-certificate.md', ['allowedLiters', 'calculation'], 'Add 250 L only because the current certificate qualifies this account and month; revise the ordinary-only amount.', 'the signed current HC-18 qualification'),
      p('allocation authorization', 'pass-04-authorization.md', ['decision'], 'Issue only if the signed quantity matches the computed allowance and is within the cap.', 'the signed November WA-18 authorization'),
    ],
    passStates: [
      { accountId: 'H-18', allowedLiters: '0 L', calculation: 'not calculated', decision: 'hold' },
      { accountId: 'H-18', allowedLiters: '2,800 L', calculation: '4,000 L × 70% = 2,800 L; below 3,100 L cap', decision: 'hold' },
      { accountId: 'H-18', allowedLiters: '3,050 L', calculation: '4,000 L × 70% + 250 L = 3,050 L; below 3,100 L cap', decision: 'hold' },
      { accountId: 'H-18', allowedLiters: '3,050 L', calculation: '4,000 L × 70% + 250 L = 3,050 L; below 3,100 L cap', decision: 'issue' },
    ],
    justified_revision: { pass: 3, field: 'allowedLiters', reason: 'Current signed HC-18 qualifies H-18 for an additional 250 L after the ordinary 2,800 L D3 allocation.' },
    derivation: 'Ordinary allocation is 4,000×0.70=2,800. Current hardship certificate adds 250 for a total 3,050, below 3,100 cap. WA-18 authorizes the same amount; issue.',
  },
  {
    slug: 'vaccine_logger_correction', group: 'v15:vaccine_logger_correction:world',
    domain: 'clinic vaccine cold-chain excursion assessment',
    instruction: 'Assess whether one vaccine lot stays within its cumulative warm-exposure limit after applying the logger correction and pharmacy quarantine rule.',
    fields: {
      lotId: 'complete vaccine lot identifier',
      minutesAboveLimit: 'whole-number minutes above 8.0 degrees C',
      comparison: 'brief comparison of corrected exposure minutes with the allowed maximum',
      decision: 'exactly lowercase release or quarantine',
    },
    value_policy: 'Sum only interval minutes above 8.0 degrees C. Apply the documented logger correction once; do not average away exposure time.',
    decision_rule: 'Release only if corrected cumulative exposure is no more than 45 minutes and the pharmacist release review is signed; otherwise quarantine.',
    initial: { lotId: 'UNKNOWN', minutesAboveLimit: '0', comparison: 'not evaluated', decision: 'quarantine' },
    evidence: {
      'pass-01-lot-record.md': 'Clinic lot VC-77 contains 240 doses of vaccine and arrived on 5 November 2028. The lot identifier on the carton and receiving record is VC-77.',
      'pass-02-logger-report.md': 'Logger report LR-77 for vaccine lot VC-77 records two intervals above 8.0 degrees C: 15 minutes and 23 minutes. The preliminary total is the sum of those intervals.',
      'pass-03-correction-and-limit.md': 'Calibration correction CC-77 for lot VC-77 adds 12 minutes to LR-77 because the logger clock omitted an interval. Corrected cumulative time above 8.0 degrees C is the preliminary total plus 12 minutes. Clinic policy permits at most 45 cumulative minutes.',
      'pass-04-pharmacy-review.md': 'Pharmacist P. Nair signed review PR-77 for vaccine lot VC-77 on 6 November 2028. Policy requires quarantine whenever corrected exposure exceeds 45 minutes; PR-77 records no separate exception or temperature-stability study.',
    },
    passes: [
      p('lot receipt', 'pass-01-lot-record.md', ['lotId'], 'Identify the same received lot that is evaluated by later logger facts.', 'the receiving and carton identifiers for VC-77'),
      p('preliminary logger', 'pass-02-logger-report.md', ['minutesAboveLimit', 'comparison'], 'Sum the two recorded intervals as a preliminary exposure total.', 'the two above-limit intervals in LR-77'),
      p('logger correction and policy', 'pass-03-correction-and-limit.md', ['minutesAboveLimit', 'comparison'], 'Apply the missing 12-minute correction and compare the corrected total with the inclusive maximum.', 'the calibration correction and 45-minute maximum for VC-77'),
      p('pharmacy review', 'pass-04-pharmacy-review.md', ['decision'], 'Release only if corrected exposure is within limit and a signed review supports release; otherwise quarantine.', 'the signed PR-77 review and policy exception status'),
    ],
    passStates: [
      { lotId: 'VC-77', minutesAboveLimit: '0', comparison: 'not evaluated', decision: 'quarantine' },
      { lotId: 'VC-77', minutesAboveLimit: '38', comparison: '15 + 23 = 38 preliminary minutes', decision: 'quarantine' },
      { lotId: 'VC-77', minutesAboveLimit: '50', comparison: '38 + 12 = 50 minutes; 50 > 45 maximum', decision: 'quarantine' },
      { lotId: 'VC-77', minutesAboveLimit: '50', comparison: '38 + 12 = 50 minutes; 50 > 45 maximum', decision: 'quarantine' },
    ],
    justified_revision: { pass: 3, field: 'minutesAboveLimit', reason: 'Calibration correction CC-77 adds an omitted 12-minute interval to the preliminary 38-minute exposure.' },
    derivation: 'Preliminary exposure is 15+23=38 minutes. Corrected exposure is 38+12=50, exceeding the 45-minute maximum. The signed review contains no exception; quarantine.',
  },
  {
    slug: 'demand_response_meter_settlement', group: 'v15:demand_response_meter_settlement:world',
    domain: 'commercial demand-response reduction settlement',
    instruction: 'Calculate the event reduction for one demand-response account using corrected baseline meter reads and decide whether the threshold earns payment.',
    fields: {
      accountId: 'complete demand-response account identifier',
      reductionPercent: 'percentage with exactly two decimal places and a trailing percent sign',
      calculation: 'brief expression showing the three-read baseline and event usage',
      decision: 'exactly lowercase pay or hold',
    },
    value_policy: 'Baseline is the arithmetic mean of the three corrected reference reads. Reduction is (baseline minus event usage) divided by baseline, times 100, rounded half-up to two decimal places.',
    decision_rule: 'Pay only if corrected reduction is at least 15.00% and the settlement officer signed the account settlement.',
    initial: { accountId: 'UNKNOWN', reductionPercent: '0.00%', calculation: 'not calculated', decision: 'hold' },
    evidence: {
      'pass-01-enrollment.md': 'Demand-response account DR-88 enrolled for the 12 July 2028 event. Contract requires a reduction of at least 15.00% from the three-read reference baseline.',
      'pass-02-meter-reads.md': 'Initial reference reads for DR-88 are 100 kWh, 120 kWh, and 120 kWh. Event usage is 102 kWh. The initial baseline is the arithmetic mean of the three reference reads.',
      'pass-03-settlement-correction.md': 'Meter correction MC-88 replaces the first reference read of 100 kWh with 120 kWh. The corrected baseline is the mean of 120, 120, and 120 kWh. Event usage remains 102 kWh. Round percentage reduction half-up to two decimal places.',
      'pass-04-settlement-signature.md': 'Settlement officer R. Sun signed settlement DS-88 for DR-88 on 15 July 2028. The settlement rule pays accounts meeting or exceeding 15.00% corrected reduction.',
    },
    passes: [
      p('enrollment contract', 'pass-01-enrollment.md', ['accountId'], 'Identify the account and the contractual inclusive percentage threshold.', 'the DR-88 enrollment and threshold'),
      p('initial meter settlement', 'pass-02-meter-reads.md', ['reductionPercent', 'calculation'], 'Compute preliminary baseline and reduction from all three initial reads and event usage.', 'the three initial reference reads and event usage'),
      p('meter correction', 'pass-03-settlement-correction.md', ['reductionPercent', 'calculation'], 'Replace the corrected read, recalculate the baseline and percentage, and use half-up two-decimal rounding.', 'the corrected three-read baseline and event usage'),
      p('settlement signature', 'pass-04-settlement-signature.md', ['decision'], 'Pay only when corrected reduction reaches 15.00% and the settlement officer signed.', 'the signed DS-88 settlement'),
    ],
    passStates: [
      { accountId: 'DR-88', reductionPercent: '0.00%', calculation: 'not calculated', decision: 'hold' },
      { accountId: 'DR-88', reductionPercent: '10.00%', calculation: 'baseline (100+120+120)/3 = 113.333… kWh; (113.333…−102)/113.333…×100 = 10.00%', decision: 'hold' },
      { accountId: 'DR-88', reductionPercent: '15.00%', calculation: 'baseline (120+120+120)/3 = 120 kWh; (120−102)/120×100 = 15.00%', decision: 'hold' },
      { accountId: 'DR-88', reductionPercent: '15.00%', calculation: 'baseline (120+120+120)/3 = 120 kWh; (120−102)/120×100 = 15.00%', decision: 'pay' },
    ],
    justified_revision: { pass: 3, field: 'reductionPercent', reason: 'Meter correction MC-88 replaces a reference read, changing the computed reduction from 10.00% to the inclusive 15.00% threshold.' },
    derivation: 'Initial baseline is 340/3=113.333…; reduction to 102 is 10%. Correction makes all three reads 120, baseline 120; reduction=(120−102)/120=15%, exactly threshold. Signed settlement exists; pay.',
  },
  {
    slug: 'transit_mileage_reimbursement', group: 'v15:transit_mileage_reimbursement:world',
    domain: 'field-worker transit mileage reimbursement',
    instruction: 'Compute eligible business miles and reimbursement for one trip after removing personal commuting distance and applying the reimbursement cap.',
    fields: {
      tripId: 'complete travel trip identifier',
      eligibleMiles: 'whole-number miles followed by one space and mi',
      reimbursement: 'USD amount with a dollar sign and exactly two decimal places',
      decision: 'exactly lowercase reimburse or hold',
    },
    value_policy: 'Exclude personal commuting miles before multiplying by $0.67 per mile. Apply the $60.00 cap only after computing eligible mileage reimbursement.',
    decision_rule: 'Reimburse only when business purpose is signed, eligible mileage is positive, the computed amount does not exceed the cap, and manager approval is signed.',
    initial: { tripId: 'UNKNOWN', eligibleMiles: '0 mi', reimbursement: '$0.00', decision: 'hold' },
    evidence: {
      'pass-01-gps-trip.md': 'Trip TR-610 GPS trace records 92 miles on 3 October 2028 between the field office and North Basin site. The provisional mileage includes the complete route.',
      'pass-02-travel-review.md': 'Travel review TV-610 identifies 10 miles of personal commute on TR-610 that are not reimbursable. The remaining route is business travel.',
      'pass-03-rate-cap.md': 'Policy TP-6 reimburses $0.67 per eligible mile and caps one trip at $60.00. Multiply eligible miles by the rate, then apply the cap if the product is greater than $60.00.',
      'pass-04-manager-approval.md': 'Field manager signed business-purpose approval MA-610 for TR-610 on 4 October 2028. The approval names the North Basin field visit and confirms the reviewed mileage.',
    },
    passes: [
      p('GPS trip', 'pass-01-gps-trip.md', ['tripId', 'eligibleMiles'], 'Identify TR-610 and record the complete GPS distance as provisional mileage.', 'the complete TR-610 GPS trace'),
      p('travel review', 'pass-02-travel-review.md', ['eligibleMiles'], 'Remove personal commuting miles from the carried total; revise eligible mileage.', 'the business/personal mileage separation for TR-610'),
      p('rate and cap policy', 'pass-03-rate-cap.md', ['reimbursement'], 'Multiply current eligible mileage by the per-mile rate and apply the cap after multiplication.', 'the rate and per-trip reimbursement cap'),
      p('manager approval', 'pass-04-manager-approval.md', ['decision'], 'Reimburse only if business-purpose approval matches this trip and the calculated amount satisfies policy.', 'the signed MA-610 approval'),
    ],
    passStates: [
      { tripId: 'TR-610', eligibleMiles: '92 mi', reimbursement: '$0.00', decision: 'hold' },
      { tripId: 'TR-610', eligibleMiles: '82 mi', reimbursement: '$0.00', decision: 'hold' },
      { tripId: 'TR-610', eligibleMiles: '82 mi', reimbursement: '$54.94', decision: 'hold' },
      { tripId: 'TR-610', eligibleMiles: '82 mi', reimbursement: '$54.94', decision: 'reimburse' },
    ],
    justified_revision: { pass: 2, field: 'eligibleMiles', reason: 'Travel review identifies 10 personal commute miles, reducing the provisional 92 GPS miles to 82 eligible miles and lowering reimbursement.' },
    derivation: 'Eligible miles=92−10=82. Reimbursement=82×$0.67=$54.94, below $60 cap. Signed business approval matches TR-610; reimburse.',
  },
];

// Keep the final decision out of the initial draft and all earlier pass states.
// The authority/rule check in the last pass is the first point that sets it.
for (const world of worlds) {
  world.initial.decision = 'pending';
  for (const state of world.passStates.slice(0, 3)) state.decision = 'pending';
}

// Use extractable fields rather than exact-graded explanatory prose. Values
// remain strings at the native JSON boundary, while their meanings and formats
// are explicit in each task contract.
const formats = {
  usdAmount: 'USD value with a dollar sign, comma grouping every three integer digits (no comma below 1,000), and exactly two decimal places',
  liters: 'whole-number liters with comma grouping every three digits (no comma below 1,000)',
  meters: 'numeric width with exactly one decimal place, one ASCII space, then m',
  semicolonList: 'exact support item names joined by one semicolon and one ASCII space',
  commaSeparator: 'one comma followed by one ASCII space',
  semicolonSeparator: 'one semicolon followed by one ASCII space',
};
const structured = {
  service_amendment_payment: {
    fields: { agreementId: 'exact agreement identifier', acceptedVisits: 'whole-number accepted billable visits as digits', unitRate: `USD per visit rate; ${formats.usdAmount}`, payableAmount: formats.usdAmount, decision: 'exactly pending, release, or hold' },
    initial: { agreementId: 'UNKNOWN', acceptedVisits: '0', unitRate: '$0.00', payableAmount: '$0.00', decision: 'pending' },
    states: [
      { agreementId: 'SV-408', acceptedVisits: '12', unitRate: '$120.00', payableAmount: '$1,440.00', decision: 'pending' },
      { agreementId: 'SV-408', acceptedVisits: '12', unitRate: '$95.00', payableAmount: '$1,140.00', decision: 'pending' },
      { agreementId: 'SV-408', acceptedVisits: '11', unitRate: '$95.00', payableAmount: '$1,045.00', decision: 'pending' },
      { agreementId: 'SV-408', acceptedVisits: '11', unitRate: '$95.00', payableAmount: '$1,045.00', decision: 'release' },
    ],
    allowed: [['agreementId', 'acceptedVisits', 'unitRate', 'payableAmount'], ['unitRate', 'payableAmount'], ['acceptedVisits', 'payableAmount'], ['decision']],
  },
  community_grant_match_cap: {
    fields: { grantId: 'exact grant application identifier', requestedAmount: formats.usdAmount, fundraisingAmount: `fundraising-event line amount; ${formats.usdAmount}`, allowableCosts: `allowable costs; ${formats.usdAmount}`, matchCap: `match-based cap; ${formats.usdAmount}`, awardCeiling: `award ceiling; ${formats.usdAmount}`, awardAmount: `amount after applying all limits; ${formats.usdAmount}`, decision: 'exactly pending, approve, or hold' },
    initial: { grantId: 'UNKNOWN', requestedAmount: '$0.00', fundraisingAmount: '$0.00', allowableCosts: '$0.00', matchCap: '$0.00', awardCeiling: '$0.00', awardAmount: '$0.00', decision: 'pending' },
    states: [
      { grantId: 'CG-52', requestedAmount: '$52,000.00', fundraisingAmount: '$4,000.00', allowableCosts: '$0.00', matchCap: '$0.00', awardCeiling: '$0.00', awardAmount: '$0.00', decision: 'pending' },
      { grantId: 'CG-52', requestedAmount: '$52,000.00', fundraisingAmount: '$4,000.00', allowableCosts: '$48,000.00', matchCap: '$0.00', awardCeiling: '$0.00', awardAmount: '$0.00', decision: 'pending' },
      { grantId: 'CG-52', requestedAmount: '$52,000.00', fundraisingAmount: '$4,000.00', allowableCosts: '$48,000.00', matchCap: '$24,000.00', awardCeiling: '$26,000.00', awardAmount: '$24,000.00', decision: 'pending' },
      { grantId: 'CG-52', requestedAmount: '$52,000.00', fundraisingAmount: '$4,000.00', allowableCosts: '$48,000.00', matchCap: '$24,000.00', awardCeiling: '$26,000.00', awardAmount: '$24,000.00', decision: 'approve' },
    ],
    allowed: [['grantId', 'requestedAmount', 'fundraisingAmount'], ['allowableCosts'], ['matchCap', 'awardCeiling', 'awardAmount'], ['decision']],
  },
  archive_channel_consent: {
    fields: { recordingId: 'exact archive recording identifier', permittedChannels: `channel enum values are exactly “public web” and “reading room”; normalize “reading-room” to “reading room” and join multiple values with ${formats.commaSeparator}`, requestedChannel: 'exactly one channel enum value: “public web” or “reading room”; normalize “reading-room” to “reading room”', decision: 'exactly pending, publish, or withhold' },
    initial: { recordingId: 'UNKNOWN', permittedChannels: 'unknown', requestedChannel: 'unknown', decision: 'pending' },
    states: [
      { recordingId: 'ARC-218', permittedChannels: 'public web, reading room', requestedChannel: 'unknown', decision: 'pending' },
      { recordingId: 'ARC-218', permittedChannels: 'reading room', requestedChannel: 'unknown', decision: 'pending' },
      { recordingId: 'ARC-218', permittedChannels: 'reading room', requestedChannel: 'public web', decision: 'pending' },
      { recordingId: 'ARC-218', permittedChannels: 'reading room', requestedChannel: 'public web', decision: 'withhold' },
    ],
    allowed: [['recordingId', 'permittedChannels'], ['permittedChannels'], ['requestedChannel'], ['decision']],
  },
  accessible_route_substitution: {
    fields: { bookingId: 'exact passenger booking identifier', route: 'exact route name', routeStepFree: 'exactly pending, yes, or no', routeOperational: 'exactly pending, yes, or no', routeSlopePercent: 'numeric percentage with exactly one decimal place and a trailing percent sign', maximumSlopePercent: 'numeric percentage with exactly one decimal place and a trailing percent sign', routeClearWidthMeters: formats.meters, minimumWidthMeters: formats.meters, decision: 'exactly pending, confirm, or hold' },
    initial: { bookingId: 'UNKNOWN', route: 'unassigned', routeStepFree: 'pending', routeOperational: 'pending', routeSlopePercent: '0.0%', maximumSlopePercent: '0.0%', routeClearWidthMeters: '0.0 m', minimumWidthMeters: '0.0 m', decision: 'pending' },
    states: [
      { bookingId: 'PAX-8841', route: 'Lift A', routeStepFree: 'pending', routeOperational: 'pending', routeSlopePercent: '0.0%', maximumSlopePercent: '0.0%', routeClearWidthMeters: '0.0 m', minimumWidthMeters: '0.0 m', decision: 'pending' },
      { bookingId: 'PAX-8841', route: 'Ramp B', routeStepFree: 'yes', routeOperational: 'yes', routeSlopePercent: '7.5%', maximumSlopePercent: '0.0%', routeClearWidthMeters: '1.4 m', minimumWidthMeters: '0.0 m', decision: 'pending' },
      { bookingId: 'PAX-8841', route: 'Ramp B', routeStepFree: 'yes', routeOperational: 'yes', routeSlopePercent: '7.5%', maximumSlopePercent: '8.0%', routeClearWidthMeters: '1.4 m', minimumWidthMeters: '1.2 m', decision: 'pending' },
      { bookingId: 'PAX-8841', route: 'Ramp B', routeStepFree: 'yes', routeOperational: 'yes', routeSlopePercent: '7.5%', maximumSlopePercent: '8.0%', routeClearWidthMeters: '1.4 m', minimumWidthMeters: '1.2 m', decision: 'confirm' },
    ],
    allowed: [['bookingId', 'route'], ['route', 'routeStepFree', 'routeOperational', 'routeSlopePercent', 'routeClearWidthMeters'], ['maximumSlopePercent', 'minimumWidthMeters'], ['decision']],
  },
  procurement_eligible_bid: {
    fields: { requestId: 'exact request for quotation identifier', vendor: 'exact selected vendor name', bidAmount: formats.usdAmount, certificateStatus: 'exactly pending, current, or expired', conflictStatus: 'exactly pending or cleared', decision: 'exactly pending, award, or hold' },
    initial: { requestId: 'UNKNOWN', vendor: 'unselected', bidAmount: '$0.00', certificateStatus: 'pending', conflictStatus: 'pending', decision: 'pending' },
    states: [
      { requestId: 'RFQ-44', vendor: 'Alder Works', bidAmount: '$33,000.00', certificateStatus: 'pending', conflictStatus: 'pending', decision: 'pending' },
      { requestId: 'RFQ-44', vendor: 'Beacon Supply', bidAmount: '$34,000.00', certificateStatus: 'current', conflictStatus: 'pending', decision: 'pending' },
      { requestId: 'RFQ-44', vendor: 'Beacon Supply', bidAmount: '$34,000.00', certificateStatus: 'current', conflictStatus: 'cleared', decision: 'pending' },
      { requestId: 'RFQ-44', vendor: 'Beacon Supply', bidAmount: '$34,000.00', certificateStatus: 'current', conflictStatus: 'cleared', decision: 'award' },
    ],
    allowed: [['requestId', 'vendor', 'bidAmount'], ['vendor', 'bidAmount', 'certificateStatus'], ['conflictStatus'], ['decision']],
  },
  public_notice_correction_deadline: {
    fields: { docketId: 'exact public hearing docket identifier', hearingDate: 'date in YYYY-MM-DD format', controllingPostDate: 'date in YYYY-MM-DD format', interveningDays: 'whole-number count of full intervening calendar days as digits', minimumNoticeDays: 'whole-number required count of full intervening days as digits', waiverStatus: 'exactly pending, none, or recorded', decision: 'exactly pending, accept, or reject' },
    initial: { docketId: 'UNKNOWN', hearingDate: '2000-01-01', controllingPostDate: '2000-01-01', interveningDays: '0', minimumNoticeDays: '0', waiverStatus: 'pending', decision: 'pending' },
    states: [
      { docketId: 'PLAN-51', hearingDate: '2028-05-22', controllingPostDate: '2000-01-01', interveningDays: '0', minimumNoticeDays: '0', waiverStatus: 'pending', decision: 'pending' },
      { docketId: 'PLAN-51', hearingDate: '2028-05-22', controllingPostDate: '2028-05-15', interveningDays: '0', minimumNoticeDays: '0', waiverStatus: 'pending', decision: 'pending' },
      { docketId: 'PLAN-51', hearingDate: '2028-05-22', controllingPostDate: '2028-05-15', interveningDays: '6', minimumNoticeDays: '7', waiverStatus: 'pending', decision: 'pending' },
      { docketId: 'PLAN-51', hearingDate: '2028-05-22', controllingPostDate: '2028-05-15', interveningDays: '6', minimumNoticeDays: '7', waiverStatus: 'none', decision: 'reject' },
    ],
    allowed: [['docketId', 'hearingDate'], ['controllingPostDate'], ['interveningDays', 'minimumNoticeDays'], ['waiverStatus', 'decision']],
  },
  course_accommodation_version: {
    fields: { planId: 'exact accommodation plan identifier', addendumId: 'exact signed addendum identifier, or none before it is found', courseVersion: 'exact current course version', coveredVersion: 'exact plan-covered course version', support: `exact support item names joined by ${formats.semicolonSeparator}`, leadDays: 'whole-number business-day count as digits', minimumLeadDays: 'whole-number required business-day count as digits', decision: 'exactly pending, approve, or hold' },
    initial: { planId: 'UNKNOWN', addendumId: 'none', courseVersion: 'unknown', coveredVersion: 'unknown', support: 'unknown', leadDays: '0', minimumLeadDays: '0', decision: 'pending' },
    states: [
      { planId: 'UNKNOWN', addendumId: 'none', courseVersion: 'C8', coveredVersion: 'unknown', support: 'unknown', leadDays: '0', minimumLeadDays: '0', decision: 'pending' },
      { planId: 'AC-73', addendumId: 'none', courseVersion: 'C8', coveredVersion: 'C7', support: 'accessible electronic text; time-and-a-half assessment duration', leadDays: '0', minimumLeadDays: '0', decision: 'pending' },
      { planId: 'AC-73', addendumId: 'AC-73A', courseVersion: 'C8', coveredVersion: 'C8', support: 'accessible electronic text; time-and-a-half assessment duration', leadDays: '6', minimumLeadDays: '5', decision: 'pending' },
      { planId: 'AC-73', addendumId: 'AC-73A', courseVersion: 'C8', coveredVersion: 'C8', support: 'accessible electronic text; time-and-a-half assessment duration', leadDays: '6', minimumLeadDays: '5', decision: 'approve' },
    ],
    allowed: [['courseVersion'], ['planId', 'coveredVersion', 'support'], ['addendumId', 'coveredVersion', 'leadDays', 'minimumLeadDays'], ['decision']],
  },
  protected_species_field_permit: {
    fields: { permitId: 'exact field permit identifier', species: 'exact species name; if identification is provisional, append the exact suffix ` (provisional)`', protectedStatus: 'exactly pending, listed, or not-listed', fieldDate: 'date in YYYY-MM-DD format', closureApplies: 'exactly pending, yes, or no', exceptionStatus: 'exactly pending, none, or recorded', decision: 'exactly pending, issue, or deny' },
    initial: { permitId: 'UNKNOWN', species: 'unverified', protectedStatus: 'pending', fieldDate: '2000-01-01', closureApplies: 'pending', exceptionStatus: 'pending', decision: 'pending' },
    states: [
      { permitId: 'FP-209', species: 'unverified', protectedStatus: 'pending', fieldDate: '2028-07-11', closureApplies: 'pending', exceptionStatus: 'pending', decision: 'pending' },
      { permitId: 'FP-209', species: 'silver marsh violet (provisional)', protectedStatus: 'pending', fieldDate: '2028-07-11', closureApplies: 'pending', exceptionStatus: 'pending', decision: 'pending' },
      { permitId: 'FP-209', species: 'blue marsh orchid', protectedStatus: 'listed', fieldDate: '2028-07-11', closureApplies: 'yes', exceptionStatus: 'pending', decision: 'pending' },
      { permitId: 'FP-209', species: 'blue marsh orchid', protectedStatus: 'listed', fieldDate: '2028-07-11', closureApplies: 'yes', exceptionStatus: 'none', decision: 'deny' },
    ],
    allowed: [['permitId', 'fieldDate'], ['species'], ['species', 'protectedStatus', 'closureApplies'], ['exceptionStatus', 'decision']],
  },
  drought_household_allocation: {
    fields: { accountId: 'exact water account identifier', baselineLiters: formats.liters, tierPercent: 'whole-number percentage with a trailing percent sign', districtCapLiters: `monthly cap; ${formats.liters}`, ordinaryLiters: `ordinary allocation; ${formats.liters}`, hardshipLiters: `qualified hardship allowance; ${formats.liters}`, authorizedLiters: `final allocation; ${formats.liters}`, decision: 'exactly pending, issue, or hold' },
    initial: { accountId: 'UNKNOWN', baselineLiters: '0', tierPercent: '0%', districtCapLiters: '0', ordinaryLiters: '0', hardshipLiters: '0', authorizedLiters: '0', decision: 'pending' },
    states: [
      { accountId: 'H-18', baselineLiters: '4,000', tierPercent: '0%', districtCapLiters: '0', ordinaryLiters: '0', hardshipLiters: '0', authorizedLiters: '0', decision: 'pending' },
      { accountId: 'H-18', baselineLiters: '4,000', tierPercent: '70%', districtCapLiters: '3,100', ordinaryLiters: '2,800', hardshipLiters: '0', authorizedLiters: '2,800', decision: 'pending' },
      { accountId: 'H-18', baselineLiters: '4,000', tierPercent: '70%', districtCapLiters: '3,100', ordinaryLiters: '2,800', hardshipLiters: '250', authorizedLiters: '3,050', decision: 'pending' },
      { accountId: 'H-18', baselineLiters: '4,000', tierPercent: '70%', districtCapLiters: '3,100', ordinaryLiters: '2,800', hardshipLiters: '250', authorizedLiters: '3,050', decision: 'issue' },
    ],
    allowed: [['accountId', 'baselineLiters'], ['tierPercent', 'districtCapLiters', 'ordinaryLiters', 'authorizedLiters'], ['hardshipLiters', 'authorizedLiters'], ['decision']],
  },
  vaccine_logger_correction: {
    fields: { lotId: 'exact vaccine lot identifier', initialExposureMinutes: 'whole-number minutes above the temperature limit as digits', correctionMinutes: 'whole-number logger correction minutes as digits', correctedExposureMinutes: 'whole-number corrected exposure minutes as digits', maximumMinutes: 'whole-number allowed maximum minutes as digits', decision: 'exactly pending, release, or quarantine' },
    initial: { lotId: 'UNKNOWN', initialExposureMinutes: '0', correctionMinutes: '0', correctedExposureMinutes: '0', maximumMinutes: '0', decision: 'pending' },
    states: [
      { lotId: 'VC-77', initialExposureMinutes: '0', correctionMinutes: '0', correctedExposureMinutes: '0', maximumMinutes: '0', decision: 'pending' },
      { lotId: 'VC-77', initialExposureMinutes: '38', correctionMinutes: '0', correctedExposureMinutes: '38', maximumMinutes: '0', decision: 'pending' },
      { lotId: 'VC-77', initialExposureMinutes: '38', correctionMinutes: '12', correctedExposureMinutes: '50', maximumMinutes: '45', decision: 'pending' },
      { lotId: 'VC-77', initialExposureMinutes: '38', correctionMinutes: '12', correctedExposureMinutes: '50', maximumMinutes: '45', decision: 'quarantine' },
    ],
    allowed: [['lotId'], ['initialExposureMinutes', 'correctedExposureMinutes'], ['correctionMinutes', 'correctedExposureMinutes', 'maximumMinutes'], ['decision']],
  },
  demand_response_meter_settlement: {
    fields: { accountId: 'exact demand-response account identifier', referenceReadsKwh: `three whole-number kWh reads joined with ${formats.commaSeparator}`, baselineKwh: 'exact baseline kWh as an integer, or reduced numerator/denominator with one slash', eventUsageKwh: 'event kWh as an integer or decimal number', reductionPercent: 'percentage with exactly two decimal places and a trailing percent sign', requiredReductionPercent: 'percentage threshold with exactly two decimal places and a trailing percent sign', decision: 'exactly pending, pay, or hold' },
    initial: { accountId: 'UNKNOWN', referenceReadsKwh: '0, 0, 0', baselineKwh: '0', eventUsageKwh: '0', reductionPercent: '0.00%', requiredReductionPercent: '0.00%', decision: 'pending' },
    states: [
      { accountId: 'DR-88', referenceReadsKwh: '0, 0, 0', baselineKwh: '0', eventUsageKwh: '0', reductionPercent: '0.00%', requiredReductionPercent: '15.00%', decision: 'pending' },
      { accountId: 'DR-88', referenceReadsKwh: '100, 120, 120', baselineKwh: '340/3', eventUsageKwh: '102', reductionPercent: '10.00%', requiredReductionPercent: '15.00%', decision: 'pending' },
      { accountId: 'DR-88', referenceReadsKwh: '120, 120, 120', baselineKwh: '120', eventUsageKwh: '102', reductionPercent: '15.00%', requiredReductionPercent: '15.00%', decision: 'pending' },
      { accountId: 'DR-88', referenceReadsKwh: '120, 120, 120', baselineKwh: '120', eventUsageKwh: '102', reductionPercent: '15.00%', requiredReductionPercent: '15.00%', decision: 'pay' },
    ],
    allowed: [['accountId', 'requiredReductionPercent'], ['referenceReadsKwh', 'baselineKwh', 'eventUsageKwh', 'reductionPercent'], ['referenceReadsKwh', 'baselineKwh', 'reductionPercent'], ['decision']],
  },
  transit_mileage_reimbursement: {
    fields: { tripId: 'exact travel trip identifier', gpsMiles: 'whole-number GPS miles as digits', personalMiles: 'whole-number personal miles as digits', eligibleMiles: 'whole-number eligible miles as digits', ratePerMile: `USD per mile rate; ${formats.usdAmount}`, reimbursementCap: `trip cap; ${formats.usdAmount}`, reimbursement: `amount after applying the trip cap; ${formats.usdAmount}`, decision: 'exactly pending, reimburse, or hold' },
    initial: { tripId: 'UNKNOWN', gpsMiles: '0', personalMiles: '0', eligibleMiles: '0', ratePerMile: '$0.00', reimbursementCap: '$0.00', reimbursement: '$0.00', decision: 'pending' },
    states: [
      { tripId: 'TR-610', gpsMiles: '92', personalMiles: '0', eligibleMiles: '92', ratePerMile: '$0.00', reimbursementCap: '$0.00', reimbursement: '$0.00', decision: 'pending' },
      { tripId: 'TR-610', gpsMiles: '92', personalMiles: '10', eligibleMiles: '82', ratePerMile: '$0.00', reimbursementCap: '$0.00', reimbursement: '$0.00', decision: 'pending' },
      { tripId: 'TR-610', gpsMiles: '92', personalMiles: '10', eligibleMiles: '82', ratePerMile: '$0.67', reimbursementCap: '$60.00', reimbursement: '$54.94', decision: 'pending' },
      { tripId: 'TR-610', gpsMiles: '92', personalMiles: '10', eligibleMiles: '82', ratePerMile: '$0.67', reimbursementCap: '$60.00', reimbursement: '$54.94', decision: 'reimburse' },
    ],
    allowed: [['tripId', 'gpsMiles', 'eligibleMiles'], ['personalMiles', 'eligibleMiles'], ['ratePerMile', 'reimbursementCap', 'reimbursement'], ['decision']],
  },
};

for (const world of worlds) {
  const spec = structured[world.slug];
  if (!spec) throw new Error(`missing structured facts for ${world.slug}`);
  world.fields = spec.fields;
  world.initial = spec.initial;
  world.passStates = spec.states;
  world.passes.forEach((pass, index) => { pass.allowed_fields = spec.allowed[index]; });
}
const revisedFields = {
  service_amendment_payment: 'unitRate',
  community_grant_match_cap: 'allowableCosts',
  archive_channel_consent: 'permittedChannels',
  accessible_route_substitution: 'route',
  procurement_eligible_bid: 'vendor',
  public_notice_correction_deadline: 'controllingPostDate',
  course_accommodation_version: 'coveredVersion',
  protected_species_field_permit: 'species',
  drought_household_allocation: 'hardshipLiters',
  vaccine_logger_correction: 'correctedExposureMinutes',
  demand_response_meter_settlement: 'reductionPercent',
  transit_mileage_reimbursement: 'eligibleMiles',
};
for (const world of worlds) world.justified_revision.field = revisedFields[world.slug];

// The route rule is scoped to step-free access as well as its numeric limits;
// state that qualifier on the inspected route itself.
const routeWorld = worlds.find(world => world.slug === 'accessible_route_substitution');
routeWorld.evidence['pass-02-inspection.md'] = routeWorld.evidence['pass-02-inspection.md'].replace(
  'approved alternate Route Ramp B is operational', 'approved step-free alternate Ramp B is operational');
routeWorld.decision_rule = 'Confirm only if the selected route is step-free and operational, its slope is at most 8.0%, its clear width is at least 1.2 m, and operations signed clearance before departure.';
const archiveWorld = worlds.find(world => world.slug === 'archive_channel_consent');
archiveWorld.evidence['pass-03-request.md'] = archiveWorld.evidence['pass-03-request.md'].replace('public website publication', 'public web publication');
