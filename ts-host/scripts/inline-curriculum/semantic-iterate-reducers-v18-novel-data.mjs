const joinIds = rows => rows.length ? rows.map(row => row.id).join('; ') : 'none';

function caseFrom(spec) {
  const rows = spec.candidates;
  const ordered = [...rows].sort(spec.compare);
  const provisional = spec.provisional(ordered);
  const selected = spec.select(rows.filter(spec.eligible).sort(spec.compare));
  const selectedItems = spec.format(selected), provisionalItems = spec.format(provisional);
  const measure = spec.measure(selected), provisionalMeasure = spec.measure(provisional);
  const authorized = typeof spec.authorized === 'function' ? spec.authorized(selected) : spec.authorized;
  const decision = selected.length === 0 ? spec.noAction : (authorized ? spec.approvedAction : 'hold');
  const fields = {
    caseId: `Exact request identifier for the ${spec.domain} decision.`,
    selectedItems: spec.selectionFormat,
    measure: spec.measureFormat,
    decision: spec.decisionFormat ?? `Final disposition: exactly ${spec.finalDecisions.join(', ')}. Selection and authorization are separate; do not change selectedItems merely because authorization is absent.`,
  };
  const initial = { caseId: 'UNKNOWN', selectedItems: 'pending', measure: 'pending', decision: 'pending' };
  const passStates = [
    { ...initial, caseId: spec.requestId },
    { ...initial, caseId: spec.requestId, selectedItems: provisionalItems, measure: provisionalMeasure },
    { ...initial, caseId: spec.requestId, selectedItems, measure },
    { ...initial, caseId: spec.requestId, selectedItems, measure, decision },
  ];
  const evidence = {
    'pass-01-request.md': `${spec.owner} opened ${spec.domain} request ${spec.requestId}. ${spec.requestText} Decision rule: ${spec.ruleText}`,
    'pass-02-register.md': `${spec.registerHeading} for ${spec.requestId}:\n${rows.map(spec.registerLine).join('\n')}`,
    'pass-03-conditions.md': `${spec.auditHeading} for ${spec.requestId}:\n${rows.map(spec.auditLine).join('\n')}\n${spec.exceptionText}`,
    'pass-04-authority.md': `${spec.authorityHeading} for ${spec.requestId}. ${spec.authorityText}`,
  };
  const passes = [
    { name:'request scope and rule', evidence_path:'pass-01-request.md', allowed_fields:['caseId'], constraint:'Copy the complete request identifier and establish the requested operation and explicit decision rule.', source_scope:`request ${spec.requestId}` },
    { name:'candidate register', evidence_path:'pass-02-register.md', allowed_fields:['selectedItems','measure'], constraint:'Make a provisional ordering or worklist from registered metrics only. Eligibility, exceptions, and authorization have not yet been checked; keep it provisional.', source_scope:`register for ${spec.requestId}` },
    { name:'conditions and exceptions', evidence_path:'pass-03-conditions.md', allowed_fields:['selectedItems','measure'], constraint:'Apply every stated condition and scoped exception to the candidates. Recompute the selected set and measure; add no unstated conditions.', source_scope:`condition audit for ${spec.requestId}` },
    { name:'execution authority', evidence_path:'pass-04-authority.md', allowed_fields:['decision'], constraint:'Keep the evidence-derived selection and measure. Determine whether execution is authorized under the exact stated authority rule.', source_scope:`authority record for ${spec.requestId}` },
  ];
  const changed=['selectedItems','measure'].find(field=>passStates[1][field]!==passStates[2][field]);
  if (!changed) throw new Error(`${spec.slug}: no source-derived provisional correction`);
  return {
    slug:spec.slug, group:`v18:${spec.slug}:world`, domain:spec.domain, fields,
    field_enums:{decision:{intermediate:['pending',...spec.finalDecisions],final:spec.finalDecisions}},
    initial,passes,passStates,evidence,
    instruction:`${spec.instruction}\n\nUse task.json and the four named source records. The candidate register supports only a provisional result. Apply the full condition audit, including scoped exceptions, before fixing selectedItems and measure. Apply authorization only to decision: selection evidence and permission to execute are distinct. Follow the exact tie rule and output formats. If nothing qualifies, use ${spec.noAction}. Do not infer facts absent from current and accumulated source records.`,
    justified_revision:{pass:3,field:changed,reason:`Pass-three source facts revise the provisional result (${provisionalItems}) to the qualified result (${selectedItems}) under the explicit rule.`},
    decision_rule:spec.ruleText,
    source_summary:{requestId:spec.requestId,candidates:rows.map(row=>({...row})),selectedItems,measure,decision,authorized},
  };
}

const authoredWorlds=[
  caseFrom({
    slug:'software-release-gate',domain:'software release gating',owner:'Platform release desk',requestId:'REL-640',
    requestText:'Choose the newest candidate build eligible for production deployment. The deployment window is open until 18:00 UTC.',
    ruleText:'A build qualifies if its required tests passed and it has no blocker, or exactly one noncritical blocker covered by a waiver naming that same complete build ID. Select the qualifying build with the greatest numeric version rank. Deployment requires a signed change order and an open deployment window.',
    instruction:'Resolve the production release candidate for REL-640. A waiver is scoped to the exact build named in it.',
    registerHeading:'Build version register',auditHeading:'Test, blocker, and waiver audit',authorityHeading:'Change order and deployment window record',
    candidates:[
      {id:'BUILD-8',versionRank:8,tests:true,blockers:1,critical:false,waiverBuild:'BUILD-7'},
      {id:'BUILD-7',versionRank:7,tests:true,blockers:1,critical:false,waiverBuild:'BUILD-7'},
      {id:'BUILD-6',versionRank:6,tests:true,blockers:0,critical:false,waiverBuild:'none'},
      {id:'BUILD-9',versionRank:9,tests:false,blockers:0,critical:false,waiverBuild:'none'},
    ],
    compare:(a,b)=>b.versionRank-a.versionRank||a.id.localeCompare(b.id),provisional:rows=>rows.slice(0,1),
    eligible:r=>r.tests&&(r.blockers===0||(!r.critical&&r.blockers===1&&r.waiverBuild===r.id)),select:rows=>rows.slice(0,1),format:joinIds,
    measure:rows=>rows.length?String(rows[0].versionRank):'0',registerLine:r=>`${r.id}: numeric version rank ${r.versionRank}.`,
    auditLine:r=>`${r.id}: required tests passed=${r.tests}; blocker count=${r.blockers}; blocker critical=${r.critical}; waiver names build=${r.waiverBuild}.`,
    exceptionText:'A noncritical blocker waiver applies only to the exact build ID it names. It cannot waive failed tests, a critical blocker, or a different build.',
    authorityText:'Signed change order CO-884 covers REL-640. The production deployment window remains open until 18:00 UTC.',authorized:true,
    selectionFormat:'One complete build ID, or none.',measureFormat:'Numeric version rank of the selected build as digits; 0 when none.',
    decisionFormat:'Use deploy only when a build qualifies, a signed change order is present, and the deployment window is open. If a build qualifies but either authority condition is missing, use hold. Use no_action only when no build qualifies. Return exactly one bare literal: deploy, hold, or no_action.',
    finalDecisions:['deploy','hold','no_action'],approvedAction:'deploy',noAction:'no_action',
  }),
  caseFrom({
    slug:'dependency-upgrade-batch',domain:'software dependency upgrade batching',owner:'Open-source security maintainers',requestId:'DEP-219',
    requestText:'Select at most two dependency updates for the next maintenance batch.',
    ruleText:'An update qualifies when review is complete, compatibility tests pass, and it has no unresolved dependency conflict. Select at most two qualifying updates by descending measured risk reduction, ties by ascending complete dependency ID. Merging requires two maintainer approvals.',
    instruction:'Prepare the DEP-219 upgrade batch. Report qualified updates independently of whether the maintainers can merge them now.',
    registerHeading:'Risk-reduction register',auditHeading:'Review, compatibility, and conflict audit',authorityHeading:'Maintainer approval record',
    candidates:[
      {id:'PKG-A',reduction:91,reviewed:true,tests:true,conflict:false}, {id:'PKG-B',reduction:99,reviewed:false,tests:true,conflict:false},
      {id:'PKG-C',reduction:91,reviewed:true,tests:true,conflict:false}, {id:'PKG-D',reduction:86,reviewed:true,tests:true,conflict:false},
      {id:'PKG-E',reduction:96,reviewed:true,tests:true,conflict:true},
    ],
    compare:(a,b)=>b.reduction-a.reduction||a.id.localeCompare(b.id),provisional:rows=>rows.slice(0,2),
    eligible:r=>r.reviewed&&r.tests&&!r.conflict,select:rows=>rows.slice(0,2),format:joinIds,
    measure:rows=>rows.length?rows.map(r=>String(r.reduction)).join('; '):'none',
    registerLine:r=>`${r.id}: estimated risk reduction ${r.reduction} points.`,
    auditLine:r=>`${r.id}: security review complete=${r.reviewed}; compatibility tests pass=${r.tests}; unresolved dependency conflict=${r.conflict}.`,
    exceptionText:'No risk score overrides an incomplete review, failed compatibility test, or unresolved dependency conflict.',
    authorityText:'One maintainer approval is recorded; two distinct approvals are required to merge this batch.',authorized:false,
    selectionFormat:'Up to two complete dependency IDs joined by exactly semicolon and one space; descending risk reduction then ID; none if empty.',
    measureFormat:'Corresponding risk reduction points in selected order, joined by exactly semicolon and one space; none if empty.',
    finalDecisions:['merge','hold','no_action'],approvedAction:'merge',noAction:'no_action',
  }),
  caseFrom({
    slug:'research-instrument-booking',domain:'research instrument booking',owner:'Shared laboratory scheduler',requestId:'LAB-362',
    requestText:'Book one 90-minute instrument slot for the microscopy team before 11:00.',
    ruleText:'A slot qualifies only when the instrument is calibrated, the assigned researcher has current training, and the slot has no reservation conflict. An after-hours safety exception is allowed only when the permit names the same instrument and slot. Choose the earliest qualifying start time, ties by ascending complete slot ID. Booking requires the principal investigator approval and equipment availability.',
    instruction:'Resolve instrument booking LAB-362. Apply an after-hours exception only to the exact instrument and slot named by its permit.',
    registerHeading:'Instrument slot register',auditHeading:'Calibration, training, reservation, and safety-permit audit',authorityHeading:'Booking authority record',
    candidates:[
      {id:'SLOT-A',start:'08:00',end:'09:30',duration:90,instrument:'MIC-4',calibrated:false,training:true,conflict:false,afterHours:false,permit:'none'},
      {id:'SLOT-B',start:'08:30',end:'10:00',duration:90,instrument:'MIC-2',calibrated:true,training:true,conflict:false,afterHours:false,permit:'none'},
      {id:'SLOT-C',start:'08:30',end:'10:00',duration:90,instrument:'MIC-3',calibrated:true,training:true,conflict:false,afterHours:true,permit:'MIC-3/SLOT-C'},
      {id:'SLOT-D',start:'08:10',end:'09:40',duration:90,instrument:'MIC-1',calibrated:true,training:false,conflict:false,afterHours:false,permit:'none'},
    ],
    compare:(a,b)=>a.start.localeCompare(b.start)||a.id.localeCompare(b.id),provisional:rows=>rows.slice(0,1),
    eligible:r=>r.calibrated&&r.training&&!r.conflict&&(!r.afterHours||r.permit===`${r.instrument}/${r.id}`),select:rows=>rows.slice(0,1),format:joinIds,
    measure:rows=>rows.length?rows[0].start:'none',registerLine:r=>`${r.id}: start ${r.start}; duration ${r.duration} minutes; end ${r.end}; instrument ${r.instrument}.`,
    auditLine:r=>`${r.id}: calibrated=${r.calibrated}; researcher training current=${r.training}; reservation conflict=${r.conflict}; after-hours=${r.afterHours}; safety permit names=${r.permit}.`,
    exceptionText:'A permit for another instrument or slot does not waive the after-hours rule. It cannot waive calibration, training, or an existing reservation.',
    authorityText:'The PI approved LAB-362 and the selected equipment is available.',authorized:true,
    selectionFormat:'One complete slot ID or none.',measureFormat:'Selected slot start time in HH:MM or none.',
    finalDecisions:['book','hold','no_action'],approvedAction:'book',noAction:'no_action',
  }),
  caseFrom({
    slug:'privacy-export-minimization',domain:'privacy-preserving data export',owner:'Regional outcomes analytics unit',requestId:'PX-105',
    requestText:'The regional outcome report requests all five schema fields: region, age_band, patient_name, rare_condition, and outcome_code. The approved purpose is regional-outcomes.',
    ruleText:'Start with all five requested fields. Keep a field when it is public or its restriction has explicit consent for the exact approved purpose, regional-outcomes. Then remove every direct identifier, regardless of request status, public classification, or consent. Sort the remaining fields by schema position. The data protection officer must approve the export.',
    instruction:'Derive the minimum permitted field list for PX-105. Do not confuse a field being selected with approval to export the file.',
    registerHeading:'Schema position register',auditHeading:'Classification, consent-purpose, and identifier audit',authorityHeading:'Export approval record',
    candidates:[
      {id:'region',position:1,requested:true,classification:'public',consentedPurposes:[],directIdentifier:false},
      {id:'age_band',position:2,requested:true,classification:'restricted',consentedPurposes:['regional-outcomes'],directIdentifier:false},
      {id:'patient_name',position:3,requested:true,classification:'public',consentedPurposes:['regional-outcomes'],directIdentifier:true},
      {id:'rare_condition',position:4,requested:true,classification:'restricted',consentedPurposes:['care-quality'],directIdentifier:false},
      {id:'outcome_code',position:5,requested:true,classification:'public',consentedPurposes:[],directIdentifier:false},
    ],
    compare:(a,b)=>a.position-b.position||a.id.localeCompare(b.id),provisional:rows=>rows,eligible:r=>r.requested&&!r.directIdentifier&&(r.classification==='public'||r.consentedPurposes.includes('regional-outcomes')),select:rows=>rows,format:joinIds,
    measure:rows=>String(rows.length),registerLine:r=>`${r.id}: schema position ${r.position}.`,
    auditLine:r=>`${r.id}: requested for this report=${r.requested}; classification=${r.classification}; explicit consent purposes=${r.consentedPurposes.join(',')||'none'}; direct identifier=${r.directIdentifier}.`,
    exceptionText:'Consent for care-quality does not cover regional-outcomes. Public classification never overrides the direct-identifier exclusion.',
    authorityText:'The data protection officer has not approved export PX-105. Preserve the minimum eligible field list, but set the disposition to hold.',authorized:false,
    selectionFormat:'Field names joined by exactly semicolon and one space in ascending schema position; none if empty.',measureFormat:'Number of included fields as digits only.',
    finalDecisions:['export','hold','no_action'],approvedAction:'export',noAction:'no_action',
  }),
  caseFrom({
    slug:'incident-mitigation-selection',domain:'incident response mitigation selection',owner:'Service incident commander',requestId:'INC-903',
    requestText:'Select at most two mitigation actions for outage INC-903 to reduce customer impact.',
    ruleText:'An action qualifies if all listed prerequisites are complete and its blast radius is within the approved cap. An untested action may qualify only under a commander waiver naming that exact action. Select at most two by descending measured risk reduction, ties by ascending complete action ID. Execution requires the incident commander authorization.',
    instruction:'Compute the mitigation selection for INC-903 and report authorization separately.',
    registerHeading:'Mitigation impact register',auditHeading:'Prerequisite, blast-radius, and exact-action waiver audit',authorityHeading:'Execution authority record',
    candidates:[
      {id:'ACT-A',reduction:94,prerequisites:false,blast:2,cap:3,tested:true,waiver:'none'},
      {id:'ACT-B',reduction:91,prerequisites:true,blast:2,cap:3,tested:true,waiver:'none'},
      {id:'ACT-C',reduction:91,prerequisites:true,blast:3,cap:3,tested:false,waiver:'ACT-C'},
      {id:'ACT-D',reduction:86,prerequisites:true,blast:4,cap:3,tested:true,waiver:'none'},
    ],
    compare:(a,b)=>b.reduction-a.reduction||a.id.localeCompare(b.id),provisional:rows=>rows.slice(0,2),
    eligible:r=>r.prerequisites&&r.blast<=r.cap&&(r.tested||r.waiver===r.id),select:rows=>rows.slice(0,2),format:joinIds,
    measure:rows=>rows.length?rows.map(r=>String(r.reduction)).join('; '):'none',
    registerLine:r=>`${r.id}: measured risk reduction ${r.reduction} points.`,
    auditLine:r=>`${r.id}: prerequisites complete=${r.prerequisites}; blast radius=${r.blast}; approved cap=${r.cap}; tested=${r.tested}; waiver names=${r.waiver}.`,
    exceptionText:'The exact-action waiver can permit an untested action only inside the approved blast-radius cap and after its prerequisites are complete.',
    authorityText:'No commander execution authorization is recorded yet; keep the selected mitigations and set the disposition to hold.',authorized:false,
    selectionFormat:'Up to two complete action IDs joined by exactly semicolon and one space; descending risk reduction then ID; none if empty.',
    measureFormat:'Corresponding risk reduction points in the same order, joined by exactly semicolon and one space; none if empty.',
    finalDecisions:['execute','hold','no_action'],approvedAction:'execute',noAction:'no_action',
  }),
  caseFrom({
    slug:'research-sample-release',domain:'research sample batch release',owner:'Laboratory quality assurance',requestId:'SR-448',
    requestText:'Identify sample batches eligible for the next analysis run.',
    ruleText:'A batch qualifies when chain of custody is intact and quality control passed. A failed initial QC may be superseded only by a final passing retest linked to that same complete batch ID. Order qualifying batches by collection time ascending, then complete batch ID. Release requires the QA lead signature.',
    instruction:'Prepare the SR-448 eligible analysis batch list. A retest corrects only the batch to which it is explicitly linked.',
    registerHeading:'Collection and initial QC register',auditHeading:'Chain-of-custody and linked retest audit',authorityHeading:'Analysis release authority',
    candidates:[
      {id:'SMP-A',collected:'07:10',custody:true,initialQc:'fail',retest:'pass:SMP-A'},
      {id:'SMP-B',collected:'07:25',custody:true,initialQc:'pass',retest:'none'},
      {id:'SMP-C',collected:'07:05',custody:false,initialQc:'pass',retest:'none'},
      {id:'SMP-D',collected:'07:40',custody:true,initialQc:'fail',retest:'pass:SMP-A'},
    ],
    compare:(a,b)=>a.collected.localeCompare(b.collected)||a.id.localeCompare(b.id),provisional:rows=>rows,
    eligible:r=>r.custody&&(r.initialQc==='pass'||r.retest===`pass:${r.id}`),select:rows=>rows,format:joinIds,
    measure:rows=>String(rows.length),registerLine:r=>`${r.id}: collection time ${r.collected}; initial QC ${r.initialQc}.`,
    auditLine:r=>`${r.id}: chain of custody intact=${r.custody}; final retest record=${r.retest}.`,
    exceptionText:'A retest labeled for SMP-A cannot supersede the failed QC on SMP-D. Retest identity must match the complete batch ID.',
    authorityText:'QA lead signature for SR-448 is recorded.',authorized:true,
    selectionFormat:'Complete sample batch IDs joined by exactly semicolon and one space, ordered by collection time then ID; none if empty.',measureFormat:'Number of eligible sample batches as digits only.',
    finalDecisions:['release','hold','no_action'],approvedAction:'release',noAction:'no_action',
  }),
  caseFrom({
    slug:'volunteer-shift-coverage',domain:'volunteer shift coverage reduction',owner:'Community response coordinator',requestId:'VS-610',
    requestText:'Fill the two critical response roles: interpreter (priority 1) and radio operator (priority 2). A volunteer may fill at most one role.',
    ruleText:'A person-role assignment qualifies when the person has that role qualification, is available for the shift, and has at least 10 hours of rest; the rest threshold may be waived only by a signed exception naming that exact person and role. Assign roles by ascending role priority, then descending fit score, then ascending complete person ID, without assigning one person twice. Staffing requires coordinator approval.',
    instruction:'Build the maximum valid VS-610 assignments under rest, availability, role qualification, exception-scope, and one-role-per-person rules.',
    registerHeading:'Candidate assignment and fit register',auditHeading:'Qualification, availability, rest, and waiver audit',authorityHeading:'Staffing authorization record',
    candidates:[
      {id:'VOL-A/INTERPRETER',person:'VOL-A',role:'interpreter',priority:1,fit:95,qualified:true,available:true,rest:7,waiver:'VOL-A/radio'},
      {id:'VOL-B/INTERPRETER',person:'VOL-B',role:'interpreter',priority:1,fit:90,qualified:true,available:true,rest:12,waiver:'none'},
      {id:'VOL-B/RADIO',person:'VOL-B',role:'radio',priority:2,fit:99,qualified:true,available:true,rest:12,waiver:'none'},
      {id:'VOL-C/RADIO',person:'VOL-C',role:'radio',priority:2,fit:88,qualified:true,available:true,rest:10,waiver:'none'},
      {id:'VOL-D/RADIO',person:'VOL-D',role:'radio',priority:2,fit:97,qualified:false,available:true,rest:15,waiver:'none'},
    ],
    compare:(a,b)=>a.priority-b.priority||b.fit-a.fit||a.person.localeCompare(b.person),
    provisional:rows=>{const chosen=[],roles=new Set(),people=new Set();for(const r of rows){if(!roles.has(r.role)&&!people.has(r.person)){chosen.push(r);roles.add(r.role);people.add(r.person);}}return chosen;},
    eligible:r=>r.qualified&&r.available&&(r.rest>=10||r.waiver===`${r.person}/${r.role}`),
    select:rows=>{const chosen=[],roles=new Set(),people=new Set();for(const r of rows){if(!roles.has(r.role)&&!people.has(r.person)){chosen.push(r);roles.add(r.role);people.add(r.person);}}return chosen;},
    format:rows=>rows.length?rows.map(r=>`${r.role}=${r.person}`).join('; '):'none',measure:rows=>String(rows.length),
    registerLine:r=>`${r.id}: priority ${r.priority}; fit score ${r.fit}.`,
    auditLine:r=>`${r.id}: person=${r.person}; role=${r.role}; qualified=${r.qualified}; available=${r.available}; rest=${r.rest} hours; exact person-role waiver=${r.waiver}.`,
    exceptionText:'The recorded waiver names VOL-A/radio, so it cannot waive the rest requirement for VOL-A/interpreter. A waiver never creates a missing role qualification or authorizes assigning VOL-B twice.',
    authorityText:'Coordinator approval is recorded for VS-610.',authorized:true,
    selectionFormat:'Serialize each assignment with its role label in lowercase and the complete person ID exactly as listed, joined by exactly semicolon and one space in ascending role priority. Format: interpreter=<PERSON_ID>; radio=<PERSON_ID>. Return the literal lowercase string none if no assignments qualify.',measureFormat:'Number of distinct roles filled as digits only.',
    finalDecisions:['staff','hold','no_action'],approvedAction:'staff',noAction:'no_action',
  }),
  caseFrom({
    slug:'records-retention-disposition',domain:'records retention disposition',owner:'Public records counsel',requestId:'RT-227',
    requestText:'Prepare a disposition list using cutoff date 2026-10-01. The list is a proposed selection, not itself authority to destroy records.',
    ruleText:'Select a record for destruction only when its retention expiry is on or before the cutoff, no active legal hold applies, and custodian notice is complete. An active legal hold overrides age and notice. Sort selected records by expiry date ascending, then complete record ID. Counsel signature is required before destruction.',
    instruction:'Compute the records that qualify for the RT-227 proposed destruction list and state separately whether destruction is authorized.',
    registerHeading:'Retention-expiry register',auditHeading:'Legal-hold and notice audit',authorityHeading:'Counsel disposition authority record',
    candidates:[
      {id:'REC-A',expiry:'2019-04-01',hold:true,notice:true},{id:'REC-B',expiry:'2017-02-14',hold:true,notice:true},
      {id:'REC-C',expiry:'2028-01-01',hold:false,notice:true},{id:'REC-D',expiry:'2020-06-30',hold:false,notice:false},
      {id:'REC-E',expiry:'2026-10-01',hold:false,notice:false},
    ],
    compare:(a,b)=>a.expiry.localeCompare(b.expiry)||a.id.localeCompare(b.id),provisional:rows=>rows,
    eligible:r=>r.expiry<='2026-10-01'&&!r.hold&&r.notice,select:rows=>rows,format:joinIds,measure:rows=>String(rows.length),
    registerLine:r=>`${r.id}: retention expiry ${r.expiry}.`,auditLine:r=>`${r.id}: active legal hold=${r.hold}; custodian notice complete=${r.notice}.`,
    exceptionText:'A legal hold overrides an elapsed retention period. Missing notice prevents selection but does not alter the expiry date.',
    authorityText:'Counsel has not signed a destruction order for RT-227. Report the eligible proposed list, but do not represent destruction as authorized.',authorized:false,
    selectionFormat:'When records qualify, return complete record IDs joined by exactly semicolon and one space, sorted by expiry date ascending then ID. When none qualify, return the literal lowercase string none (not an empty string).',measureFormat:'Count of records in the proposed destruction list as digits only.',
    finalDecisions:['destroy','hold','no_action'],approvedAction:'destroy',noAction:'no_action',
  }),
];

export const worlds=authoredWorlds;
