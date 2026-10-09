import { caseFrom } from './semantic-iterate-reducers-v18-novel-data.mjs';

const yes = true, no = false;
const idOrder = (a, b) => a.id.localeCompare(b.id);
const world = (input) => {
  const { slug, domain, requestId, request, rule, instruction, heading, candidates, metric = 'metric',
  eligible, authorized, authorityText, exception, action, noAction = 'no_action', select = rows => rows.slice(0, 1),
  metricFormat, metricValue = rows => rows.length ? String(rows[0][metric]) : '0', registerLine, auditLine,
  direction = 'desc',
  tieRule = 'ties by ascending complete item ID' } = input;
  const requiredText = { slug, domain, requestId, request, rule, instruction, authorityText, exception,
    action, metricFormat, 'register heading':heading?.register, 'audit heading':heading?.audit,
    'authority heading':heading?.authority };
  for (const [name, value] of Object.entries(requiredText))
    if (typeof value !== 'string' || !value.trim()) throw new Error(`V22 authored ${name} must be a nonempty string`);
  if (!Array.isArray(candidates) || candidates.length < 2 || candidates.some(x =>
      typeof x.id !== 'string' || !x.id.trim() || typeof x.metric !== 'number' || !Number.isFinite(x.metric)))
    throw new Error(`V22 authored ${slug} candidates require IDs and finite numeric measures`);
  if (typeof eligible !== 'function' || typeof registerLine !== 'function' || typeof auditLine !== 'function')
    throw new Error(`V22 authored ${slug} is missing its eligibility or evidence renderer`);
  return ({
  slug, sourceGroup: `v22:${slug}:world`, domain, owner: `${domain} operations desk`, requestId, direction,
  requestText: request, ruleText: `${rule} Selection cardinality: select exactly one eligible item when at least one qualifies; otherwise select none. Rank eligible items by the stated measure, ${direction === 'asc' ? 'lowest' : 'greatest'} first; resolve equal measures by ascending complete item ID. Do not add tied items. Decision mapping: if no item qualifies, use ${noAction}; if a nonempty selection has the required recorded authority, use ${action}; otherwise use hold. Authority never changes the selected item or measure.`,
  instruction, registerHeading: heading.register, auditHeading: heading.audit, authorityHeading: heading.authority,
  candidates, compare: (a, b) => (direction === 'asc' ? a[metric] - b[metric] : b[metric] - a[metric]) || idOrder(a, b),
  provisional: rows => [...rows].sort((a,b)=>(direction === 'asc' ? a[metric]-b[metric] : b[metric]-a[metric])||idOrder(a,b)).slice(0, select(candidates).length),
  eligible, select, format: rows => rows.length ? rows.map(x => x.id).join('; ') : 'none',
  measure: metricValue, registerLine, auditLine, exceptionText: exception, authorityText, authorized,
  selectionFormat: `Exactly one complete item ID when any item qualifies, ranked by ${metric}, ${direction === 'asc' ? 'lowest' : 'greatest'} first; resolve equal measures by ascending complete item ID. Return none if no item qualifies. Do not include additional tied items.`,
  measureFormat: metricFormat, decisionFormat: `Use ${action} only for a nonempty eligible selection when the recorded authority is present. Use hold for a nonempty selection without authority. Use ${noAction} only when no item qualifies. Return one bare literal: ${[action,'hold',noAction].join(', ')}.`,
  finalDecisions: [action, 'hold', noAction], approvedAction: action, noAction,
  });
};

export const reducerDomainSpecs = [
  world({slug:'accessible-procurement',domain:'accessible procurement bid review',requestId:'PROC-810',request:'Select the lowest-cost bid that meets the accessibility and interoperability requirements for the public service portal.',rule:'A bid qualifies only if its accessibility audit passes, the listed identity protocol is compatible, and its total is within the stated 180000 credit cap. Select the lowest total.',instruction:'Choose a compliant public-service portal bid. A low price cannot compensate for a failed accessibility audit or incompatible identity protocol.',heading:{register:'Bid price register',audit:'Accessibility, interoperability, and cap audit',authority:'Procurement authorization'},candidates:[
    {id:'BID-A41',metric:164000,accessible:yes,protocol:yes,withinCap:yes},{id:'BID-A42',metric:151000,accessible:no,protocol:yes,withinCap:yes},{id:'BID-A43',metric:172000,accessible:yes,protocol:no,withinCap:yes},{id:'BID-A44',metric:176000,accessible:yes,protocol:yes,withinCap:yes}],eligible:x=>x.accessible&&x.protocol&&x.withinCap,authorized:yes,authorityText:'The purchasing officer approved a compliant award for PROC-810.',exception:'An accessibility exception is not authorized; all three stated bid requirements are mandatory.',action:'award',metricFormat:'Total bid credits as digits; 0 when none.',registerLine:x=>`${x.id}: total price ${x.metric} credits.`,auditLine:x=>`${x.id}: accessibility audit passes=${x.accessible}; identity protocol interoperable=${x.protocol}; within 180000-credit cap=${x.withinCap}.`,direction:'asc'}),
  world({slug:'vendor-api-migration',domain:'vendor API migration readiness',requestId:'API-821',request:'Choose the connector release for the next deployment window based on contract compatibility and rollback readiness.',rule:'A release qualifies only when contract tests pass, rollback is verified, and the vendor support window covers deployment. Select greatest verified migration coverage.',instruction:'Select a migration-ready connector release; do not treat a high coverage score as proof of rollback or support.',heading:{register:'Connector coverage register',audit:'Contract, rollback, and support checks',authority:'Deployment window approval'},candidates:[
    {id:'CONN-R6',metric:88,contracts:yes,rollback:no,support:yes},{id:'CONN-R7',metric:84,contracts:yes,rollback:yes,support:yes},{id:'CONN-R8',metric:93,contracts:no,rollback:yes,support:yes},{id:'CONN-R9',metric:84,contracts:yes,rollback:yes,support:yes}],eligible:x=>x.contracts&&x.rollback&&x.support,authorized:no,authorityText:'The deployment lead has not approved the next window; approval is pending.',exception:'Coverage never waives failed contract tests, absent rollback verification, or an expired support window.',action:'deploy',metricFormat:'Verified coverage points as digits; 0 when none.',registerLine:x=>`${x.id}: verified endpoint coverage ${x.metric} points.`,auditLine:x=>`${x.id}: contract tests pass=${x.contracts}; rollback verified=${x.rollback}; vendor support covers window=${x.support}.`}),
  world({slug:'research-cohort-classification',domain:'research cohort inclusion review',requestId:'RES-832',request:'Identify the highest-enrollment cohort that meets the prespecified population and measurement protocol.',rule:'A cohort qualifies only when the population matches, the measurement protocol is preregistered, and missing outcome data is at most 5 percent. Select the greatest enrolled sample.',instruction:'Determine the qualifying research cohort from the stated inclusion protocol; enrollment size alone is not sufficient.',heading:{register:'Enrollment register',audit:'Population, preregistration, and missing-data audit',authority:'Research oversight record'},candidates:[
    {id:'COHORT-C1',metric:240,population:yes,preregistered:yes,missingAtMost5:yes},{id:'COHORT-C2',metric:310,population:yes,preregistered:no,missingAtMost5:yes},{id:'COHORT-C3',metric:225,population:no,preregistered:yes,missingAtMost5:yes},{id:'COHORT-C4',metric:240,population:yes,preregistered:yes,missingAtMost5:yes}],eligible:x=>x.population&&x.preregistered&&x.missingAtMost5,authorized:yes,authorityText:'The research oversight committee recorded inclusion approval for RES-832.',exception:'A larger cohort cannot replace a failed population criterion, missing preregistration, or excess missing outcome data.',action:'include',metricFormat:'Enrolled participant count as digits; 0 when none.',registerLine:x=>`${x.id}: enrolled sample ${x.metric} participants.`,auditLine:x=>`${x.id}: population matches protocol=${x.population}; protocol preregistered=${x.preregistered}; missing outcomes at most 5 percent=${x.missingAtMost5}.`}),
  world({slug:'maintenance-work-order-priority',domain:'municipal maintenance work-order triage',requestId:'MNT-843',request:'Schedule one work order with the greatest verified safety-risk reduction for this shift.',rule:'A work order qualifies when the hazard is verified, the crew has the listed certification, and required parts are on site. Select greatest risk reduction points.',instruction:'Prioritize work orders using the verified hazard and readiness checks. An urgent score does not make an unready crew safe to dispatch.',heading:{register:'Risk reduction register',audit:'Hazard and dispatch readiness audit',authority:'Shift dispatch authorization'},candidates:[
    {id:'WO-31',metric:76,hazard:yes,certified:yes,parts:yes},{id:'WO-32',metric:91,hazard:yes,certified:no,parts:yes},{id:'WO-33',metric:76,hazard:yes,certified:yes,parts:yes},{id:'WO-34',metric:89,hazard:no,certified:yes,parts:yes}],eligible:x=>x.hazard&&x.certified&&x.parts,authorized:no,authorityText:'The shift supervisor has not yet released a crew for dispatch.',exception:'Do not dispatch on an unverified hazard or without both crew certification and parts.',action:'dispatch',metricFormat:'Risk reduction points as digits; 0 when none.',registerLine:x=>`${x.id}: estimated risk reduction ${x.metric} points.`,auditLine:x=>`${x.id}: hazard verified=${x.hazard}; assigned crew certified=${x.certified}; required parts on site=${x.parts}.`}),
  world({slug:'document-claim-scope',domain:'public report claim substantiation',requestId:'DOC-854',request:'Select the report claims supported by an in-scope source dated within the reporting period.',rule:'A claim qualifies only when its cited source directly supports the claim and the source date falls in the reporting period. Select the supported claim with the highest evidence grade.',instruction:'Review claims against their actual cited source and reporting window. A citation existing does not establish that it supports the claim.',heading:{register:'Evidence-grade register',audit:'Citation support and date-scope audit',authority:'Publication review'},candidates:[
    {id:'CLAIM-D1',metric:3,directSupport:yes,inPeriod:yes},{id:'CLAIM-D2',metric:5,directSupport:no,inPeriod:yes},{id:'CLAIM-D3',metric:3,directSupport:yes,inPeriod:yes},{id:'CLAIM-D4',metric:4,directSupport:yes,inPeriod:no}],eligible:x=>x.directSupport&&x.inPeriod,authorized:yes,authorityText:'The editor signed the report publication review for DOC-854.',exception:'Evidence grades use the stated ordinal scale from 1 (limited) to 5 (strong); a high grade cannot repair indirect support or a source outside the reporting period.',action:'publish',metricFormat:'Evidence grade on the 1–5 scale as digits; 0 when none.',registerLine:x=>`${x.id}: evidence grade ${x.metric} on the 1–5 scale.`,auditLine:x=>`${x.id}: cited source directly supports the claim=${x.directSupport}; source date is in reporting period=${x.inPeriod}.`}),
  world({slug:'community-service-eligibility',domain:'community service enrollment',requestId:'SVC-865',request:'Select the highest-need applicant who meets residency and program eligibility requirements.',rule:'An applicant qualifies when residency is verified, the documented need meets the program threshold, and duplicate enrollment is absent. Select greatest need score.',instruction:'Identify the eligible applicant with greatest documented need. Intake priority does not override residency, threshold, or duplicate checks.',heading:{register:'Need-score register',audit:'Residency, threshold, and duplicate enrollment audit',authority:'Enrollment capacity release'},candidates:[
    {id:'APP-E7',metric:82,resident:yes,threshold:yes,notDuplicate:yes},{id:'APP-E8',metric:96,resident:yes,threshold:yes,notDuplicate:no},{id:'APP-E9',metric:82,resident:yes,threshold:yes,notDuplicate:yes},{id:'APP-E10',metric:91,resident:no,threshold:yes,notDuplicate:yes}],eligible:x=>x.resident&&x.threshold&&x.notDuplicate,authorized:yes,authorityText:'The program manager released one enrollment place for SVC-865.',exception:'Do not enroll an applicant with unverified residency, insufficient documented need, or duplicate active enrollment.',action:'enroll',metricFormat:'Need-score points as digits; 0 when none.',registerLine:x=>`${x.id}: documented need score ${x.metric}.`,auditLine:x=>`${x.id}: residency verified=${x.resident}; need meets threshold=${x.threshold}; no duplicate enrollment=${x.notDuplicate}.`}),
  world({slug:'records-retention-disposition',domain:'records retention disposition',requestId:'REC-876',request:'Identify the record series eligible for scheduled destruction under the approved retention schedule.',rule:'A series qualifies only when its retention period has expired and no legal hold or audit freeze applies. Select the greatest number of eligible records.',instruction:'Determine what may be destroyed under the schedule. Expired retention alone is not enough when a hold or freeze remains active.',heading:{register:'Record count and schedule register',audit:'Expiry, legal hold, and audit freeze review',authority:'Records officer disposition release'},candidates:[
    {id:'SERIES-F3',metric:460,expired:yes,noLegalHold:yes,noAuditFreeze:yes},{id:'SERIES-F4',metric:520,expired:yes,noLegalHold:no,noAuditFreeze:yes},{id:'SERIES-F5',metric:460,expired:yes,noLegalHold:yes,noAuditFreeze:yes},{id:'SERIES-F6',metric:610,expired:no,noLegalHold:yes,noAuditFreeze:yes}],eligible:x=>x.expired&&x.noLegalHold&&x.noAuditFreeze,authorized:no,authorityText:'The records officer has not signed the scheduled destruction release.',exception:'Legal holds and audit freezes block destruction even after the retention period expires.',action:'destroy',metricFormat:'Eligible record count as digits; 0 when none.',registerLine:x=>`${x.id}: records in series ${x.metric}.`,auditLine:x=>`${x.id}: retention period expired=${x.expired}; no legal hold=${x.noLegalHold}; no audit freeze=${x.noAuditFreeze}.`}),
  world({slug:'incident-mitigation-selection',domain:'service incident mitigation planning',requestId:'INC-887',request:'Choose the mitigation that reduces the most user-impact minutes while meeting safety and rollback requirements.',rule:'A mitigation qualifies only when its safety review passes, rollback is tested, and affected data is backed up. Select greatest verified user-impact minutes avoided.',instruction:'Choose a safe, reversible mitigation from the tested options. Estimated benefit cannot waive safety, rollback, or backup requirements.',heading:{register:'Impact reduction register',audit:'Safety, rollback, and backup review',authority:'Incident commander action authority'},candidates:[
    {id:'MIT-G2',metric:740,safe:yes,rollback:yes,backup:yes},{id:'MIT-G3',metric:920,safe:yes,rollback:no,backup:yes},{id:'MIT-G4',metric:740,safe:yes,rollback:yes,backup:yes},{id:'MIT-G5',metric:880,safe:no,rollback:yes,backup:yes}],eligible:x=>x.safe&&x.rollback&&x.backup,authorized:yes,authorityText:'The incident commander approved one mitigation action for INC-887.',exception:'A mitigation with failed safety review, untested rollback, or no data backup is ineligible regardless of estimated benefit.',action:'apply',metricFormat:'User-impact minutes avoided as digits; 0 when none.',registerLine:x=>`${x.id}: verified user-impact reduction ${x.metric} minutes.`,auditLine:x=>`${x.id}: safety review passes=${x.safe}; rollback tested=${x.rollback}; affected data backed up=${x.backup}.`}),
];

const defs = reducerDomainSpecs;
export const worlds = defs.flatMap((base, domainIndex) => [0,1].map((variantIndex) => {
  // Each scenario gets new metrics, a changed leading candidate condition, request ID, and group.
  const candidates = base.candidates.map((c,i) => ({...c,
    id: `${c.id}-${variantIndex ? 'B' : 'A'}${domainIndex+1}${i+1}`,
    metric: c.metric + (variantIndex ? (domainIndex === 4 ? (i % 2 === 0 ? 1 : -1) : ((i % 2 === 0 ? 11 : -13) * (domainIndex + 1))) : 0)}));
  if (variantIndex) {
    const firstEligible = candidates.find(base.eligible);
    if (!firstEligible) throw new Error(`${base.slug}: base facts have no eligible candidate`);
    const condition = Object.keys(firstEligible).find(key => typeof firstEligible[key] === 'boolean');
    if (!condition) throw new Error(`${base.slug}: no explicit condition to counterfactually revise`);
    firstEligible[condition] = !firstEligible[condition];
    if (!candidates.some(base.eligible)) throw new Error(`${base.slug}: variant B unexpectedly has no eligible item`);
  }
  const suffix = variantIndex ? 'B' : 'A';
  const requestId = base.requestId.replace(/\d+$/, String(Number(base.requestId.match(/\d+$/)[0]) + variantIndex*100));
  const eligible = base.eligible;
  const worldSpec = {...base, slug:`${base.slug}-${suffix.toLowerCase()}`, sourceGroup:`v22:${base.slug}:world`, requestId,
    candidates, eligible, authorized: variantIndex ? !base.authorized : base.authorized,
    authorityText: variantIndex
      ? (base.authorized ? `The approving authority was withdrawn before disposition; no release is present for ${requestId}.` : `The authorized signer approved this request (${requestId}).`)
      : base.authorityText,
    instruction: `${base.instruction} Request ${requestId}.`};
  worldSpec.authorityText = `AUTHORITY STATUS: ${worldSpec.authorized ? 'approved' : 'not approved'}. ${worldSpec.authorityText}`;
  // Counterfactual B revises candidate conditions and measures, then flips authority status.
  const made = caseFrom(worldSpec);
  made.group = worldSpec.sourceGroup;
  made.authorized = worldSpec.authorized;
  made.approvedAction = worldSpec.approvedAction;
  made.noAction = worldSpec.noAction;
  for (const key of ['eligible','compare','select','format','measure'])
    Object.defineProperty(made, key, { value: worldSpec[key], enumerable: false });
  made.decision_rule = worldSpec.ruleText;
  made.instruction = worldSpec.instruction;
  made.source_summary.scenario = suffix;
  made.source_summary.parent_group = null;
  made.source_summary.domain_index = domainIndex;
  return made;
}));

export const sourceDomains = defs.map(({slug,domain}) => ({slug,domain,variants:2}));
