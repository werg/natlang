#!/usr/bin/env node
/** Structured fictional research cases with host-derived decisions and pivotal, source-grounded evidence. */
import { createHash } from 'node:crypto';
import { mkdirSync, openSync, writeFileSync, closeSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const metric={schema:'natlang.skill-research/1',kind:'research-classification'};
export const variants=['empty','helpful-body-poor-description','description-only'];
export const labels={
  'incident-triage':['critical','urgent','routine'],
  'procurement-exceptions':['eligible','ineligible'],
  'study-effect':['causal-evidence','association-only','inconclusive'],
  'policy-scope':['applies','does-not-apply'],
  'software-root-cause':['release-regression','capacity-exhaustion','dependency-failure','data-corruption'],
  'records-retention':['retain','eligible-for-disposal','legal-hold'],
};
const sha=x=>createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const norm=x=>JSON.stringify(x,Object.keys(x||{}).sort());
const assert=(ok,message)=>{if(!ok)throw Error(message)};

const ruleTexts={
 'incident-triage':'Rule R-17: A verified hazardous release is Critical only when the emergency band, occupied zone, and staffed-shift conditions all hold. Verified exposure without that full conjunction is Urgent. A verified no-release event without symptoms is Routine.',
 'procurement-exceptions':'Rule P-22: A bid is eligible when one complete route qualifies: local-standard (complete, local, under cap), safety-waiver (critical item, signed waiver, no equivalent), or warehouse-depot (warehouse class, active depot, under depot cap). Ineligible requires all three routes to be verified and closed.',
 'study-effect':'Rule S-9: Causal evidence requires valid intervention assignment, valid outcome measurement, and an interval supporting the prespecified direction. Records without assigned intervention support association only when selection and outcome records are valid. Conflicting methods or an interval spanning decision bounds are inconclusive.',
 'policy-scope':'Rule J-31: The protected-record policy applies to protected records processed in Northmere or covered by an effective Northmere-key exception. It does not apply only after the record class, processing location, and exception register have all been checked.',
 'software-root-cause':'Rule O-12: Assign a root cause only when the linked event chain includes an initiating signal, a discriminating intervention or recovery, and independent evidence ruling out the competing layer. Correlation without a distinguishing test is unresolved.',
 'records-retention':'Rule A-6: Active legal holds and open audits or disputes block disposal. Otherwise retain until the record-class schedule expires; disposal requires an expired period and completed restriction checks.',
};

const record=(id,title,statement,context,fields)=>({id,title,text:`${statement} ${context}`,fields});
function factsFor(family,mode){
 const m={};
 const set=(facts,records,pivots)=>({facts,records,pivots});
 if(family==='incident-triage'){
  if(['critical-release','critical-vessel'].includes(mode)){
   const vessel=mode==='critical-vessel'?'Vessel 7':'Bay 4',pressure=mode==='critical-vessel'?15:14;
   const facts={ruleActive:true,authorityClear:true,confirmed:true,emergency:true,occupied:true,staffed:true,symptoms:false};
   return set(facts,[
    record('sensor-event','Calibrated sensor event',`Detector ${vessel} confirmed a solvent release at ${pressure} units; emergency band begins at 12 units.`,`Calibration certificate C-814 applies to event E-${mode}; the measurement device was checked at the start and end of the shift.`,['confirmed','emergency']),
    record('zone-register','Occupancy and zone register',`${vessel} was occupied during the event interval.`,`Access badge record A-91 links the occupants to event E-${mode}; no evacuation had started at the first measurement.`,['occupied']),
    record('shift-roster','Control-room shift roster',`The staffed response window covered the event timestamp for ${vessel}.`,`Roster revision 6 was signed before the event. Its start and end times use the facility local clock recorded in event E-${mode}.`,['staffed']),
    record('symptom-log','Worker exposure report','No worker symptom was recorded during the verified sampling window.',`The clinic log and shift supervisor report refer to the same incident identifier E-${mode}.`,['symptoms']),
   ],[{record:'sensor-event',text:`Detector ${vessel} confirmed a solvent release at ${pressure} units; emergency band begins at 12 units.`,changes:{confirmed:false}},{record:'sensor-event',text:`Detector ${vessel} confirmed a solvent release at ${pressure} units; emergency band begins at 12 units.`,changes:{emergency:false}},{record:'zone-register',text:`${vessel} was occupied during the event interval.`,changes:{occupied:false}},{record:'shift-roster',text:`The staffed response window covered the event timestamp for ${vessel}.`,changes:{staffed:false}}]);
  }
  if(['urgent-exposure','urgent-action'].includes(mode)){
   const symptom=mode==='urgent-exposure',facts={ruleActive:true,authorityClear:true,confirmed:true,emergency:false,actionBand:true,occupied:!symptom,staffed:true,symptoms:symptom,exposureVerified:true};
   const releaseText=symptom?'Exposure screen verified vapor contact below the emergency band.':'Detector D-22 confirmed 8 units, above action band 5 and below emergency band 12.';
   return set(facts,[record('reading','Incident measurement',releaseText,`Reading is linked to case E-${mode}; sensor calibration and timestamp trace are attached. The nearby alarm threshold belongs to this same pressure unit.`,['confirmed','actionBand','emergency']),record('people','Occupancy and exposure record',symptom?'A worker symptom report is linked to the verified vapor-contact interval.':'The bay was occupied during the action-band reading.',`Badge ledger and exposure interview share case key E-${mode}; the interview was recorded after the reading and before mitigation.`,symptom?['symptoms','exposureVerified']:['occupied']),record('response-window','Shift response log','The control room acknowledged the event during an active staffed response window.',`Dispatch timestamp and roster revision 6 were reconciled against the same local-clock source.`,['staffed'])],[{record:'reading',text:releaseText,changes:symptom?{exposureVerified:false}:{actionBand:false}},{record:'people',text:symptom?'A worker symptom report is linked to the verified vapor-contact interval.':'The bay was occupied during the action-band reading.',changes:symptom?{symptoms:false}:{occupied:false} }]);
  }
  if(['routine-contained','routine-calibration'].includes(mode)){
   const calibration=mode==='routine-calibration',facts={ruleActive:true,authorityClear:true,confirmed:false,emergency:false,actionBand:false,occupied:false,staffed:false,symptoms:false,exposureVerified:true,calibration};
   const sensor=calibration?'Independent hand-held detector measured zero release during calibration window K-41.':'Independent detector measured zero solvent release in gallery G-3.';
   const symptoms='Clinic and shift logs record no exposure symptoms for the linked event interval.';
   return set(facts,[record('independent-reading','Independent detector record',sensor,`Instrument serial H-18 was independently checked against the daily standard. Event key E-${mode} appears on both the raw reading and witness sheet.`,['confirmed','emergency','actionBand']),record('exposure-check','Exposure review',symptoms,`The clinic and supervisor sources were reconciled by employee badge ID and event time; the record contains no pending interview.`,['symptoms','exposureVerified']),record('process-state','Process and isolation log','The process line remained stable and no release isolation was required.',`The operator log spans the sensor interval and links the event to the same equipment tag used by the detector record.`,['calibration'])],[{record:'independent-reading',text:sensor,changes:{confirmed:true,actionBand:true}},{record:'exposure-check',text:symptoms,changes:{symptoms:true}}]);
  }
  const facts={ruleActive:true,authorityClear:true,confirmed:true,emergency:true,occupied:true,staffed:true,symptoms:false,scheduleConflict:true};
  return set(facts,[record('schedule-a','Staffing schedule revision 18','Schedule 18 ends staffed coverage at 19:00 local time.',`Signed schedule 18 lists facility PS-4 and applies on the incident date; its approval ID is APR-188.`,['scheduleA']),record('schedule-b','Staffing schedule revision 19','Schedule 19 ends staffed coverage at 20:00 local time.',`Schedule 19 is also marked effective for PS-4. The approval ledger points to a missing signature attachment.`,['scheduleB']),record('precedence','Operations approval register','The register contains no precedence or correction entry between schedules 18 and 19.',`Records office lookup used facility PS-4 and the event date. Both schedule revisions remain current in the register.`,['precedence'])],[{record:'schedule-a',text:'Schedule 18 ends staffed coverage at 19:00 local time.',changes:{scheduleConflict:false,staffed:false}},{record:'schedule-b',text:'Schedule 19 ends staffed coverage at 20:00 local time.',changes:{scheduleConflict:false,staffed:true}},{record:'precedence',text:'The register contains no precedence or correction entry between schedules 18 and 19.',changes:{scheduleConflict:false,staffed:true}}]);
 }
 if(family==='procurement-exceptions'){
  if(mode==='local-approved'||mode==='local-near-cap'){
   const amount=mode==='local-approved'?18400:19950, facts={ruleActive:true,authorityClear:true,standardLocal:true,standardComplete:true,standardUnderCap:true,safetyRouteClosed:true,warehouseRouteClosed:true,routeReviewComplete:true};
   return set(facts,[record('supplier','Vendor registration and intake',`Vendor N-14 is registered in Northmere and the application is complete.`,`Registry key V-14 matches the bid cover sheet; the signed safety declaration is included in packet B-${amount}.`,['standardLocal','standardComplete']),record('bid-total','Bid valuation worksheet',`The accepted bid total is ${amount} credits, below the current 20,000-credit standard cap.`,`The worksheet reconciles line totals, tax treatment, and submitted revisions against the same bid identifier.`,['standardUnderCap']),record('route-screen','Alternative route review','The safety-waiver and warehouse-depot routes were checked and do not apply to this item.',`Reviewer R-5 records the item as ordinary stock; the route screen is attached to bid packet B-${amount}.`,['safetyRouteClosed','warehouseRouteClosed','routeReviewComplete'])],[{record:'supplier',text:`Vendor N-14 is registered in Northmere and the application is complete.`,changes:{standardLocal:false}},{record:'supplier',text:`Vendor N-14 is registered in Northmere and the application is complete.`,changes:{standardComplete:false}},{record:'bid-total',text:`The accepted bid total is ${amount} credits, below the current 20,000-credit standard cap.`,changes:{standardUnderCap:false}}]);
  }
  if(mode==='safety-award'){
   const facts={ruleActive:true,authorityClear:true,standardLocal:false,standardComplete:true,standardUnderCap:false,safetyCritical:true,waiverSigned:true,noEquivalent:true,warehouseRouteClosed:true,routeReviewComplete:true};
   return set(facts,[record('technical-review','Safety item assessment','The requested valve is classified safety-critical and the equivalence review found no substitute.',`Engineering review ER-99 identifies valve VX-7; it compares approved substitutes by pressure rating and certification, not vendor preference.`,['safetyCritical','noEquivalent']),record('waiver','Signed sole-source waiver','The named valve waiver is signed by the authorized safety officer.',`Waiver W-204 names VX-7 and references engineering review ER-99. Its signature predates the purchase request.`,['waiverSigned']),record('route-screen','Alternative route review','The local-standard route fails its supplier/price test; the warehouse-depot route is not applicable.',`The bid has no Northmere registration and is not a warehouse delivery; both findings are linked to request Q-204.`,['standardLocal','standardUnderCap','warehouseRouteClosed','routeReviewComplete'])],[{record:'technical-review',text:'The requested valve is classified safety-critical and the equivalence review found no substitute.',changes:{noEquivalent:false}},{record:'waiver',text:'The named valve waiver is signed by the authorized safety officer.',changes:{waiverSigned:false}},{record:'technical-review',text:'The requested valve is classified safety-critical and the equivalence review found no substitute.',changes:{safetyCritical:false}}]);
  }
  if(mode==='ineligible-closed-routes'){
   const facts={ruleActive:true,authorityClear:true,standardLocal:false,standardComplete:false,standardUnderCap:false,safetyCritical:false,waiverSigned:false,noEquivalent:false,warehouseClass:false,depotActive:false,warehouseUnderCap:false,routeReviewComplete:true};
   return set(facts,[record('standard-route','Standard route eligibility screen','Standard route is blocked: vendor is nonlocal, the file is incomplete, and total exceeds its cap.',`Three separate checklist fields are signed by intake; the block applies to bid Q-511, not the umbrella project.`,['standardLocal','standardComplete','standardUnderCap']),record('safety-route','Safety exception eligibility screen','Safety route is blocked: item is not critical and no signed waiver or no-equivalent finding is present.',`Safety office checked the item ID against its certification and waiver registers for request Q-511.`,['safetyCritical','waiverSigned','noEquivalent']),record('warehouse-route','Warehouse route eligibility screen','Warehouse route is blocked: this is not a warehouse contract and no qualifying depot is active.',`Contract-type code and depot registry were checked on the intake date; no warehouse addendum is referenced.`,['warehouseClass','depotActive','warehouseUnderCap'])],[{record:'standard-route',text:'Standard route is blocked: vendor is nonlocal, the file is incomplete, and total exceeds its cap.',changes:{standardLocal:undefined}},{record:'safety-route',text:'Safety route is blocked: item is not critical and no signed waiver or no-equivalent finding is present.',changes:{safetyCritical:undefined}},{record:'warehouse-route',text:'Warehouse route is blocked: this is not a warehouse contract and no qualifying depot is active.',changes:{warehouseClass:undefined}}]);
  }
  const facts={ruleActive:true,authorityClear:true,amendmentConflict:true,thresholdA:20000,thresholdB:15000,precedence:false,amount:18000,localAddress:'Northmere',depotAddress:'Northmere'};
  return set(facts,[record('amendment-a','Supplier amendment 8','Signed amendment 8 uses the registered office as the controlling supplier address.',`Amendment 8 is shown effective on 1 July in register AR-8 and covers standard and warehouse bids.`,['thresholdA','localAddress']),record('amendment-b','Supplier amendment 9','Signed amendment 9 uses the principal operating depot instead of registered office.',`Amendment 9 is also listed effective on 1 July; neither record references a superseding signature.`,['thresholdB','depotAddress']),record('amendment-register','Precedence register','No precedence entry resolves amendments 8 and 9 for request Q-701.',`The clerk searched the signed amendment register by request type, date, and both supplier locations.`,['precedence'])],[{record:'amendment-a',text:'Signed amendment 8 uses the registered office as the controlling supplier address.',changes:{amendmentConflict:false,precedence:'A'}},{record:'amendment-b',text:'Signed amendment 9 uses the principal operating depot instead of registered office.',changes:{amendmentConflict:false,precedence:'B'}},{record:'amendment-register',text:'No precedence entry resolves amendments 8 and 9 for request Q-701.',changes:{amendmentConflict:false,precedence:'A'}}]);
 }
 if(family==='study-effect'){
  if(mode==='randomized-effect'||mode==='randomized-replication'){
   const facts={ruleActive:true,authorityClear:true,randomAssignment:true,allocationValid:true,outcomeValid:true,intervalSupports:true,selectionConfound:false};
   return set(facts,[record('allocation','Allocation audit','The concealed random sequence assigned each enrolled site to intervention or comparison.',`Sequence hash and enrollment timestamps reconcile for protocol ST-${mode}; no site chose its own arm.`,['randomAssignment','allocationValid']),record('endpoint','Primary endpoint report','The prespecified primary endpoint was independently measured and its interval excludes the null boundary.',`Outcome definition and measurement window were frozen before enrollment; endpoint review is blinded to allocation labels.`,['outcomeValid','intervalSupports']),record('selection','Selection-bias assessment','Enrollment and follow-up checks found no material differential loss between arms.',`Participant flow is reconciled to the site ledger; all exclusions have dated reasons and arm codes.`,['selectionConfound'])],[{record:'allocation',text:'The concealed random sequence assigned each enrolled site to intervention or comparison.',changes:{randomAssignment:false}},{record:'allocation',text:'The concealed random sequence assigned each enrolled site to intervention or comparison.',changes:{allocationValid:false}},{record:'endpoint',text:'The prespecified primary endpoint was independently measured and its interval excludes the null boundary.',changes:{intervalSupports:false}}]);
  }
  if(mode==='observational-selection'||mode==='observational-registry'){
   const facts={ruleActive:true,authorityClear:true,interventionAssigned:false,selfSelection:true,outcomeValid:true,associationObserved:true,randomAllocation:false};
   return set(facts,[record('assignment','Intervention assignment log','The investigators assigned no intervention; participants selected their own service pathway.',`The study began after the care choices were recorded. The assignment log is linked to cohort registry C-81.`,['interventionAssigned','selfSelection']),record('outcome','Outcome measurement report','The outcome was measured from records created before the analysis began.',`Endpoint extraction was performed under the frozen study plan, with clinic and date identifiers retained for audit.`,['outcomeValid','associationObserved']),record('protocol','Protocol history','No random allocation or investigator-controlled treatment assignment is specified in the registered protocol.',`Protocol versions 1 and 2 were compared; version 2 adds a follow-up schedule but no assignment procedure.`,['randomAllocation'])],[{record:'assignment',text:'The investigators assigned no intervention; participants selected their own service pathway.',changes:{interventionAssigned:true}},{record:'assignment',text:'The investigators assigned no intervention; participants selected their own service pathway.',changes:{selfSelection:false}},{record:'outcome',text:'The outcome was measured from records created before the analysis began.',changes:{outcomeValid:false}}]);
  }
  if(mode==='null-interval'||mode==='posthoc-outcome'){
   const facts={ruleActive:true,authorityClear:true,randomAssignment:false,interventionAssigned:false,selfSelection:true,associationObserved:true,allocationValid:true,outcomeValid:true,intervalSupports:false,intervalSpansDecisionBounds:true,posthocSelected:mode==='posthoc-outcome'};
   const endpoint=mode==='posthoc-outcome'?'The prespecified primary interval spans both meaningful benefit and harm; the favorable subgroup was selected after inspection.':'The prespecified effect interval crosses zero and both registered decision bounds.';
   return set(facts,[record('allocation','Allocation record','The randomization ledger matches the signed enrollment sequence.',`Auditor compared the sequence and all enrollment rows; no post-allocation switch is recorded.`,['randomAssignment','allocationValid']),record('statistics','Statistical report',endpoint,`Analysis plan AP-52 predates data lock. The post-hoc subgroup has no confirmatory sample and does not replace the registered endpoint.`,['intervalSupports','intervalSpansDecisionBounds','posthocSelected']),record('endpoint','Measurement validation','The primary outcome definition and measurement protocol match the registered study plan.',`Independent review found no change to endpoint code, window, or instrument version after enrollment.`,['outcomeValid'])],[{record:'statistics',text:endpoint,changes:{intervalSpansDecisionBounds:false,intervalSupports:true}},{record:'endpoint',text:'The primary outcome definition and measurement protocol match the registered study plan.',changes:{outcomeValid:false}}]);
  }
  const facts={ruleActive:true,authorityClear:true,allocationConflict:true,protocolRandom:true,ledgerClinicianChoice:true,correctionExists:false,randomAssignment:true,allocationValid:true,outcomeValid:true,intervalSupports:true,interventionAssigned:false,selfSelection:true};
  return set(facts,[record('protocol','Registered protocol','Protocol version 3 specifies randomized site assignment.',`Version 3 was signed before enrollment; its sequence generator and allocation concealment are described in appendix B.`,['protocolRandom']),record('ledger','Enrollment ledger','The signed enrollment ledger records clinician-selected assignments.',`Ledger rows link clinician, site, and participant IDs. The discrepancy begins at the first enrollment and continues through closeout.`,['ledgerClinicianChoice']),record('correction-register','Correction register','No signed correction reconciles the protocol and enrollment ledger.',`The records office searched amendments, errata, and dated audit notes; no controlling correction is present.`,['correctionExists'])],[{record:'protocol',text:'Protocol version 3 specifies randomized site assignment.',changes:{allocationConflict:false,ledgerClinicianChoice:false}},{record:'ledger',text:'The signed enrollment ledger records clinician-selected assignments.',changes:{allocationConflict:false,protocolRandom:false,randomAssignment:false}},{record:'correction-register',text:'No signed correction reconciles the protocol and enrollment ledger.',changes:{allocationConflict:false,correctionExists:true}}]);
 }
 if(family==='policy-scope'){
  if(['protected-onsite','superseding-order'].includes(mode)){
   const facts={ruleActive:true,authorityClear:true,protectedRecord:true,insideNorthmere:true,remoteException:false,keyNorthmere:false,operatorRegisteredNorthmere:false};
   return set(facts,[record('classification','Record classification','Accession register classifies record class PR-2 as protected.',`Signed accession ID AC-${mode} matches the packet and retention category; ordinary mirror copies remain separately indexed.`,['protectedRecord']),record('location','Processing location audit','The processing server was physically located in Northmere at the recorded processing time.',`Facility coordinate and asset ID agree across network inventory and signed site survey for case J-${mode}.`,['insideNorthmere']),record('operator','Operator registry','The processor is registered in Eastmere; operator registration is not the territorial processing test.',`Registry extract is current for the event date and is linked by operator ID, not facility alias.`,['operatorRegisteredNorthmere'])],[{record:'classification',text:'Accession register classifies record class PR-2 as protected.',changes:{protectedRecord:false}},{record:'location',text:'The processing server was physically located in Northmere at the recorded processing time.',changes:{insideNorthmere:false}}]);
  }
  if(['ordinary-outside','remote-key-unapproved'].includes(mode)){
   const ordinary=mode==='ordinary-outside',facts={ruleActive:true,authorityClear:true,protectedRecord:!ordinary,insideNorthmere:ordinary,remoteException:!ordinary,keyNorthmere:!ordinary,exceptionEffective:mode==='remote-key-unapproved'?false:true,scopeChecklistComplete:true};
   const first=ordinary?'Classification schedule marks the archive record ordinary, outside protected class PR-2.':'The protected record is processed in Southmere outside Northmere territory.';
   const second=ordinary?'Server location is Northmere, but ordinary records are outside the rule’s subject scope.':'The remote-key exception is a draft and is not effective on the processing date.';
   return set(facts,[record('classification','Classification and scope record',first,`Signed record class and case ID were verified against accession ledger AC-${mode}; nearby protected series use a different accession code.`,['protectedRecord']),record('location','Territorial location audit',second,`Location comes from the physical asset register, not operator registration. The exception status comes from the signed instrument index for this date.`,['insideNorthmere','exceptionEffective']),record('key','Key-control register',ordinary?'No remote-key exception is invoked for ordinary records.':'Key audit places control in Northmere, but the associated exception remains unapproved.',`Key ID and processing record are linked. The index retains draft status separately from operative signed instruments.`,['remoteException','keyNorthmere','scopeChecklistComplete'])],[{record:'classification',text:first,changes:{protectedRecord:!facts.protectedRecord}},{record:'location',text:second,changes:{insideNorthmere:true,exceptionEffective:true}},{record:'key',text:ordinary?'No remote-key exception is invoked for ordinary records.':'Key audit places control in Northmere, but the associated exception remains unapproved.',changes:{scopeChecklistComplete:false}}]);
  }
  const facts={ruleActive:true,authorityClear:true,boundaryConflict:true,insideNorthmere:true,scopeChecklistComplete:true,mapA:'Northmere',mapB:'Eastmere',precedence:false,protectedRecord:true};
  return set(facts,[record('map-a','Northmere facility map annex','Map annex A places facility F-27 inside Northmere.',`Annex A is signed and effective for the protected archive rule; coordinate record C-27 matches facility F-27.`,['mapA']),record('map-b','Eastmere facility map annex','Map annex B places the same facility F-27 inside Eastmere.',`Annex B is also marked current for the same coordinate and facility identifier; no corrected survey is attached.`,['mapB']),record('map-register','Boundary authority register','The register lists both map annexes as current and contains no controlling order.',`Boundary review BR-27 is open; the records office found no precedence field, corrected coordinate, or signed determination.`,['precedence'])],[{record:'map-a',text:'Map annex A places facility F-27 inside Northmere.',changes:{boundaryConflict:false,mapB:'Northmere'}},{record:'map-b',text:'Map annex B places the same facility F-27 inside Eastmere.',changes:{boundaryConflict:false,mapA:'Eastmere'}},{record:'map-register',text:'The register lists both map annexes as current and contains no controlling order.',changes:{boundaryConflict:false,precedence:true}}]);
 }
 if(family==='software-root-cause'){
  if(['release-rollback','release-migration-distractor'].includes(mode)){
   const facts={ruleActive:true,authorityClear:true,releaseChanged:true,rollbackRestored:true,capacityHealthy:true,dependencyHealthy:true,dataValid:true};
   return set(facts,[record('deploy-log','Deployment timeline','Build 812 was deployed immediately before the error rate rose; rollback to build 811 restored service.',`Release ID rel-812 is linked to incident INC-${mode}; rollback and recovery timestamps come from the same request cohort.`,['releaseChanged','rollbackRestored']),record('resource-metrics','Capacity and dependency telemetry','Worker utilization stayed below its limit and the identity dependency remained healthy.',`Metrics share the incident window and service region; no autoscaling or dependency restart occurred during the rollback.`,['capacityHealthy','dependencyHealthy']),record('data-check','Persisted record validation','Stored account keys passed checksum and schema validation before and after rollback.',`Validation sampled the affected request IDs; no migration replay or data repair ran in this interval.`,['dataValid'])],[{record:'deploy-log',text:'Build 812 was deployed immediately before the error rate rose; rollback to build 811 restored service.',changes:{rollbackRestored:false}},{record:'resource-metrics',text:'Worker utilization stayed below its limit and the identity dependency remained healthy.',changes:{dependencyHealthy:false}},{record:'data-check',text:'Stored account keys passed checksum and schema validation before and after rollback.',changes:{dataValid:false}}]);
  }
  if(mode==='capacity-scale'){
   const facts={ruleActive:true,authorityClear:true,limitReached:true,scaleRestored:true,versionUnchanged:true,dependencyHealthy:true};
   return set(facts,[record('worker-metrics','Worker pool telemetry','Worker utilization reached the configured ceiling as queue lag increased.',`Metric stream M-44 uses the active pool limit and spans the same minute as incident INC-capacity; no gaps were reported.`,['limitReached']),record('scale-test','Capacity intervention record','Adding workers drained the queue while the service version remained unchanged.',`Scale action S-44 was reversible and its start time precedes queue recovery; request mix remained stable.`,['scaleRestored','versionUnchanged']),record('dependency-check','Dependency health record','Identity and storage dependencies remained healthy throughout the lag interval.',`Independent probes succeeded in the affected region with the same endpoint versions used by Atlas.`,['dependencyHealthy'])],[{record:'worker-metrics',text:'Worker utilization reached the configured ceiling as queue lag increased.',changes:{limitReached:false}},{record:'scale-test',text:'Adding workers drained the queue while the service version remained unchanged.',changes:{scaleRestored:false}},{record:'scale-test',text:'Adding workers drained the queue while the service version remained unchanged.',changes:{versionUnchanged:false}}]);
  }
  if(mode==='dependency-fallback'){
   const facts={ruleActive:true,authorityClear:true,dependencyErrorsFirst:true,localHealthy:true,fallbackRestored:true,versionUnchanged:true};
   return set(facts,[record('dependency-trace','Directory dependency trace','Directory timeouts precede Atlas request failures in the linked trace IDs.',`Trace IDs T-81 were sampled from failed requests; their local request spans show the dependency call first.`,['dependencyErrorsFirst']),record('local-probes','Local service probes','Atlas local health checks passed and worker capacity remained available.',`Probe interval and pool metrics cover the same region and minute range as trace set T-81.`,['localHealthy']),record('fallback-test','Fallback intervention','Cached identity fallback restored the affected requests without a build change.',`Feature flag F-22 routed only the sampled requests through cache; recovery was followed by dependency restoration.`,['fallbackRestored','versionUnchanged'])],[{record:'dependency-trace',text:'Directory timeouts precede Atlas request failures in the linked trace IDs.',changes:{dependencyErrorsFirst:false}},{record:'local-probes',text:'Atlas local health checks passed and worker capacity remained available.',changes:{localHealthy:false}},{record:'fallback-test',text:'Cached identity fallback restored the affected requests without a build change.',changes:{fallbackRestored:false}}]);
  }
  if(mode==='data-replay'){
   const facts={ruleActive:true,authorityClear:true,migrationWroteInvalid:true,rollbackPersists:true,journalReplayRepairs:true,dependencyHealthy:true};
   return set(facts,[record('migration-log','Schema migration audit','Migration M-7 wrote truncated account keys in the rows linked to the failure.',`Write log, row IDs, and validation failures share migration transaction TX-7; the migration completed before incident onset.`,['migrationWroteInvalid']),record('rollback-check','Rollback and stored-data check','Code rollback did not repair the persisted invalid keys.',`Build version returned to baseline while the same row hashes continued failing validation.`,['rollbackPersists']),record('journal-replay','Verified journal reconstruction','Replay from the signed journal rebuilt the affected rows and cleared the validation errors.',`Journal sequence numbers match the persisted row IDs; no dependency or capacity change occurred during repair.`,['journalReplayRepairs'])],[{record:'migration-log',text:'Migration M-7 wrote truncated account keys in the rows linked to the failure.',changes:{migrationWroteInvalid:false}},{record:'rollback-check',text:'Code rollback did not repair the persisted invalid keys.',changes:{rollbackPersists:false}},{record:'journal-replay',text:'Replay from the signed journal rebuilt the affected rows and cleared the validation errors.',changes:{journalReplayRepairs:false}}]);
  }
  const facts={ruleActive:true,authorityClear:true,incidentJoinVerified:true,releaseSignal:true,dependencySignal:true,isolationTest:false};
  return set(facts,[record('release-hypothesis','Release hypothesis','Build 901 was deployed immediately before elevated errors began.',`Release rel-901 is linked to incident INC-mixed by the deployment ledger; the interval overlaps the first failed request cohort.`,['releaseSignal']),record('dependency-hypothesis','Dependency hypothesis','Directory latency rose during the same request interval as Atlas errors.',`Trace set T-mixed is linked to INC-mixed; local child spans are missing for part of the interval, so timing alone does not distinguish cause.`,['dependencySignal']),record('incident-link','Incident cohort reconciliation','Deployment events, directory traces, and failed requests share incident key INC-mixed and the same affected region.',`The reconciliation used request IDs from the incident ledger; adjacent-region telemetry was excluded.`,['incidentJoinVerified'])],[{record:'release-hypothesis',text:'Build 901 was deployed immediately before elevated errors began.',changes:{releaseSignal:false}},{record:'dependency-hypothesis',text:'Directory latency rose during the same request interval as Atlas errors.',changes:{dependencySignal:false}},{record:'incident-link',text:'Deployment events, directory traces, and failed requests share incident key INC-mixed and the same affected region.',changes:{incidentJoinVerified:false}}]);
 }
 if(family==='records-retention'){
  if(mode==='expired-ordinary'||mode==='released-hold'){
   const facts={ruleActive:true,authorityClear:true,ordinaryClass:true,ageExpired:true,holdClear:true,auditClear:true,disputeClear:true};
   return set(facts,[record('accession','Accession and category record','Accession ledger classifies record R-77 as ordinary operational material.',`Signed accession ID AC-77 matches the archive object and its creation date; protected-series copies have different IDs.`,['ordinaryClass']),record('age','Retention clock calculation','Record R-77 is eight years old, beyond the ordinary seven-year period.',`Creation and cutoff dates were normalized to the schedule’s calendar-year convention and checked against the archive index.`,['ageExpired']),record('restriction-check','Restriction clearance register','Hold, audit, and dispute registers show no active restriction for R-77.',`The search used exact record ID R-77; a department-level clear flag was not used as a substitute.`,['holdClear','auditClear','disputeClear'])],[{record:'accession',text:'Accession ledger classifies record R-77 as ordinary operational material.',changes:{ordinaryClass:false}},{record:'age',text:'Record R-77 is eight years old, beyond the ordinary seven-year period.',changes:{ageExpired:false}},{record:'restriction-check',text:'Hold, audit, and dispute registers show no active restriction for R-77.',changes:{holdClear:false}}]);
  }
  if(mode==='active-hold'||mode==='hold-release'){
   const facts={ruleActive:true,authorityClear:true,recordIdMatch:true,holdActive:mode==='active-hold',holdReleased:mode==='hold-release',ageExpired:true,otherRestrictionsClear:true};
   return set(facts,[record('hold-order','Counsel hold order','Signed hold H-19 names archive record R-19 and remains active.',`The legal registry ties hold H-19 to exact record ID R-19; the release field is empty at the review timestamp.`,['recordIdMatch','holdActive']),record('release-ledger','Hold release ledger',mode==='active-hold'?'No signed release for hold H-19 is recorded.':'Signed release for hold H-19 predates the current disposition review.',`The ledger is keyed by hold number and exact record ID; unrelated case releases are excluded.`,['holdReleased']),record('schedule','Category schedule','The ordinary retention period for this record class has expired.',`Schedule 6 applies to this accession class. Age calculation and schedule start date were independently checked.`,['ageExpired','otherRestrictionsClear'])],[{record:'hold-order',text:'Signed hold H-19 names archive record R-19 and remains active.',changes:{recordIdMatch:false}},{record:'hold-order',text:'Signed hold H-19 names archive record R-19 and remains active.',changes:{holdActive:false}},{record:'release-ledger',text:mode==='active-hold'?'No signed release for hold H-19 is recorded.':'Signed release for hold H-19 predates the current disposition review.',changes:{holdReleased:mode==='active-hold'}}]);
  }
  if(mode==='audit-open'||mode==='customer-dispute'){
   const audit=mode==='audit-open';const facts={ruleActive:true,authorityClear:true,ordinaryClass:true,ageExpired:true,auditOpen:audit,disputeOpen:!audit,recordLinked:true};
   const statement=audit?'Open audit A-31 includes exact record R-31 in its sample; no clearance is recorded.':'Customer dispute D-18 remains open for exact record R-18; no closure date is recorded.';
   return set(facts,[record('category-age','Accession and age record','Record is ordinary-class and its seven-year period has expired.',`Accession key matches the retention index; age is computed from creation date, not migration date.`,['ordinaryClass','ageExpired']),record('restriction','Restriction docket',statement,`Docket status was queried by exact record identifier at the disposition review time; a neighboring case number is not treated as the same record.`,audit?['auditOpen','recordLinked']:['disputeOpen','recordLinked']),record('schedule','Disposition schedule','Expired ordinary records may be disposed only after the exact-record restriction search is clear.',`Schedule 6 requires the signed disposition log and restriction-clearance timestamp.`,['auditOpen','disputeOpen'])],[{record:'category-age',text:'Record is ordinary-class and its seven-year period has expired.',changes:{ageExpired:false}},{record:'restriction',text:statement,changes:audit?{auditOpen:false}:{disputeOpen:false}},{record:'restriction',text:statement,changes:{recordLinked:false}}]);
  }
  const facts={ruleActive:true,authorityClear:true,classA:'ordinary',classB:'protected',scheduleA:7,scheduleB:10,precedence:false,age:8,holdClear:true,auditClear:true,disputeClear:true};
  return set(facts,[record('schedule-a','Ordinary schedule revision','Schedule 6 assigns seven years to ordinary records.',`Schedule 6 is signed and current for archive unit AR-2; no later ordinary-series change appears in its register.`,['classA','scheduleA']),record('schedule-b','Protected-series schedule revision','Schedule 9 assigns ten years to protected research records.',`Schedule 9 is also marked current for the same accession code because its category field is disputed.`,['classB','scheduleB']),record('schedule-register','Category precedence register','The accession correction log does not resolve whether R-44 is ordinary or protected.',`Both schedules appear current for R-44; there is no signed reclassification or precedence entry.`,['precedence']),record('restriction-check','Exact-record restriction lookup','Exact record R-44 has no active hold, audit, or dispute.',`The legal, audit, and customer registers were checked by record ID R-44 at the disposition date.`,['holdClear','auditClear','disputeClear'])],[{record:'schedule-a',text:'Schedule 6 assigns seven years to ordinary records.',changes:{classB:'ordinary',precedence:'A'}},{record:'schedule-b',text:'Schedule 9 assigns ten years to protected research records.',changes:{classA:'protected',precedence:'B'}},{record:'schedule-register',text:'The accession correction log does not resolve whether R-44 is ordinary or protected.',changes:{precedence:'A'}}]);
 }
 return m;
}

export function resolveResearchCase(family,facts){
 if(!facts||facts.ruleActive!==true||facts.authorityClear!==true)return null;
 if(facts.scheduleConflict||facts.amendmentConflict||facts.allocationConflict||facts.boundaryConflict||facts.categoryConflict)return null;
 if(family==='incident-triage'){
  if(facts.sensorValid===false)return null;
  if(facts.confirmed===true&&facts.emergency===true&&facts.occupied===true&&facts.staffed===true)return 'critical';
  if(facts.confirmed===true&&facts.emergency===true)return 'urgent';
  if(facts.symptoms===true&&facts.exposureVerified===true)return 'urgent';
  if(facts.confirmed===true&&facts.actionBand===true&&facts.occupied===true&&facts.staffed===true)return 'urgent';
  if(facts.confirmed===false&&facts.symptoms===false)return 'routine';
  return null;
 }
 if(family==='procurement-exceptions'){
  if(facts.precedence==='A')return facts.localAddress==='Northmere'&&facts.amount<=facts.thresholdA?'eligible':'ineligible';
  if(facts.precedence==='B')return facts.depotAddress==='Northmere'&&facts.amount<=facts.thresholdB?'eligible':'ineligible';
  const standard=facts.standardLocal===true&&facts.standardComplete===true&&facts.standardUnderCap===true;
  const safety=facts.safetyCritical===true&&facts.waiverSigned===true&&facts.noEquivalent===true;
  const warehouse=facts.warehouseClass===true&&facts.depotActive===true&&facts.warehouseUnderCap===true;
  if(standard||safety||warehouse)return 'eligible';
  if(facts.routeReviewComplete===true&&[facts.standardLocal,facts.standardComplete,facts.standardUnderCap,facts.safetyCritical,facts.waiverSigned,facts.noEquivalent,facts.warehouseClass,facts.depotActive,facts.warehouseUnderCap].every(x=>x!==undefined))return 'ineligible';
  return null;
 }
 if(family==='study-effect'){
  if(facts.outcomeValid===false)return null;
  if(facts.intervalSpansDecisionBounds===true||facts.allocationConflict===true)return 'inconclusive';
  if((facts.randomAssignment===true||facts.protocolRandom===true)&&facts.allocationValid===true&&facts.outcomeValid===true&&facts.intervalSupports===true&&!facts.selectionConfound)return 'causal-evidence';
  if(facts.interventionAssigned===false&&facts.selfSelection===true&&facts.outcomeValid===true)return 'association-only';
  if(facts.intervalSpansDecisionBounds===false&&facts.allocationConflict===false)return 'association-only';
  return null;
 }
 if(family==='policy-scope'){
  if(facts.scopeChecklistComplete===false)return null;
  if(facts.protectedRecord===true&&(facts.insideNorthmere===true||(facts.remoteException===true&&facts.exceptionEffective===true&&facts.keyNorthmere===true)))return 'applies';
  if(facts.protectedRecord===false||facts.scopeChecklistComplete===true)return 'does-not-apply';
  return null;
 }
 if(family==='software-root-cause'){
  if(facts.incidentJoinVerified===false)return null;
  if(facts.isolationTest===true){
   if(facts.testWinner==='release')return 'release-regression';
   if(facts.testWinner==='capacity')return 'capacity-exhaustion';
   if(facts.testWinner==='dependency')return 'dependency-failure';
  }
  if(facts.releaseChanged===true&&facts.rollbackRestored===true&&facts.capacityHealthy===true&&facts.dependencyHealthy===true&&facts.dataValid===true)return 'release-regression';
  if(facts.limitReached===true&&facts.scaleRestored===true&&facts.versionUnchanged===true&&facts.dependencyHealthy===true)return 'capacity-exhaustion';
  if(facts.dependencyErrorsFirst===true&&facts.localHealthy===true&&facts.fallbackRestored===true&&facts.versionUnchanged===true)return 'dependency-failure';
  if(facts.migrationWroteInvalid===true&&facts.rollbackPersists===true&&facts.journalReplayRepairs===true&&facts.dependencyHealthy===true)return 'data-corruption';
  const signals=[['releaseSignal','release-regression'],['capacitySignal','capacity-exhaustion'],['dependencySignal','dependency-failure']].filter(([key])=>facts[key]===true);
  if(signals.length===1)return signals[0][1];
  return null;
 }
 if(family==='records-retention'){
  if(facts.recordLinked===false)return null;
  if(facts.precedence==='A'){facts={...facts,categoryConflict:false,ordinaryClass:true,protectedClass:false,ageExpired:true};}
  if(facts.precedence==='B'){facts={...facts,categoryConflict:false,ordinaryClass:false,protectedClass:true,ageExpired:false};}
  if(facts.holdActive===true&&facts.recordIdMatch===true&&!facts.holdReleased)return 'legal-hold';
  if(facts.auditOpen===true||facts.disputeOpen===true)return 'retain';
  if(facts.protectedClass===true&&facts.ageExpired===false)return 'retain';
  if(facts.ordinaryClass===true&&facts.ageExpired===true&&facts.holdClear===true&&facts.auditClear===true&&facts.disputeClear===true)return 'eligible-for-disposal';
  if(facts.holdReleased===true&&facts.ageExpired===true&&facts.otherRestrictionsClear===true)return 'eligible-for-disposal';
  if(facts.ordinaryClass===true&&facts.ageExpired===false)return 'retain';
  return null;
 }
 throw new Error(`unknown research family ${family}`);
}

const families = [
 ['incident-triage',['critical-release','urgent-action','routine-contained','conflict']],
 ['procurement-exceptions',['local-approved','safety-award','ineligible-closed-routes','conflict']],
 ['study-effect',['randomized-effect','observational-selection','null-interval','conflict']],
 ['policy-scope',['protected-onsite','ordinary-outside','remote-key-unapproved','conflict']],
 ['software-root-cause',['release-rollback','capacity-scale','dependency-fallback','conflict']],
 ['records-retention',['expired-ordinary','active-hold','audit-open','conflict']],
];

export function buildLandscapeCase(family, mode, suffix='') {
 const base=factsFor(family,mode), facts=structuredClone(base.facts), docs=structuredClone(base.records);
 for(const d of docs)d.id+=suffix;
 const gold=resolveResearchCase(family,facts);
 if(!labels[family].includes(gold) && gold!==null)throw Error(`resolver returned unlisted label ${family}/${mode}: ${gold}`);
 const pivotal=[];
 for(const pivot of base.pivots){
  const altered={...facts,...pivot.changes};
  const changed=resolveResearchCase(family,altered);
  if(changed!==gold)pivotal.push({sourceId:pivot.record+suffix,text:pivot.text,changedLabel:changed});
 }
 if(pivotal.length<1)throw Error(`case needs an outcome-pivotal source change: ${family}/${mode}`);
 const ruleId=`governing-rule${suffix}`;
 docs.unshift(record(ruleId,'Governing decision rule',ruleTexts[family],`Fictional ruleset ${family}; revision 4, effective for the case date.`,['ruleActive','authorityClear']));
 const allowedEvidence=[{sourceId:ruleId,text:ruleTexts[family]},...new Map(pivotal.map(row=>[JSON.stringify([row.sourceId,row.text]),{sourceId:row.sourceId,text:row.text}])).values()];
 const catalog=docs.map(({id,title,kind})=>({id,title,kind:kind||'record'}));
 const packet={disclaimer:'Fictional, project-generated case. Apply only the supplied fictional rule set.',domain:family,
  question:`Determine the classification for case ${sha(family+':'+mode).slice(0,10)}. Return null and unresolved=true when the retrieved records do not determine the result.`,
  allowedLabels:labels[family],catalog};
 const expected={kind:'research-classification',label:gold,unresolved:gold===null,
  requiredEvidence:allowedEvidence,documents:docs.map(({id,text})=>({id,text}))};
 return {packet,expected,docs,facts,pivotal};
}

const target={kind:'improvement-case',entry:'solve.nl',source:{schema:'natlang.research-landscape/2',id:'fictional-multidocument-research-v2'},files:{'solve.nl':`---\nargs: { packet: string }\nreturns: string\n---\nThe packet contains a fictional case brief, allowed classification IDs, and a catalog of source IDs. Retrieve the governing rule and every decision-relevant record using research.search(query) and research.read(sourceId). Join records by their case, entity, and effective-date identifiers. Do not infer missing facts from catalog titles. Choose one exact allowed label only when the sources resolve it; otherwise return label null and unresolved true. Cite source IDs with contiguous evidence copied from retrieved documents. Return JSON only: {"label":"allowed ID or null","unresolved":true,"citations":[{"sourceId":"...","evidence":"..."}]}.\n`}};

function serviceSource(documents){
 const data=JSON.stringify(documents.map(({id,title,text,kind})=>({id,title,text,kind:kind||'record'})));
 return `type ResearchHit = { id: string; title: string; kind: string };\nconst PRIVATE_DOCUMENTS = ${data};\nexport function search(query: string): ResearchHit[] { const words=query.toLowerCase().split(/[^\\p{L}\\p{N}]+/u).filter(word=>word.length>2); return PRIVATE_DOCUMENTS.map(document=>({document,score:words.filter(word=>(document.title+' '+document.text).toLowerCase().includes(word)).length})).filter(row=>row.score>0).sort((a,b)=>b.score-a.score||a.document.id.localeCompare(b.document.id)).slice(0,12).map(row=>({id:row.document.id,title:row.document.title,kind:row.document.kind})); }\nexport function read(sourceId: string): string { const document=PRIVATE_DOCUMENTS.find(row=>row.id===sourceId); if(!document)throw new Error('Unknown research source ID'); return document.text; }`;
}

export function buildResearchLandscapeEpisodes(){
 const episodes=[];
 for(const [family,modes] of families){
  const rows=modes.map((mode,i)=>({mode,...buildLandscapeCase(family,mode,`-${family}-${i}`)}));
  for(const variant of variants)for(let replica=0;replica<3;replica++){
   const make=(row,split)=>{
    const packet={...row.packet,caseId:sha(`${family}:${row.mode}:${replica}:${variant}`).slice(0,12)};
    return {id:`${family}-${row.mode}-r${replica}-${variant}`,group:`research-v2/${family}/${row.mode}`,
     args:[JSON.stringify(packet)],expected:row.expected,services:{research:serviceSource(row.docs)}};
   };
   const support=rows.slice(0,2).map(row=>make(row,'support'));
   const query=rows.slice(2).map(row=>make(row,'query'));
   const skillText=`---\nname: research-classifier\ndescription: ${variant==='description-only'?'Retrieve fictional policy sources and classify cases.':'Miscellaneous notes.'}\n---\n${variant==='helpful-body-poor-description'?'Search for the governing rule and linked case facts. Read the source records, check dates and exceptions, and cite exact supporting text. Return unresolved when records conflict.':''}\n`;
   const library=variant==='empty'?{kind:'empty',skills:{}}:{kind:'existing',skills:{'research-classifier':{'SKILL.md':skillText}}};
   episodes.push({version:'natlang.skill-episode/1',id:`landscape-${family}-${variant}-r${replica}`,family:`research-landscape:${family}`,split:'train',
    source_groups:rows.map(row=>`research-v2/${family}/${row.mode}`),license:'project-generated',target,library,
    support:{cases:support},query:{cases:query},operations:['create','revise'],limits:{maxSteps:6},
    provenance:{generator:'natlang.research-landscape/2',metric,replica,variant,fictional_seed:true,
     query_holdout:modes.slice(2),quality_note:'Controlled synthetic seed; not a measured benchmark. Required evidence includes only rule and facts whose alteration changes or unresolveds the host-derived outcome.'}});
  }
 }
 return episodes;
}

export function auditLandscapeConstruction(){
 const checks=[];
 for(const [family,modes] of families)for(const mode of modes){
  const c=buildLandscapeCase(family,mode);
  if(!c.expected.requiredEvidence.length||c.pivotal.length<1)throw Error(`insufficient decision evidence: ${family}/${mode}`);
  checks.push({family,mode,label:c.expected.label,unresolved:c.expected.unresolved,documents:c.docs.length,pivotal:c.pivotal.length,pivotalOutcomes:c.pivotal.map(p=>p.changedLabel),required:c.expected.requiredEvidence.length});
 }
 return checks;
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const at=process.argv.indexOf('--out'),out=at>=0?process.argv[at+1]:undefined;
 if(!out||out.startsWith('--'))throw Error('Usage: build-research-landscape-v2.mjs --out DIR');
 const episodes=buildResearchLandscapeEpisodes(),audit=auditLandscapeConstruction();mkdirSync(out,{recursive:true});
 const body=episodes.map(e=>JSON.stringify(e)).join('\n')+'\n',path=join(out,'research-landscape-v2.jsonl');
 const fd=openSync(path,'wx');try{writeFileSync(fd,body)}finally{closeSync(fd)}
 const manifest={schema:'natlang.research-landscape/2',episodes:episodes.length,case_appearances:episodes.reduce((n,e)=>n+e.support.cases.length+e.query.cases.length,0),unique_scenarios:audit.length,
  cases_per_episode:{support:2,query:2},families:families.map(x=>x[0]),split_groups:Object.fromEntries(families.map(([family,modes])=>[family,{support:modes.slice(0,2),query:modes.slice(2)}])),
  group_appearances:{support:episodes.length/6,query:episodes.length/6},audit,sha256:sha(body),model_calls:0,
  note:'Controlled synthetic seeds only. Six families, four unique scenarios per family. Each scenario appears in nine condition/replica episodes on its original side of the support/query split. Host-only references and raw dossier documents are not included in the target prompt; retrieval is through a typed search/read service.'};
 const mf=openSync(join(out,'manifest.json'),'wx');try{writeFileSync(mf,JSON.stringify(manifest,null,2)+'\n')}finally{closeSync(mf)}
 process.stdout.write(JSON.stringify(manifest)+'\n');
}
