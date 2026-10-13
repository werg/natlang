/** Exact authored-fixture invariant verifier. Story policy remains in the NL application. */
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {storyEvidence} from '../../../../applications/witness-rewrite-studio/host/storyEvidence.ts';
const bag = values => JSON.stringify([...values].sort());
export function verifyResult(result, oracle, snapshot, request) {
  const failures=[];const require=(ok,code)=>{if(!ok)failures.push(code);};
  require(result?.baseRevision===oracle.baseRevision,'base-revision');
  require(result?.requestId===oracle.requestId,'request-identity');
  require(bag(result?.selectedProposalIds??[])===bag(oracle.selectedProposalIds),'selected-proposal-identity');
  const edits=result?.edits??[];
  require(edits.length===oracle.canonical_edits.length,'edit-cardinality');
  for(const expected of oracle.canonical_edits){
    const actual=edits.filter(e=>e.sourceId===expected.sourceId&&e.sourcePath===expected.sourcePath);
    require(actual.length===1,`edit-identity:${expected.sourceId}`);
    if(actual.length===1)for(const key of ['sceneId','oldQuote','replacement'])require(actual[0][key]===expected[key],`edit-${key}:${expected.sourceId}`);
  }
  require(bag(result?.unchangedSceneIds??[])===bag(oracle.unchangedSceneIds),'unchanged-scene-set');
  require(bag(result?.preservedUncertaintyIds??[])===bag(oracle.preservedUncertaintyIds),'protected-uncertainty-set');
  const coverage=result?.obligationCoverage??[];
  require(bag(coverage.map(c=>c.obligationId))===bag(oracle.required_obligation_ids),'obligation-coverage-set');
  require(coverage.every(c=>c.status==='satisfied'&&Array.isArray(c.sourceIds)&&c.sourceIds.length>0&&c.sourceIds.every(id=>snapshot.records.some(r=>r.id===id))),'grounded-satisfied-coverage');
  for(const [candidateId,required] of Object.entries(oracle.required_rejections)){
    const rejected=(result?.rejectedProposals??[]).filter(r=>r.candidateId===candidateId);
    require(rejected.length===1,`rejection-identity:${candidateId}`);
    require(required.every(id=>rejected[0]?.violatedObligationIds?.includes(id)),`rejection-obligations:${candidateId}`);
  }
  require((result?.unresolved??[]).length===0,'unresolved-complete-packet');
  require((result?.sourceChecks??[]).length>0&&result.sourceChecks.every(c=>c.valid===true),'reported-source-checks');
  require((result?.consumedLocalDecisionIds??[]).length>=5&&new Set(result.consumedLocalDecisionIds).size===result.consumedLocalDecisionIds.length,'local-decision-ledger');
  require(storyEvidence.checkEdits(snapshot,edits,request.allowedEditPaths).every(c=>c.valid),'exact-patch-checks');
  require(storyEvidence.checkWitnesses(snapshot,edits.flatMap(e=>e.witnesses??[])).every(c=>c.valid),'exact-original-edit-witnesses');
  return {ok:failures.length===0,failures,semantic_witness_entailment:'requires independent review',live_literal_contribution:'requires actual full-project trace',training_admission:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href){
  if(process.argv.length!==3)throw Error('usage: node verify-result.mjs RESULT_JSON');
  const base=new URL('./harbor-retcon/',import.meta.url);
  const [result,oracle,snapshot,request]=await Promise.all([process.argv[2],new URL('oracle.json',base),new URL('snapshot.json',base),new URL('request.json',base)].map(async p=>JSON.parse(await readFile(p,'utf8'))));
  const verdict=verifyResult(result,oracle,snapshot,request);console.log(JSON.stringify(verdict));if(!verdict.ok)process.exitCode=1;
}
