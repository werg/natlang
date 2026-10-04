import test from 'node:test';
import assert from 'node:assert/strict';
import { buildContractNliEpisodes } from '../scripts/skills/build-contractnli-episodes.mjs';
import { scoreContractNliObjective } from '../src/skills/contractnli-objective.ts';

const hypotheses = Object.fromEntries(Array.from({length:17},(_,i)=>[`nda-${i+1}`,{short_description:`Clause ${i+1}`,hypothesis:`The agreement contains term ${i+1}.`}]));
const longText = (prefix) => `${prefix} establishes the obligations of the parties, the permitted uses of confidential information, and the treatment of materials after a request or termination.`;
function doc(id, firstText, split='train') {
  const secondText=longText(`Contract ${id} states a different specific clause`);
  const text=`${firstText}\n${secondText}`;
  const spans=[[0,Array.from(firstText).length],[Array.from(firstText).length+1,Array.from(text).length]];
  const annotations=Object.fromEntries(Object.keys(hypotheses).map((hyp,i)=>{
    const choice=['Entailment','Contradiction','NotMentioned'][(Number(id)+i)%3];
    return [hyp,{choice,spans:choice==='NotMentioned'?[]:[(Number(id)+i)%2]}];
  }));
  return {id:Number(id),file_name:`${split}-${id}.txt`,text,document_type:'sec-text',url:`https://example.test/${split}/${id}`,spans,
    annotation_sets:[{annotations}]};
}
function data(docs){return {documents:docs,labels:hypotheses};}
function fixture(){
  const shared=longText('The parties agree to protect and restrict confidential information under the agreement');
  const train=Array.from({length:18},(_,i)=>doc(i,i<2?shared:longText(`Train ${i} distinct contractual language`)));
  train[5].text=`😀 ${train[5].text}`;
  const prefixPoints=Array.from('😀 ').length;
  train[5].spans=train[5].spans.map(([start,end])=>[start+prefixPoints,end+prefixPoints]);
  const dev=[doc(100,shared,'dev'),doc(101,longText('Development contract wording'),'dev')];
  const test=[doc(200,longText('Protected test contract wording'),'test'),doc(201,longText('Another protected test agreement'),'test')];
  return {train:data(train),dev:data(dev),test:data(test)};
}
function buildFixture(){
  const f=fixture(), inputHashes={train:'t',dev:'d',test:'x'};
  return buildContractNliEpisodes({...f,inputHashes,sourceRevision:'revision',archiveSha256:'archive',sourceManifestSha256:'source-manifest'});
}

test('ContractNLI scorer grades exact choices and source-verified evidence without trusting model claims',()=>{
  const packet={schema:'natlang.contractnli-task/1',document:{id:'1',span_ids:['0','1']},hypotheses:[{id:'nda-1',text:'A'},{id:'nda-2',text:'B'}]};
  const expected={kind:'contract-nli-classification',scope:'single-supplied-contract',available_span_ids:['0','1'],hypotheses:{
    'nda-1':{choice:'Entailment',evidence_alternatives:[['1'],['0','1']]},'nda-2':{choice:'NotMentioned',evidence_alternatives:[]}}};
  const right={annotations:{'nda-1':{choice:'entailment',span_ids:[1,0]},'nda-2':{choice:'not mentioned',span_ids:[]}}};
  assert.equal(scoreContractNliObjective(packet,right,expected).quality,1);
  const missingEvidence={annotations:{'nda-1':{choice:'ENTAILMENT',span_ids:[]},'nda-2':{choice:'NotMentioned'}}};
  assert.ok(scoreContractNliObjective(packet,missingEvidence,expected).quality<1);
  const fabricated={annotations:{'nda-1':{choice:'Entailment',span_ids:[99]},'nda-2':{choice:'NotMentioned',span_ids:[]}}};
  assert.equal(scoreContractNliObjective(packet,fabricated,expected).gates.valid_response,false);
  const wrong={annotations:{'nda-1':{choice:'Contradiction',span_ids:[1]},'nda-2':{choice:'NotMentioned',span_ids:[]}}};
  assert.ok(scoreContractNliObjective(packet,wrong,expected).quality<1);
});

test('episodes use train contracts only, preserve protected dev/test IDs, and close repeated paragraphs globally',()=>{
  const {episodes,lineage,held,protectedEvaluation,audit}=buildFixture();
  assert.ok(episodes.length>1);
  assert.equal(audit.train_documents,18);
  assert.ok(held.some(row=>row.document_id==='0'&&row.reason==='shared_source_paragraph_with_original_dev_or_test'));
  const supportGroups=new Set(episodes.flatMap(ep=>ep.support.cases.map(row=>row.group)));
  const queryGroups=new Set(episodes.flatMap(ep=>ep.query.cases.map(row=>row.group)));
  assert.equal([...supportGroups].some(group=>queryGroups.has(group)),false);
  for(const ep of episodes){
    assert.ok(ep.support.cases.length<=6&&ep.query.cases.length<=3);
    assert.ok(new Set(ep.support.cases.map(row=>row.group)).size>=2);
    assert.deepEqual(ep.provenance.metric,{schema:'natlang.skill-contractnli/1',kind:'contract-nli-classification'});
    for(const row of [...ep.support.cases,...ep.query.cases]){
      const packet=JSON.parse(row.args[0]);
      assert.equal(packet.hypotheses.length,17);
      assert.equal(Object.hasOwn(packet,'annotations'),false);
      assert.equal(Object.hasOwn(packet,'host_only_oracle'),false);
      assert.doesNotMatch(row.services.research,/"annotations"|"choice"|"evidence_alternatives"/u);
    }
  }
  const emittedIds=new Set([...episodes.flatMap(ep=>[...ep.support.cases,...ep.query.cases].map(row=>row.id.split(':').at(-1))),...held.map(row=>row.document_id)]);
  assert.equal(lineage.length,audit.train_cases_emitted);
  assert.ok(protectedEvaluation.original_role_preserved);
  assert.deepEqual(protectedEvaluation.original_splits.test.documents.map(row=>row.document_id),['200','201']);
  assert.ok(episodes.every(ep=>ep.query.cases.every(row=>!['100','101','200','201'].includes(row.id.split(':').at(-1)))));
});

test('source evidence IDs are checked against exact source span indexes',()=>{
  const {episodes}=buildFixture();
  const row=episodes[0].support.cases.find(item=>Object.values(item.expected.hypotheses).some(x=>x.evidence_alternatives.length));
  const packet=JSON.parse(row.args[0]);
  const annotations=Object.fromEntries(Object.entries(row.expected.hypotheses).map(([id,gold])=>[id,{choice:gold.choice,span_ids:gold.evidence_alternatives[0]??[]} ]));
  assert.equal(scoreContractNliObjective(packet,{annotations},row.expected).quality,1);
});
